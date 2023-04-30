import wikipedia as wiki
import pickle
import math
import os 
import time 
from nltk.corpus import wordnet as wn
import nltk
import numpy as np
from collections import defaultdict,deque
from gensim.parsing.preprocessing import remove_stopword_tokens,strip_punctuation,strip_multiple_whitespaces, strip_non_alphanum, strip_numeric
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from  scipy import sparse


path = os.path.dirname(os.path.abspath(__file__))
TOPICS = ['computer science','physics','football','tennis','esports','music','astronomy','olympics'
          ,'operating system','music bands','oscars','world war 2','university','film','ancient history',
           'machine learning','economy','algorithms','basketball','mma' ]
ADDITIONAL_TOPICS = ['hardware','space','weather','chess','mass culture','water','databases','computational methods']

def save_and_load_wiki_pages(subjects):
    for subject in subjects:
        for article_dict in load_wiki_pages(subject):
            # path = os.path.dirname(os.path.abspath(__file__))
            pickle.dump(article_dict,open(f'{path}/data/{article_dict["title"]}','wb'))

def load_wiki_pages(subject):
    titles = wiki.search(subject,results = 500)
    for title in titles:
        try:
            page = wiki.page(title)
            yield {
                'title': page.title,
                'content': page.content,
                'url': page.url,
            }
        except wiki.exceptions.WikipediaException:
            continue
        except Exception as e:
            print(f'Exception encountered :\n{str(e)}')
    
def create_articles_mapper():
    articles = dict()
    for local_path in read_pickles():
        # print(local_path)
        data = pickle.load(open(local_path,"rb"))
        new_data = {key: val for key,val in data.items() if key != 'title'}
        articles[data['title']] = new_data
    pickle.dump(articles,open(f'{path}/data/utils/ARTICLE_DICT_{len(articles.keys())}','wb'))
    return articles

def create_articles_mapper_non_local(subjects):
    articles = dict()
    counter = 0
    for subject in subjects:
        for data in load_wiki_pages(subject):
            print(f'[ {subject.upper()} ]: number: {counter} title: {data["title"]}')
            articles[data['title']] = {'content':data['content'], 'url':data['url']}
            counter +=1
    pickle.dump(articles,open(f'{path}/data/utils/ARTICLE_DICT_{len(articles.keys())}','wb'))
    return articles


def load_articles():
    return pickle.load(open(f'{path}/data/utils/ARTICLE_DICT_11770',"rb"))

def add_articles(subjects):
    articles = load_articles()
    counter = 0
    for subject in subjects:
        for data in load_wiki_pages(subject):
            print(f'[ {subject.upper()} ]: number: {counter} title: {data["title"]}')
            articles[data['title']] = {
                'content': data['content'], 'url': data['url']}
            counter += 1
    pickle.dump(articles, open(
        f'{path}/data/utils/ARTICLE_DICT_{len(articles.keys())}', 'wb'))
    return articles
    

def read_pickles():
    for doc in os.listdir(os.path.join(path,'data')):
        cand_path = os.path.join(path,'data',doc)
        if not os.path.isdir(cand_path):
            yield cand_path 


class ArticlesParser:
    def __init__(self, articles = None):
        self.articles = articles
        self.proceeded_articles = dict()
        self.unique_words_ids = None
        self.unique_words_collection = None
        self.article_titles = None
        self.word_per_documents_counter = defaultdict(lambda :0)
        self.term_by_document_matrix = None
        self.Ak_matrix = None
        self.Uk = None
        self.Sk = None 
        self.all_words_data = defaultdict(lambda : 0)
        self.lemmatizer = WordNetLemmatizer()

    def run_creation_process(self):
        print("PARSING ARTICLES ...")
        self.parse_articles()
        print("CREATING BAGS OF WORDS FOR EACH DOCUMENT...") 
        self.create_bags_of_words()
        print("CREATING TERM BY DOCUMENT MATRIX...") 
        self.form_term_by_document_matrix()
        print("MAKING IDF MULTIPLICATIONS ...")
        self.make_IDF_multiplications()
        print("NORMALIZING VECTORS ...")
        self.normalize_vectors()
        print("FORMING AK MATRIX ...")
        self.form_AK_matrix(k=100)


    def parse_articles(self):
        self.article_titles = list(self.articles.keys())
        for article_title in self.article_titles:
            self.proceeded_articles[article_title] = dict()
            self.proceeded_articles[article_title]['content'] = self.tokenize_and_lemmatize(self.articles[article_title]['content'])
        self.unique_words_collection = list(filter(lambda x: self.all_words_data[x]>3  ,self.all_words_data.keys()))
        self.unique_words_ids = { self.unique_words_collection[i] : i  for i in range(len(self.unique_words_collection))}

    def create_bags_of_words(self):
        for article in self.article_titles:
            self.proceeded_articles[article]['bow'] = self.to_bag_of_words( self.proceeded_articles[article]['content'])

    def to_bag_of_words(self, word_dict):
        article_word_set = word_dict.keys()
        bag = sparse.dok_matrix(np.zeros((len(self.unique_words_collection),1)))
        for word in article_word_set:
            word_id = self.unique_words_ids.get(word)
            if (word_id==None):
                continue
            bag[word_id,0] = word_dict[word]
        return sparse.csc_matrix(bag)

    def tokenize_and_lemmatize(self,content):
        word_counter =  defaultdict(lambda :0)
        """
            mapper for lemmatization tags ( verbs, nouns, adjectives etc)
            as default i assume noun ( if result flag is not specified here in mapper)
        """
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        """
            content preprocessing
        """
        content = content.lower()
        content = strip_numeric(content)
        content = strip_punctuation(content)
        content = strip_multiple_whitespaces(content)
        content = strip_non_alphanum(content)
        tokens = word_tokenize(content)
        tokens = remove_stopword_tokens(tokens)
        tokens = list(filter(lambda x: len(x)>1, tokens))
        """
            lemmatization
        """
        unique_words = set()
        for token, tag in pos_tag(tokens):
            lemma = self.lemmatizer.lemmatize(token, tag_map[tag[0]])
            word_counter[lemma]+=1
            self.all_words_data[lemma]+=1
            unique_words.add(lemma)
        for word in unique_words:
            self.word_per_documents_counter[word]+=1
        return word_counter

    def form_term_by_document_matrix(self):
        words_no = len(self.unique_words_collection)
        articles_no = len(self.article_titles)
        self.term_by_document_matrix = sparse.lil_matrix((words_no, articles_no))
        articles_title = self.article_titles
        for i in range(articles_no):
            self.term_by_document_matrix[:,i] = self.proceeded_articles[articles_title[i]]['bow']
        # self.term_by_document_matrix = sparse.csc_matrix(self.term_by_document_matrix)

    def form_AK_matrix(self, k):
        print(f'Type of term matrix: {type(self.term_by_document_matrix)}')
        u, s ,v_t = sparse.linalg.svds(self.term_by_document_matrix, k=k)
        print(f"SVD PROCEEDED ...")
        print(f'Non zero elements in u: {np.count_nonzero(u)}, shapes: {u.shape}')
        print(f'Non zero elements in s: {np.count_nonzero(s)}, shapes: {s.shape}')
        print(f'Non zero elements in v_t: {np.count_nonzero(v_t)}, shapes: {v_t.shape}')
        s = np.diag(s)
        print("Started multiplication ...")
        # matrix = u @ s @ v_t
        print("Before Sk multiplication")
        Sk = s @ v_t
        print("After Sk multiplication")
        # print("After multiplication ...")
        # print(f'None zero Akmatrix elemenets : {np.nonzero(matrix)}, shape: {matrix.shape}')
        # self.Ak_matrix = u @ s @ v_t
        # self.Ak_matrix = sparse.csc_matrix(matrix)
        self.Uk = u
        self.Sk = Sk
        # self.Ak_matrix = matrix

    def make_IDF_multiplications(self):
        words = list(self.unique_words_ids.keys())
        n = len(words)
        i=0
        articles_no = len(self.article_titles)
        for word in words:
            print(f"[IDF] {i}/{n}")
            i+=1
            word_id = self.unique_words_ids[word]
            unique_article_occurences = self.word_per_documents_counter[word]
            multip = math.log(articles_no/unique_article_occurences)
            self.term_by_document_matrix[word_id]*=multip 
        self.term_by_document_matrix = sparse.csc_matrix(self.term_by_document_matrix)


    def count_articles_with_word(self,word):
        return sum([1 for article in self.proceeded_articles.keys() if word in self.proceeded_articles[article]['content']]) 

    def normalize_vectors(self):
        n=len(self.proceeded_articles.keys())
        for i in range(n):
            print(f"Iteration : {i}/{n}") 
            vector = self.term_by_document_matrix.getcol(i)
            norm = self.get_vector_norm(vector)
            self.term_by_document_matrix[:,i] /= norm
        

    def get_vector_norm(self, vector):
        val = sparse.linalg.norm(vector)
        # print(f"Norm : {val}")
        return val
        # return math.sqrt(vector.power(2).sum())


    def proceed_query(self, query, article_no, Ak = False ):
        Uk = None 
        Sk = None 
        matrix = None
        if Ak:
            Uk = self.Uk
            Sk = self.Sk
        else:
            matrix = self.term_by_document_matrix
        start = time.time()    
        query_proceeded = self.tokenize_and_lemmatize(query)
        query_bow = self.to_bag_of_words(query_proceeded).transpose()
        vector_norm = self.get_vector_norm(query_bow)
        query_bow /= vector_norm
        probs = deque()
        for i in range(len(self.article_titles)):
            if not Ak:
                article = matrix.getcol(i)
                prod = query_bow @ article
            else:
                Sk_vec = Sk[:,i] 
                prod = query_bow @ Uk @ Sk_vec 
            if not Ak:
                prod = prod[0,0]
            else:
                prod = prod[0]
            doc_cos = prod 
            probs.append((doc_cos,i))
            # sum_val += doc_cos
            # print(f'Iteration :{i}, Probability: {doc_cos}')
        probs = list(probs)
        probs.sort(key = lambda x: x[0], reverse = True)
        print(len(probs))
        print(probs[:article_no])
        # print(f"Sum of probs: {sum_val}")
        print("Articles found by algortihm: ")
        result_articles = []
        for prob, index in probs[:article_no]:
            found = self.article_titles[index]
            print(f'\tARTICLE: {found}, PROBABILITY: {prob}, url: {self.articles[found]["url"]}')
            result_articles.append(
                {"name": found, "probability": prob*100, "url": self.articles[found]["url"]})
        print(f"TOOK : {time.time()-start}")
        return result_articles
    def save(self):
        pickle.dump(self.term_by_document_matrix,open(f'{path}/data/utils/TERM_BY_DOCUMENT_{len(self.article_titles)}','wb'))
        # pickle.dump(self.Ak_matrix,open(f'{path}/data/utils/AK_MATRIX_{len(self.article_titles)}','wb'))
        pickle.dump(self.unique_words_ids,open(f'{path}/data/utils/UNIQUE_WORDS_IDS_{len(self.article_titles)}','wb'))
        pickle.dump(self.unique_words_collection,open(f'{path}/data/utils/UNIQUE_WORDS_COLLECTION_{len(self.article_titles)}','wb'))
        pickle.dump(self.article_titles,open(f'{path}/data/utils/ARTICLE_TITLES_{len(self.article_titles)}','wb'))

    def save_Ak(self):
        # pickle.dump(self.Ak_matrix, open(
        #     f'{path}/data/utils/AK_MATRIX_{len(self.article_titles)}', 'wb'))
        pickle.dump(self.Uk, open(
            f'{path}/data/utils/UK_MATRIX_{len(self.article_titles)}', 'wb'))
        pickle.dump(self.Sk, open(
            f'{path}/data/utils/SK_MATRIX_{len(self.article_titles)}', 'wb'))

    def load_from_pickle(self):
        self.articles = load_articles()
        print("Started unpickling ...")
        # M = pickle.load(open(f'{path}/data/utils/TERM_BY_DOCUMENT_8461',"rb"))
        self.term_by_document_matrix = pickle.load(open(f'{path}/data/utils/TERM_BY_DOCUMENT_11770',"rb"))
        # self.term_by_document_matrix = sparse.csc_matrix(M)
        # print("Unplickling Ak Matrix: ")
        # self.Ak_matrix = pickle.load(open(f'{path}/data/utils/AK_MATRIX_11770', "rb"))
        # print("Unpickled Ak Matrix ... ")
        # ak_M = pickle.load(open(f'{path}/data/utils/AK_MATRIX_11770', "rb"))
        # self.Ak_matrix = sparse.csc_matrix(ak_M)
        self.unique_words_collection = pickle.load(open(f'{path}/data/utils/UNIQUE_WORDS_COLLECTION_11770', "rb"))
        self.unique_words_ids = pickle.load(open(f'{path}/data/utils/UNIQUE_WORDS_IDS_11770', "rb"))
        self.article_titles = pickle.load(open(f'{path}/data/utils/ARTICLE_TITLES_11770', "rb"))
        

        print("Loading Uk ...")
        self.Uk = pickle.load(open(f'{path}/data/utils/UK_MATRIX_11770', 'rb'))
        print("Loaded Uk ...")
        print("Loading Sk ...")
        self.Sk = pickle.load(open(f'{path}/data/utils/SK_MATRIX_11770', 'rb'))
        print("Loaded Sk ...")

        print("END of unpickling ...")


def query_util(query, k):
    print(f'Inside query_util - query: {query}')
    AP = ArticlesParser()
    AP.load_from_pickle()
    results = AP.proceed_query(query, k ,True)
    return results

if __name__=='__main__':
    # save_and_load_wiki_pages(create_articles_mapper_non_local(TOPICS))
    # create_articles_mapper_non_local(TOPICS)
    AP = ArticlesParser()
    AP.load_from_pickle()
    query = "second world war poland"
    print("Results without ak: ")
    AP.proceed_query(query,5,False)
    print('Results with ak:')
    AP.proceed_query(query,5,True)
