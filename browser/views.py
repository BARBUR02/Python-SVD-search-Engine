from django.shortcuts import render
from django.http import HttpResponseRedirect
from wiki_parser.document_loader import query_util
# Create your views here.


def redirect_view(request):
    return HttpResponseRedirect('browser')


def browser_view(request):
    context = {}
    if request.method == 'POST':
        print(request)
        data = request.POST
        query = data.get('search')
        print(f'Query: {query}')
        if query=="":
            result = []
        else:
            result = query_util(query, 5)
        context = {"articles" : result}
    if request.method == 'GET':
        print("GET method")
    return render(request,"browser/browser.html", context)