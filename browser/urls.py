from django.urls import path
from . import views

urlpatterns = [
    path('', views.redirect_view),
    path('browser', views.browser_view, name='search-base')
]
