
from django.contrib import admin
from django.urls import path
from similarity_app.views import train_word2vec_view, calculate_similarity_view

urlpatterns = [
    path('train_word2vec/', train_word2vec_view, name='train_word2vec'),
    path('calculate_similarity/', calculate_similarity_view, name='calculate_similarity'),
]