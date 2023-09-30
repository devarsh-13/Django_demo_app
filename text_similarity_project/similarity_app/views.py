from django.shortcuts import render
# similarity_app/views.py
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from similarity_app.utils  import load_data, train_word2vec, calculate_similarity
import json

@csrf_exempt
@require_POST
def train_word2vec_view(request):
    data = load_data('similarity_app/Text_Similarity_Dataset.csv')
    word2vec_model = train_word2vec(data)
    joblib.dump(word2vec_model, 'word2vec_model.pkl')
    return JsonResponse({'message': 'Word2Vec model trained and saved successfully.'})

@csrf_exempt
@require_POST
def calculate_similarity_view(request):
    data = json.loads(request.body)
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    word2vec_model = joblib.load('word2vec_model.pkl')
    similarity = calculate_similarity(text1, text2, word2vec_model)

    similarity = float(similarity)

    return JsonResponse({'similarity_score': similarity})

# Create your views here.
