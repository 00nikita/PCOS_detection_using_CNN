from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd 
#from .models import model.pkl
#from models import model.pkl
# Create your views here.
import pickle
# Load the model from the pickle file
with open('models\model.pkl', 'rb') as f:
    model1 = pickle.load(f)
model = tf.keras.models.load_model('models\model.h5')

def home(request):
	return render(request,'home.html')


@csrf_exempt
def predict_through_images(request):
    if request.method == 'POST':
        # Get the uploaded image from the request
        img = Image.open(request.FILES['image'])
        # Preprocess the image data
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = img.reshape((1, 224, 224, 3))
        # Make a prediction using the model
        pred = model.predict(img)
        # Convert the prediction result to a human-readable format
        result = {'class': 'positive' if pred[0][0] > 0.5 else 'negative', 'score': str(pred[0][0])}

        return render(request,'result.html',result)
    else:
    	return render(request,'upload.html')

@csrf_exempt
def predict_through_data(request):
    if request.method == 'POST':

        attribute_1 = request.POST.get('attribute_1')
        attribute_2 = request.POST.get('attribute_2')
        attribute_3 = request.POST.get('attribute_3')
        attribute_4 = request.POST.get('attribute_4')
        attribute_5 = request.POST.get('attribute_5')
        attribute_6 = request.POST.get('attribute_6')
        attribute_7 = request.POST.get('attribute_7')
        attribute_8 = request.POST.get('attribute_8')
        attribute_9 = request.POST.get('attribute_9')
        attribute_10 = request.POST.get('attribute_10')
        attribute_11 = request.POST.get('attribute_11')
        attribute_12 = request.POST.get('attribute_12')
        attribute_13 = request.POST.get('attribute_13')
        attribute_14 = request.POST.get('attribute_14')
        attribute_15 = request.POST.get('attribute_15')
        attribute_16 = request.POST.get('attribute_16')
        attribute_17 = request.POST.get('attribute_17')
        attribute_18 = request.POST.get('attribute_18')
        attribute_19 = request.POST.get('attribute_19')
        attribute_20 = request.POST.get('attribute_20')


        pred = model1.predict([[attribute_1,attribute_2,attribute_3,attribute_4,attribute_5,attribute_6,attribute_7,attribute_8,attribute_9,attribute_10,attribute_11,attribute_12,attribute_13,attribute_14,attribute_15,attribute_16,attribute_17,attribute_18,attribute_19,attribute_20]])
        result = {'class': 'positive' if pred==1 else 'negative', 'score': str(pred)}
        return render(request,'result.html',result)
        
    else:
        return render(request,'upload1.html')