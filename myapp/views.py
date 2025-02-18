import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http.response import JsonResponse,HttpResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.decorators import api_view
from rest_framework import status,filters
from .models import modelpredict
from .serializer import predictserializer
# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    """Read image data and preprocess for model input."""
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # Resize for model
    image = image / 255.0  # Normalize pixel values
    return image

@api_view(['GET','POST'])
def florai(request):
    if request.method == 'GET':
        data=modelpredict.objects.all()
        serializer=predictserializer(data,many=True)
        return Response(serializer.data)
    if request.method == 'POST':
        serializer=predictserializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_201_CREATED)
    return Response(serializer.data,status=status.HTTP_400_BAD_REQUEST)
@api_view(['GET','PUT','PATCH','DELETE'])
def Restflorai(request,pk):
    try:
        data = modelpredict.objects.get(pk=pk)
    except status.DoesNotExist:
        return HttpResponse(status=status.HTTP_404_NOT_FOUND)
    if request.method == 'GET':
        serializer = predictserializer(data)
        return Response(serializer.data)
    if request.method == 'PUT':
        serializer = predictserializer(data, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    if request.method == 'PATCH':
        serializer = predictserializer(data, data=request.data,partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    if request.method == "DELETE":
        data.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)