import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http.response import JsonResponse, HttpResponse
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import api_view
from rest_framework import status
from .models import modelpredict
from .serializer import predictserializer

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    """Read image data and preprocess for model input."""
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3-channel RGB
    image = image.resize((256, 256))  # Resize for model input
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@api_view(['GET', 'POST'])
def florai(request):
    if request.method == 'GET':
        data = modelpredict.objects.all()
        serializer = predictserializer(data, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        if 'image' not in request.FILES:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image'].read()
        image_array = read_file_as_image(image_file)
        
        # Make AI prediction
        predictions = MODEL.predict(image_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Save prediction to the database
        serializer = predictserializer(data={"image": request.FILES['image'], "prediction": predicted_class, "confidence": confidence})
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
def Restflorai(request, pk):
    try:
        data = modelpredict.objects.get(pk=pk)
    except modelpredict.DoesNotExist:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = predictserializer(data)
        return Response(serializer.data)

    if request.method in ['PUT', 'PATCH']:
        serializer = predictserializer(data, data=request.data, partial=(request.method == 'PATCH'))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    if request.method == "DELETE":
        data.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
