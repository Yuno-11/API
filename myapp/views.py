import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework import status

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # Resize for model compatibility
    image = image / 255.0  # Normalize pixel values
    return image

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def put(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # Process Image
        image_bytes = file.read()
        image = read_file_as_image(image_bytes)
        image_batch = np.expand_dims(image, axis=0)

        # AI Prediction
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return Response({
            "class": predicted_class,
            "confidence": confidence
        }, status=status.HTTP_200_OK)
