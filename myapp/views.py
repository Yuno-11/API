import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.conf import settings

# Load AI Model
MODEL_PATH = os.path.join(settings.BASE_DIR, "your_app", "model.h5")  # Adjust path if needed
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess the image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))  # Resize for model input
    image = np.array(image) / 255.0  # Normalize
    return image

# API View for Image Prediction
class PredictView(APIView):
    parser_classes = [MultiPartParser]  # Allow file uploads

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # Read and preprocess image
        image_bytes = file.read()
        image = read_file_as_image(image_bytes)
        image_batch = np.expand_dims(image, axis=0)  # Expand dimensions for model input

        # Make prediction
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # Response with classification result
        return Response({
            "class": predicted_class,
            "confidence": confidence
        }, status=status.HTTP_200_OK)
