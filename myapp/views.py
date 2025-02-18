import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # Resize for model compatibility
    image = image / 255.0  # Normalize pixel values
    return image

@csrf_exempt
@api_view(['PUT', 'GET'])  # Allow both PUT and POST
@parser_classes([MultiPartParser])
def predict(request):
    file = request.FILES.get('file')
    if not file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    # Process Image
    image_bytes = file.read()
    image = read_file_as_image(image_bytes)
    image_batch = np.expand_dims(image, axis=0)

    try:
        # AI Prediction
        predictions = MODEL.predict(image_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JsonResponse({"class": predicted_class, "confidence": confidence})
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
