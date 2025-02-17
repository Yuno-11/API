import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt
import threading

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Global flag to indicate if the API is processing an image
processing_flag = False

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # Resize for model compatibility
    image = image / 255.0  # Normalize pixel values
    return image

@csrf_exempt
@api_view(['PUT'])
@parser_classes([MultiPartParser])
def predict(request):
    global processing_flag

    # Check if the flag is True, meaning the API is already processing another image
    if processing_flag:
        return JsonResponse({"error": "API is currently processing another image, please try again later."}, status=400)

    # Set the flag to True, indicating that the API is processing the image
    processing_flag = True

    if request.method != "PUT":
        processing_flag = False  # Reset the flag in case of wrong method
        return JsonResponse({"error": f"Wrong method: {request.method}"}, status=400)

    file = request.FILES.get('file')
    if not file:
        processing_flag = False  # Reset the flag if no file is uploaded
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

        # Return the prediction results
        response = {
            "class": predicted_class,
            "confidence": confidence
        }
    finally:
        # Reset the flag back to False once the processing is complete
        processing_flag = False

    return JsonResponse(response)
