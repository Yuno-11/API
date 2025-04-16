import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.utils import img_to_array # type: ignore
import psycopg2
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .models import modelpredict, ESP32Data
from .serializer import predictserializer, ESP32DataSerializer

# Load Disease Detection Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = None
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"Disease model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading disease model: {e}")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load Leaf Checker Model
LEAF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "leaf_checker_model.h5")
leaf_model = None
try:
    leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH)
    print(f"Leaf checker model loaded from: {LEAF_MODEL_PATH}")
except Exception as e:
    print(f"Error loading leaf checker model: {e}")
leaf_classes = ["Potato Leaf"]

# Function to check if the image is a potato leaf
def is_potato_leaf(img_pil):
    img = img_pil.resize((128, 128))  # Resize for leaf model
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)

    prediction = leaf_model.predict(img_array)[0]
    confidence = round(np.max(prediction) * 100, 2)
    predicted_index = np.argmax(prediction)
    predicted_class = leaf_classes[predicted_index]

    return predicted_class, confidence

# Database Connection
def get_db_connection():
    return psycopg2.connect(
        dbname="railway",
        user="postgres",
        password="yqtCQhtupwWXcUgVJiQeuaUEkxcdmOuR",
        host="turntable.proxy.rlwy.net",
        port="56650"
    )

# Convert base64 to image
def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

# Convert bytea to image
def convert_bytea_to_image(bytea_data):
    return Image.open(BytesIO(bytea_data)).convert("RGB")

# Process image for model prediction
def process_image(image) -> np.ndarray:
    image = image.resize((256, 256))  # Resize image to model's expected input size
    image = np.array(image) / 255.0   # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Predict disease using the model
def predict_disease(image_pil, model, class_names):
    img_resized = image_pil.resize((256, 256))  # Resize image for model input
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(prediction)]  # Class with highest probability
    confidence = int(np.max(prediction) * 100)  # Confidence as percentage
    
    return predicted_class, confidence

# API for Direct Image Uploads & Predictions
@api_view(['GET', 'POST'])
def florai(request):
    if request.method == 'GET':
        data = modelpredict.objects.all()
        serializer = predictserializer(data, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        image_data = request.data.get('image')
        image_id = request.data.get('image_id')

        if not image_data or not image_id:
            return Response({"error": "No image or image_id provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(';base64,')[-1]

            image_pil = base64_to_image(image_data)

            # Check if image is a potato leaf first
            predicted_leaf_class, leaf_confidence = is_potato_leaf(image_pil)

            if leaf_confidence < 60:
                return Response({
                    "error": "The image is not a potato leaf.",
                }, status=status.HTTP_201_CREATED)

            # Proceed with disease prediction
            predicted_class, confidence = predict_disease(image_pil, MODEL, CLASS_NAMES)

            # Save the result to the database
            data = {
                "image_id": image_id,
                "image": image_data,
                "predict_class": predicted_class,
                "predict_accuracy": confidence,
                "predicted": True,
                "leaf_confidence": leaf_confidence
            }
            serializer = predictserializer(data=data)
            if serializer.is_valid():
                serializer.save()

            return Response({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "leaf_confidence": leaf_confidence,
                **serializer.data
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# API for Fetching & Processing ESP32 Images
@api_view(['GET', 'POST'])
def florai_esp32(request):
    if request.method == 'GET':
        data = ESP32Data.objects.all()
        serializer = ESP32DataSerializer(data, many=True) 
        return Response(serializer.data)

    # Handle POST request to receive data from ESP32
    if request.method == 'POST':
        device_id = request.data.get('device_id')
        plant_data = request.data.get('plant')

        if not device_id or not plant_data:
            return Response({"error": "Device ID or plant data is missing"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract image from plant data
            image_data = plant_data.get('image')
            if not image_data:
                return Response({"error": "Image data is missing"}, status=status.HTTP_400_BAD_REQUEST)

            # Process the base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(';base64,')[-1]
            image_pil = base64_to_image(image_data)
            # Check if image is a potato leaf first
            predicted_leaf_class, leaf_confidence = is_potato_leaf(image_pil)

            if leaf_confidence < 60:
                return Response({
                    "error": "The image is not a potato leaf.",
                }, status=status.HTTP_201_CREATED)
            
            # Predict the disease using the model
            predicted_class, confidence = predict_disease(image_pil, MODEL, CLASS_NAMES)

            # Add prediction results to plant_data
            plant_data["predict_class"] = predicted_class
            plant_data["predict_accuracy"] = confidence
            plant_data["predicted"] = True

            # Build full payload
            full_data = {
                "device_id": device_id,
                "plant": plant_data, 
                "predicted": True
            }

            serializer = ESP32DataSerializer(data=full_data)
            if serializer.is_valid():
                serializer.save()
                return Response({
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    **serializer.data
                }, status=status.HTTP_201_CREATED)
            else:
                return Response({"error": "Data Not Valid", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# API for Handling Specific Predictions
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
