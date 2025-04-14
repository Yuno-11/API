import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import psycopg2
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .models import modelpredict
from .serializer import predictserializer
import base64
from keras.utils import load_img, img_to_array # type: ignore

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
    leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH)  # Ensure leaf_model is loaded here
    print(f"Leaf checker model loaded from: {LEAF_MODEL_PATH}")
except Exception as e:
    print(f"Error loading leaf checker model: {e}")
leaf_classes = ["Potato Leaf"]

# Function to check if image is a potato leaf
def is_potato_leaf(img_pil):
    img = img_pil.resize((128, 128))  # Resize for leaf model
    img_array = img_to_array(img) / 255.0  # Updated to use img_to_array
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

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def convert_bytea_to_image(bytea_data):
    return Image.open(BytesIO(bytea_data)).convert("RGB")

def process_image(image) -> np.ndarray:
    image = image.resize((256, 256))  # Resize image to model's expected input size
    image = np.array(image) / 255.0   # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

def base64_to_image(base64_string):
    """
    Convert a base64 encoded string to a PIL Image object.

    Args:
        base64_string (str): The base64 encoded image string.

    Returns:
        PIL.Image.Image: The image object.
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def predict_disease(image_pil, model, class_names):
    # Preprocess the image for disease model
    img_resized = image_pil.resize((256, 256))  # Resize image to model's expected input size
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict disease class
    prediction = model.predict(img_array)
    
    # Get predicted class and confidence
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
                    "leaf_confidence": leaf_confidence
                }, status=status.HTTP_400_BAD_REQUEST)

            # Proceed with disease prediction using the new predict_disease function
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
@api_view(['GET'])
def florai_esp32(request):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, device_id, email, image FROM esp32_images WHERE predicted = FALSE LIMIT 1;")
        image_data = cursor.fetchone()

        if not image_data:
            return Response({"message": "No unprocessed images found"}, status=status.HTTP_200_OK)

        image_id, device_id, email, image_bytea = image_data
        image_base64 = base64.b64encode(image_bytea).decode('utf-8')
        image_pil = convert_bytea_to_image(image_bytea)

        # Optional: Apply potato leaf check here if needed
        predicted_leaf_class, leaf_confidence = is_potato_leaf(image_pil)
        if predicted_leaf_class != "Potato Leaf" or leaf_confidence < 60:
            cursor.execute("""
                UPDATE esp32_images
                SET predict_class = %s,
                    predict_accuracy = %s,
                    predicted = TRUE,
                    leaf_class = %s,
                    leaf_confidence = %s
                WHERE id = %s;
            """, ("Not Potato Leaf", leaf_confidence, predicted_leaf_class, leaf_confidence, image_id))
            conn.commit()
            cursor.close()
            conn.close()
            return Response({
                "id": image_id,
                "device_id": device_id,
                "email": email if email else None,
                "image": image_base64,
                "class": "Not Potato Leaf",
                "confidence": leaf_confidence,
                "leaf_class": predicted_leaf_class,
                "leaf_confidence": leaf_confidence,
                "predicted": True
            }, status=status.HTTP_200_OK)

        # Run disease detection
        image_processed = process_image(image_pil)
        predictions = MODEL.predict(image_processed)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = int(np.max(predictions) * 100)

        cursor.execute("""
            UPDATE esp32_images
            SET predict_class = %s,
                predict_accuracy = %s,
                predicted = TRUE,
                leaf_class = %s,
                leaf_confidence = %s
            WHERE id = %s;
        """, (predicted_class, confidence, predicted_leaf_class, leaf_confidence, image_id))
        conn.commit()
        cursor.close()
        conn.close()

        return Response({
            "id": image_id,
            "device_id": device_id,
            "email": email if email else None,
            "image": image_base64,
            "class": predicted_class,
            "confidence": confidence,
            "leaf_class": predicted_leaf_class,
            "leaf_confidence": leaf_confidence,
            "predicted": True
        }, status=status.HTTP_200_OK)

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
