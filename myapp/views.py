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

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

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
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

# Convert BYTEA to Image (for ESP32 Images)
def convert_bytea_to_image(bytea_data):
    return Image.open(BytesIO(bytea_data)).convert("RGB")

# Read Image for Direct Upload & ESP32 Processing
def process_image(image) -> np.ndarray:
    image = image.resize((256, 256))  # Resize image to model's expected input size
    image = np.array(image) / 255.0   # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# API for Direct Image Uploads & Predictions
@api_view(['GET', 'POST'])
def florai(request):
    if request.method == 'GET':
        data = modelpredict.objects.all()
        serializer = predictserializer(data, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        # Get the image data and image_id from the request
        image_data = request.data.get('image')
        image_id = request.data.get('image_id')

        # Check if the image data or image_id is missing
        if not image_data or not image_id:
            return Response({"error": "No image or image_id provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Check if the image is base64 encoded (handles base64 prefix)
            if image_data.startswith('data:image'):
                image_data = image_data.split(';base64,')[-1]

            # Convert base64 to PIL image
            image = base64_to_image(image_data)

            # Process the image
            image_processed = process_image(image)

            # Make prediction
            predictions = MODEL.predict(image_processed)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = int(np.max(predictions) * 100)

            # Save prediction to the database, including the image_id
            data = {
                "image_id": image_id,  # Store the provided image_id
                "image": image_data,    # Storing the base64 image
                "predict_class": predicted_class,
                "predict_accuracy": confidence,
                "predicted": True,      # Set predicted to True since we now have a result
            }
            serializer = predictserializer(data=data)
            if serializer.is_valid():
                serializer.save()  # Save the prediction result in the database

            # Return serialized data along with prediction results
            return Response({
                "predicted_class": predicted_class,
                "confidence": confidence,
                **serializer.data  # Includes the database-stored fields like image_id and predicted flag
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# API for Fetching & Processing ESP32 Images
@api_view(['GET'])
def florai_esp32(request):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch an unpredicted image
        cursor.execute("SELECT id, device_id, email, image FROM esp32_images WHERE predicted = FALSE LIMIT 1;")
        image_data = cursor.fetchone()

        if not image_data:
            return Response({"message": "No unprocessed images found"}, status=status.HTTP_200_OK)

        image_id, device_id, email, image_bytea = image_data

        # Convert image from BYTEA to Base64
        image_base64 = base64.b64encode(image_bytea).decode('utf-8')

        # Process the image
        image = process_image(convert_bytea_to_image(image_bytea))

        # Make AI prediction
        predictions = MODEL.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = int(np.max(predictions) * 100)

        # Update Database
        cursor.execute("""UPDATE esp32_images SET predict_class = %s, predict_accuracy = %s, predicted = TRUE WHERE id = %s;""",
                       (predicted_class, confidence, image_id))

        conn.commit()
        cursor.close()
        conn.close()

        # Return prediction result with Base64 image
        return Response({
            "id": image_id,
            "device_id": device_id,
            "email": email if email else None,
            "image": image_base64,  
            "class": predicted_class,
            "confidence": confidence,
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
