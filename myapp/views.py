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
MODEL = tf.keras.models.load_model(MODEL_PATH)
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
    image = image.resize((256, 256))  # Standard resizing
    image = np.array(image) / 255.0   # Convert to NumPy array & normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# API for Direct Image Uploads & Predictions
@api_view(['GET', 'POST'])
def florai(request):
    if request.method == 'GET':
        data = modelpredict.objects.all()
        serializer = predictserializer(data, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        if 'image' not in request.data:
            return Response({"error": "Base64 image is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract Base64 part by removing the MIME type prefix
            base64_str = request.data['image']
            if base64_str.startswith("data:image"):
                base64_str = base64_str.split(",")[1]  

            # Decode Base64 image
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data)).convert('RGB')  # Convert to RGB to avoid errors
            image_array = np.array(image)  # Convert image to array for processing

            # Make AI prediction
            predictions = MODEL.predict(np.expand_dims(image_array, axis=0))  # Ensure correct shape
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = int(np.max(predictions) * 100) 

            # Prepare data for serializer
            prediction_data = {
                "image": request.data['image'],  # Store Base64 string including MIME type
                "predict_class": predicted_class,
                "predict_accuracy": confidence,
                "predicted": True
            }

            # Serialize and save
            serializer = predictserializer(data=prediction_data)
            if serializer.is_valid():
                model_instance = serializer.save()

                # Return serialized data with Base64 image
                response_data = serializer.data
                response_data["image"] = request.data['image']  # Include Base64 image in response

                return Response(response_data, status=status.HTTP_201_CREATED)

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({"error": f"Failed to process image: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
        cursor.execute("""
            UPDATE esp32_images
            SET predict_class = %s, predict_accuracy = %s, predicted = TRUE
            WHERE id = %s;
        """, (predicted_class, confidence, image_id))

        conn.commit()
        cursor.close()
        conn.close()

        # Return prediction result with Base64 image
        return Response({
            "id": image_id,
            "device_id": device_id,
            "email": email if email else None,
            "image": image_base64,  # Return the image as Base64
            "class": predicted_class,
            "confidence": confidence,
            "predicted": True
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
