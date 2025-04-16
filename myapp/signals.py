import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import ESP32Data
import base64

# Load AI Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(image_path) -> np.ndarray:
    """Read an image from a file path and preprocess it for model input."""
    image = Image.open(image_path).convert("RGB")  # Convert to RGB format
    image = image.resize((256, 256))  # Resize for model input
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

def decode_image(image_data):
    """Decode base64 image data and return the PIL Image."""
    if image_data.startswith("data:image"):
        # Remove the base64 prefix (e.g., "data:image/png;base64,")
        header, encoded = image_data.split(",", 1)
        image_data = base64.b64decode(encoded)
    else:
        # If no prefix, decode directly
        image_data = base64.b64decode(image_data)

    # Load the image from the decoded data
    try:
        image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")  # Convert to RGB format
        image = image.resize((256, 256))  # Resize for model input
        return np.array(image) / 255.0  # Normalize pixel values
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

@receiver(post_save, sender=ESP32Data) 
def predict_on_new_data(sender, instance, created, **kwargs):
    """Automatically predict when a new record is added to esp32_data."""
    if created and instance.plant:
        plant_data = instance.plant  # This will be a JSON object

        image_data = plant_data.get('image', None)  # Adjust based on your structure
        
        if image_data:
            image_array = decode_image(image_data)
            if image_array is not None:
                # Run the AI model for prediction
                predictions = MODEL.predict(np.expand_dims(image_array, axis=0))
                predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Get the class
                confidence = float(np.max(predictions))  # Get confidence score

                # Update the instance with prediction results
                instance.predict_class = predicted_class
                instance.predict_accuracy = int(confidence * 100)  # Convert to %
                instance.predicted = True
                instance.save()  # Save the updated record