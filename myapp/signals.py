import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import esp32_data 

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

@receiver(post_save, sender=esp32_data)
def predict_on_new_data(sender, instance, created, **kwargs):
    """Automatically predict when a new record is added to esp32_data."""
    if created and instance.image:  # Only run when a new record is created
        image_path = instance.image.path  # Get the image file path
        image_array = read_file_as_image(image_path)  # Process the image

        # Run the AI model for prediction
        predictions = MODEL.predict(image_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]  # Get the class
        confidence = float(np.max(predictions))  # Get confidence score

        # Update the instance with prediction results
        instance.predict_class = predicted_class
        instance.predict_accuracy = int(confidence * 100)  # Convert to %
        instance.predicted = True
        instance.save()  # Save the updated record
