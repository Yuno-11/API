import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import ESP32Data

# Load AI model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_image_from_base64(image_base64: str) -> np.ndarray:
    """Decode a base64 image and prepare it for model prediction."""
    header, encoded = image_base64.split(",", 1)  # Remove 'data:image/jpeg;base64,'
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@receiver(post_save, sender=ESP32Data)
def predict_on_new_data(sender, instance, created, **kwargs):
    if created:
        plant_data = instance.plant
        image_base64 = plant_data.get("image")
        
        if image_base64:
            try:
                image_array = read_image_from_base64(image_base64)
                predictions = MODEL.predict(image_array)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]
                confidence = float(np.max(predictions))

                # Update instance
                instance.plant["predicted_class"] = predicted_class
                instance.plant["predict_accuracy"] = int(confidence * 100)
                instance.predicted = True
                instance.save()

            except Exception as e:
                print("Prediction error:", str(e))
