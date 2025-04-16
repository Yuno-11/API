from django.db import models
from django.contrib.postgres.fields import JSONField

class modelpredict(models.Model):
    image_id = models.CharField(max_length=255, null=True,)
    image = models.TextField(max_length=100000000000000000)
    predict_class = models.TextField(max_length=100, default='')  # Stores class name
    predict_accuracy = models.IntegerField(default=0)  # Stores confidence (percentage)
    predicted = models.BooleanField(default=False)

class ESP32Data(models.Model):
    device_id = models.CharField(max_length=100, unique=True)
    username = models.CharField(max_length=100, null=True, blank=True)
    email = models.EmailField(null=True, blank=True)
    user_password = models.CharField(max_length=100, null=True, blank=True)
    plant = models.JSONField()
    predicted = models.BooleanField(default=False)  # AI prediction status
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.device_id} - {'Predicted' if self.predicted else 'Unpredicted'}"
