from django.db import models

class modelpredict(models.Model):
    image_id = models.IntegerField(default=0)
    image = models.TextField(max_length=100000000000000000)
    predict_class = models.TextField(max_length=100, default='')  # Stores class name
    predict_accuracy = models.IntegerField(default=0)  # Stores confidence (percentage)
    predicted = models.BooleanField(default=False)

class esp32_data(models.Model):
    device_id = models.CharField(max_length=100, unique=True)
    email = models.EmailField(blank=True, null=True)
    image = models.TextField(max_length=100000000000000000)
    temperature = models.FloatField()
    humidity = models.FloatField()
    predict_class = models.CharField(max_length=50, blank=True, null=True)
    predict_accuracy = models.IntegerField(blank=True, null=True)
    predicted = models.BooleanField(default=False)

    def __str__(self):
        return f"ESP32 Device {self.device_id}"
