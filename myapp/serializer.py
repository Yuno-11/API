from rest_framework import serializers
from .models import modelpredict, esp32_data

class predictserializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = modelpredict
        fields = ['pk', 'image', 'predict_class', 'predict_accuracy', 'predicted']

class ESP32Serializer(serializers.ModelSerializer):
    class Meta:
        model = esp32_data
        fields = ['pk', 'device_id', 'email', 'image', 'temperature', 'humidity', 'predict_class', 'predict_accuracy', 'predicted']
