from rest_framework import serializers
from .models import modelpredict, esp32_data

class predictserializer(serializers.ModelSerializer):
    class Meta:
        model = modelpredict
        fields = ['image_id', 'pk', 'image', 'predict_class', 'predict_accuracy', 'predicted']

    def create(self, validated_data):
        print(f"Validated Data Before Saving: {validated_data}")  # Debug
        instance = modelpredict.objects.create(**validated_data)
        print(f"Created Instance: {instance}")  # Debug
        return instance

class ESP32Serializer(serializers.ModelSerializer):
    class Meta:
        model = esp32_data
        fields = ['pk', 'device_id', 'email', 'image', 'temperature', 'humidity', 'predicted_class', 'predict_accuracy', 'predicted']