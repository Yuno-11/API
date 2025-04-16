from rest_framework import serializers
from .models import modelpredict, ESP32Data
import base64


class predictserializer(serializers.ModelSerializer):
    class Meta:
        model = modelpredict
        fields = ['image_id', 'pk', 'image', 'predict_class', 'predict_accuracy', 'predicted']

    def create(self, validated_data):
        print(f"Validated Data Before Saving: {validated_data}")  # Debug
        instance = modelpredict.objects.create(**validated_data)
        print(f"Created Instance: {instance}")  # Debug
        return instance

class ESP32DataSerializer(serializers.ModelSerializer):
    plant = serializers.JSONField()

    class Meta:
        model = ESP32Data
        fields = [
            'device_id', 'username', 'email', 'user_password',
            'plant', 'predicted', 'timestamp'
        ]
        read_only_fields = ['timestamp']

    def validate_plant(self, value):
        required_keys = ['moisture', 'temperature', 'humidity', 'image']
        for key in required_keys:
            if key not in value:
                raise serializers.ValidationError(f"Missing key in plant data: {key}")

        # Optional: Validate image is base64
        try:
            base64.b64decode(value['image'])
        except Exception as e:
            raise serializers.ValidationError(f"Invalid base64 image: {e}")

        return value