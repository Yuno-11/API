from rest_framework import serializers
from .models import modelpredict

class predictserializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = modelpredict
        fields = ['pk', 'image', 'predict_class', 'predict_accuracy', 'predicted']
