from django.db import models

# Create your models here.

class modelpredict(models.Model):
    image=models.ImageField(upload_to="uploads/")
    predict_class=models.TextField(max_length=100000000000000000,default='')
    predict_accuracy=models.IntegerField(default=0)
    predicted=models.BooleanField(default=False)