from django.urls import path
from .views import florai,Restflorai

urlpatterns = [
    path('api/', florai, name='predict'),
    path('api/<int:pk>', Restflorai, name='predict')
]
