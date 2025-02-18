from django.urls import path
from .views import florai,Restflorai, florai_esp32

urlpatterns = [
    path('api/', florai, name='predict'),
    path('api/esp32/', florai_esp32, name='florai_esp32'),
    path('api/<int:pk>', Restflorai, name='predict')
]
