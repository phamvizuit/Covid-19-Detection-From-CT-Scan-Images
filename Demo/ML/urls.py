from django.urls import path
from . import views
urlpatterns = [
    path('',views.home, name="Home" ),
    path('prd',views.predict,name="predict")
]
