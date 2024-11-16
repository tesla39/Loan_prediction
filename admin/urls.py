"""learn URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from admin import views
urlpatterns = [
    path('',views.login),
    path('login',views.login),
    path('home',views.index),    
    path('index',views.index),
    path('contact',views.contact),  

    path('login/superuser',views.login_superuser),
    path('logout',views.log_out),
    path('details',views.details),
    path('delete_user/<int:id>',views.delete_user),

    path('loan',views.loan),





    

    
]