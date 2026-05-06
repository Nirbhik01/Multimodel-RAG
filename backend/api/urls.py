from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('getResponse', views.GetResponse),
    path('conversations', views.ListConversations),
    path('conversations/<str:conversation_id>', views.GetConversation),
    path('conversations/<str:conversation_id>/delete', views.DeleteConversation),
    path('conversations/create', views.CreateConversation),
]