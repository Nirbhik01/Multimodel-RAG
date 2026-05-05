from django.apps import AppConfig

class ApiConfig(AppConfig):
    name = 'api'
    def ready(self):
        from core.embedding.medical_models import get_medical_models
        get_medical_models()