from django.apps import AppConfig

class ApiConfig(AppConfig):
    name = 'api'
    def ready(self):
        from core.embedding.clip_model import get_clip
        get_clip()