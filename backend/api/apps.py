import os
from django.apps import AppConfig

class ApiConfig(AppConfig):
    name = 'api'
    def ready(self):
        # Preload text/image medical embedding models
        from core.embedding.medical_models import get_medical_models
        get_medical_models()

        # Preload Ollama local LLM model
        from core.generation.generate_answer import preload_model
        ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2')
        preload_model(ollama_model)

        # Preload BM25 sparse retrieval index
        from core.retrieval.bm25_index import get_bm25_index
        get_bm25_index()

        # Preload cross-encoder reranker
        from core.retrieval.cross_encoder_reranker import get_cross_encoder
        get_cross_encoder()