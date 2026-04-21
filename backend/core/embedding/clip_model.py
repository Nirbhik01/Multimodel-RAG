from transformers import CLIPProcessor, CLIPModel

_model = None
_processor = None

def get_clip():
    global _model, _processor

    if _model is None:
        model_id = "openai/clip-vit-large-patch14"
        _model = CLIPModel.from_pretrained(model_id)
        _processor = CLIPProcessor.from_pretrained(model_id)

    return _model, _processor