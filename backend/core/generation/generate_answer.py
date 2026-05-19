import ollama
import logging

logger = logging.getLogger(__name__)

def preload_model(model_name):
    """
    Pre-loads the specified model into Ollama memory to avoid latency on the first request.
    """
    try:
        logger.info(f"Pre-loading Ollama model '{model_name}'...")
        # Pre-load model by calling generate with empty prompt and keep_alive=-1 (keep in memory indefinitely)
        ollama.generate(model=model_name, keep_alive=-1)
        logger.info(f"Successfully pre-loaded Ollama model '{model_name}' into memory.")
    except Exception as e:
        logger.warning(f"Could not pre-load Ollama model '{model_name}': {e}. Ensure Ollama is running and '{model_name}' is installed.")

def generate_answer(model_name, system_prompt, prompt_text, images=None):
    """
    Generates a radiological interpretation using local Ollama model (e.g., qwen2).
    """
    messages = []
    
    # Add system prompt
    if system_prompt:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })
        
    # Add user prompt and attach images if present
    user_message = {
        'role': 'user',
        'content': prompt_text
    }
    
    if images:
        user_message['images'] = images
        
    messages.append(user_message)
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error calling Ollama with model '{model_name}': {e}")
        # Return a fallback response
        return f"Error: Failed to generate answer using local model '{model_name}'. Make sure Ollama is running and '{model_name}' is installed."
