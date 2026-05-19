import ollama
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)

def is_gpu_available():
    """
    Checks if a compatible GPU (NVIDIA CUDA) is available on the system.
    """
    # 1. Try importing torch and checking CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("[GPU Detector] PyTorch reports CUDA is available.")
            return True
    except ImportError:
        pass

    # 2. Fall back to querying nvidia-smi directly
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if res.returncode == 0 and "GPU" in res.stdout:
                logger.info("[GPU Detector] nvidia-smi successfully detected an active GPU.")
                return True
        except Exception:
            pass

    logger.info("[GPU Detector] No active GPU detected. Falling back to CPU configuration.")
    return False

def preload_model(model_name):
    """
    Pre-loads the specified model into Ollama memory to avoid latency on the first request.
    """
    try:
        logger.info(f"Pre-loading Ollama model '{model_name}'...")
        gpu_active = is_gpu_available()
        options = {'num_gpu': -1} if gpu_active else {'num_gpu': 0}
        logger.info(f"Pre-loading model using options: {options}")
        
        # Pre-load model by calling generate with empty prompt and keep_alive=-1 (keep in memory indefinitely)
        ollama.generate(model=model_name, keep_alive=-1, options=options)
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
    
    gpu_active = is_gpu_available()
    options = {'num_gpu': -1} if gpu_active else {'num_gpu': 0}
    logger.info(f"Generating answer using Ollama model options: {options}")
    
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=options
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error calling Ollama with model '{model_name}': {e}")
        # Return a fallback response
        return f"Error: Failed to generate answer using local model '{model_name}'. Make sure Ollama is running and '{model_name}' is installed."
