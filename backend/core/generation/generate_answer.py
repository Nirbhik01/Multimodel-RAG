from google import genai
import os

'''
    from dotenv import load_dotenv
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # This is only needed if you run this module by itself
'''

def generate_answer(gemini_model , GEMINI_API_KEY, context):
    client = genai.Client( api_key = GEMINI_API_KEY )

    response = client.models.generate_content(
        model = gemini_model, contents = context
    )
    return response.text