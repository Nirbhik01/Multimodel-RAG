from google import genai
import os

'''
    from dotenv import load_dotenv
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # This is only needed if you run this module by itself
'''

def generate_answer(gemini_model, GEMINI_API_KEY, contents):

    system_prompt = """
    You are an expert Radiologist AI Assistant. You specialize in analyzing chest X-rays and interpreting clinical findings.
    
    You will be provided with:
    1. A Current Patient's X-ray image and their preliminary query/observations.
    2. A Reference X-ray image and its corresponding Report (Findings and Impression) from a similar case.
    
    Your Task:
    - Compare the current Patient's image with the Reference image.
    - Evaluate the patient's query against both the visual evidence and the reference case.
    - Provide a professional, concise, and accurate Radiology Report for the Current Patient.
    - Structure your output clearly with 'Findings' and 'Impression' sections.
    - Highlight any significant similarities or differences between the current case and the reference case that aided your analysis.
    """
    
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Prepend the system prompt to the context
    full_contents = system_prompt + contents

    response = client.models.generate_content(
        model=gemini_model,
        contents=full_contents
    )
    return response.text