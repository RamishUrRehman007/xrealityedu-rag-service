import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "models/nano-banana-pro-preview"
prompt = "Generate a photorealistic image of a futuristic city with flying cars."

print(f"Testing image generation with {model_name}...")

try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    
    print("Response received.")
    if response.parts:
        for part in response.parts:
            if part.inline_data:
                print("✅ Image data found (inline_data)!")
                print(f"Mime type: {part.inline_data.mime_type}")
            elif part.text:
                print(f"⚠️ Text response only: {part.text[:100]}...")
            else:
                print(f"Unknown part type: {type(part)}")
    else:
        print("No parts in response.")
        print(response)

except Exception as e:
    print(f"❌ Error: {e}")
