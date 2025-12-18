import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "models/nano-banana-pro-preview"
try:
    model_info = genai.get_model(model_name)
    print(f"Model: {model_name}")
    print(f"Supported Methods: {model_info.supported_generation_methods}")
except Exception as e:
    print(f"Error checking {model_name}: {e}")

# Also check for explicit image generation models
print("\n--- Image Generation Models ---")
for m in genai.list_models():
    if "image" in m.name.lower() or "generateContent" in m.supported_generation_methods:
        print(f"{m.name}: {m.supported_generation_methods}")
