# check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ No API key found")
    exit(1)

genai.configure(api_key=api_key)

print("🔍 Checking available Gemini models...")

try:
    models = genai.list_models()
    
    # Convert generator to list to count and iterate
    models_list = list(models)
    print(f"📋 Found {len(models_list)} models:")
    
    generate_content_models = []
    
    for model in models_list:
        if 'generateContent' in model.supported_generation_methods:
            generate_content_models.append(model)
            print(f"✅ {model.name}")
            print(f"   Supported methods: {model.supported_generation_methods}")
            print(f"   Input tokens: {getattr(model, 'input_token_limit', 'N/A')}")
            print(f"   Output tokens: {getattr(model, 'output_token_limit', 'N/A')}")
            print()
    
    print(f"🎯 Models supporting generateContent: {len(generate_content_models)}")
    
    if generate_content_models:
        print("\n🚀 Recommended models for your application:")
        for model in generate_content_models:
            if 'gemini-pro' in model.name or 'gemini-1.5' in model.name:
                print(f"   💫 {model.name}")
    
except Exception as e:
    print(f"❌ Error checking models: {e}")