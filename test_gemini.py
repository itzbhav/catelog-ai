"""Test Gemini API connection"""
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Testing Gemini API Connection")
print("="*60)

# Check API key
api_key = os.getenv('GOOGLE_API_KEY')
print(f"\n1. API Key Check:")
if api_key:
    print(f"   ✅ Found API key: {api_key[:10]}...{api_key[-5:]}")
else:
    print(f"   ❌ No API key found in .env file!")
    exit(1)

# Configure Gemini
print(f"\n2. Configuring Gemini...")
try:
    genai.configure(api_key=api_key)
    print(f"   ✅ Gemini configured")
except Exception as e:
    print(f"   ❌ Configuration failed: {e}")
    exit(1)

# List available models
print(f"\n3. Available Models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   • {m.name}")
except Exception as e:
    print(f"   ❌ Could not list models: {e}")

# Test generation
print(f"\n4. Testing Content Generation:")
try:
    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content("Say 'Hello, I am working!' in one sentence.")
    print(f"   ✅ Response: {response.text}")
except Exception as e:
    print(f"   ❌ Generation failed: {e}")
    print(f"   Error details: {type(e).__name__}")

print("\n" + "="*60)
