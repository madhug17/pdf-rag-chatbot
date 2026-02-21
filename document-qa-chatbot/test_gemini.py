import os
from dotenv import load_dotenv
import google.generativeai as genai

print("="*60)
print("TESTING GEMINI API CONNECTION")
print("="*60)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ API key not found")
    exit(1)

print("✅ API key found")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    print("\nSending test request to Gemini...")
    response = model.generate_content("Say 'Hello from Gemini!' in one sentence.")
    
    print("\n✅ CONNECTION SUCCESSFUL!")
    print(f"\nGemini Response: {response.text}")
    print(f"\nModel: gemini-1.5-flash")
    
    print("\n" + "="*60)
    print("🎉 GEMINI API IS WORKING!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nCheck:")
    print("1. API key is correct")
    print("2. You have internet connection")
    print("3. API key has free credits")