import os
from dotenv import load_dotenv

print("="*60)
print("TESTING .ENV FILE")
print("="*60)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print(f"\n✅ SUCCESS!")
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"Length: {len(api_key)} characters")
    
    if api_key.startswith("AIza"):
        print("✅ Format correct (starts with AIza)")
    
    print("\n" + "="*60)
    print("🎉 .ENV FILE IS WORKING!")
    print("="*60)
else:
    print("\n❌ ERROR: API key not found")
