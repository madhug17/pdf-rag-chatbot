# Day 4 - Task 1: Extract text from PDF

from pypdf import PdfReader

print("="*70)
print("TASK 1: EXTRACT TEXT FROM PDF")
print("="*70)

# Load PDF - YOUR CASE STUDY PDF!
pdf_path = "data/ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf"

print(f"\n📄 Loading: {pdf_path}")

try:
    reader = PdfReader(pdf_path)
    
    print(f"✅ Loaded!")
    print(f"   Pages: {len(reader.pages)}")
    
    # Extract from Page 1
    print("\n" + "="*70)
    print("PAGE 1 TEXT:")
    print("="*70)
    
    page1 = reader.pages[0]
    text1 = page1.extract_text()
    
    print(text1)
    
    # Extract from Page 2 (if exists)
    if len(reader.pages) > 1:
        print("\n" + "="*70)
        print("PAGE 2 TEXT:")
        print("="*70)
        
        page2 = reader.pages[1]
        text2 = page2.extract_text()
        
        print(text2)
    
    # Combine all pages
    print("\n" + "="*70)
    print("COMBINED TEXT FROM ALL PAGES:")
    print("="*70)
    
    full_text = ""
    for page_num, page in enumerate(reader.pages, 1):
        print(f"\n📄 Extracting page {page_num}...")
        full_text += page.extract_text() + "\n\n"
    
    print(f"\n✅ Extracted {len(full_text)} characters total")
    print("\nFirst 300 characters:")
    print(full_text[:300] + "...")
    
    print("\n" + "="*70)
    print("🎉 TASK 1 COMPLETE!")
    print("="*70)
    
except FileNotFoundError:
    print(f"\n❌ Error: Could not find {pdf_path}")
    print("\nMake sure:")
    print("1. The PDF is in the data/ folder")
    print("2. The filename is correct (check spelling!)")
    print("\nCurrent files in data/:")
    import os
    if os.path.exists("data"):
        for file in os.listdir("data"):
            print(f"   - {file}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")