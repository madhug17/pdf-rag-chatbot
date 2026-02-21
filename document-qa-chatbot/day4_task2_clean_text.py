# 📦 External library (not built-in)
from pypdf import PdfReader  

# 📦 Built-in Python regex module
import re  

# ✅ Built-in print()
print("Task 2: Clean Extracted Text")
print("=" * 70)

pdf_path = "data/ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf"

# 📦 PdfReader() → library class
reader = PdfReader(pdf_path)

raw_text = ""

# Loop through all pages
for page in reader.pages:
    # 🧠 extract_text() → object method
    raw_text += page.extract_text() + "\n"

# ✅ len() → built-in
print(f"Extracted {len(raw_text)} characters")

print("\n" + "=" * 70)
print(raw_text[:500])   # Show first 500 characters


# ------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------

def clean_text(text):
    # 📦 re.sub() → regex library function
    
    # Remove multiple spaces
    text = re.sub(r' +', " ", text)

    # Reduce 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove space after newline
    text = re.sub(r'\n ', '\n', text)

    # Remove space before newline
    text = re.sub(r' \n', '\n', text)

    # ✅ strip() → built-in string method
    text = text.strip()

    return text


print("\nCleaning text...")

clean_text_result = clean_text(raw_text)

# Correct length comparison
print(f"Before: {len(raw_text)} characters")
print(f"After: {len(clean_text_result)} characters")
print(f"Removed: {len(raw_text) - len(clean_text_result)} characters")

print("\nCleaned Preview:")
print(clean_text_result[:500])

print("\nSample Slice (100-200):")
print(clean_text_result[100:200])
