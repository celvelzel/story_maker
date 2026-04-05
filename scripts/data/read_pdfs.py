import PyPDF2
import sys

def extract_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

print("=== SPECIFICATION ===")
print(extract_text("docs/project/COMP5423 NLP Group Project Specification-2026.pdf"))
print("\n=== INTRO ===")
print(extract_text("docs/project/project intro.pdf"))
