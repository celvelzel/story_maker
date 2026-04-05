import PyPDF2
import sys

def extract_text(pdf_path):
    print(f"--- Extracting: {pdf_path} ---")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"

intro_text = extract_text("docs/project/project intro.pdf")
with open("docs/project/project_intro_extracted.txt", "w", encoding="utf-8") as f:
    f.write(intro_text)

spec_text = extract_text("docs/project/COMP5423 NLP Group Project Specification-2026.pdf")
with open("docs/project/spec_extracted.txt", "w", encoding="utf-8") as f:
    f.write(spec_text)

print("Extraction complete. Saved to docs/project/*_extracted.txt")
