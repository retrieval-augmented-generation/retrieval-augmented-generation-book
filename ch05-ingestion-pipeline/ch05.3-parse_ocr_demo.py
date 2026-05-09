# parse_ocr_demo.py -- Demonstrate OCR fallback on a scanned PDF
import fitz
from pathlib import Path

path = Path("data/corpus/enterprise_sla_scanned.pdf")
if not path.exists():
    print(f"Not found: {path}")
    exit(1)

doc = fitz.open(str(path))
page = doc[0]

# First: try normal text extraction
text_normal = page.get_text()
print(f"Normal extraction: {len(text_normal.strip())} chars")

# Second: OCR fallback (requires Tesseract)
print("Running OCR (this may take a few seconds)...")
try:
    tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
    text_ocr = page.get_text("text", textpage=tp)
    print(f"OCR extraction: {len(text_ocr.strip())} chars")
    print(f"\nFirst 300 chars of OCR output:")
    print(text_ocr.strip()[:300])
except Exception as e:
    print(f"OCR failed: {e}")
    print("Install Tesseract: apt-get install tesseract-ocr")

doc.close()
