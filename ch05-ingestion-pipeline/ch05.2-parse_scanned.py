# parse_scanned.py -- Compare text-based vs scanned PDF parsing
import fitz
from pathlib import Path

sample_dir = Path("data/corpus")

for filename in ["enterprise_sla.pdf", "enterprise_sla_scanned.pdf"]:
    path = sample_dir / filename
    if not path.exists():
        print(f"Skipping (not found): {path}")
        continue

    doc = fitz.open(str(path))
    pages = [doc[i].get_text() for i in range(len(doc))]
    full_text = "\n\n".join(pages)
    doc.close()

    label = "scanned" if "scanned" in filename else "text-based"
    print(f"Parsing: {path} ({label})")
    print(f"Pages: {len(pages)}")
    print(f"Extracted: {len(full_text.strip()):,} chars")
    print()
