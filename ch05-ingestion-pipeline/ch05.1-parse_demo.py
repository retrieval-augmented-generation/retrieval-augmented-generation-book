import fitz  # PyMuPDF
from pathlib import Path
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Lightweight HTML-to-text converter (no heavy dependencies required)
# ---------------------------------------------------------------------------
class _HTMLTextExtractor(HTMLParser):
    """Strips tags and returns the visible text content of an HTML document."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self) -> str:
        return "".join(self._pieces).strip()


def html_to_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_pdf(path: Path) -> tuple[int, str]:
    """Return (page_count, full_text) for a PDF."""
    doc = fitz.open(str(path))
    pages = [doc[i].get_text() for i in range(len(doc))]
    doc.close()
    return len(pages), "\n\n".join(pages)


def parse_markdown(path: Path) -> tuple[int, str]:
    """Return (1, text) for a markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return 1, text


def parse_html(path: Path) -> tuple[int, str]:
    """Return (1, text) for an HTML file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return 1, html_to_text(raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path("data/corpus")

PARSERS = {
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".html": parse_html,
}

SAMPLES = sorted(
    f for f in SAMPLE_DIR.glob("*")
    if f.is_file() and f.suffix.lower() in PARSERS and f.name != "create_samples.py"
)


def main():
    for sample in SAMPLES:
        if not sample.exists():
            print(f"Skipping (not found): {sample}")
            continue

        parser = PARSERS.get(sample.suffix.lower())
        if parser is None:
            print(f"Skipping (unsupported format): {sample}")
            continue

        page_count, text = parser(sample)

        print(f"Parsing: {sample}")
        print(f"Pages: {page_count}")
        print(f"First 200 chars:")
        print(text[:200])
        print()


if __name__ == "__main__":
    main()
