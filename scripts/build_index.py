# build_index.py -- Chunk and embed the corpus, saving results to data/embeddings/.
#
# Run once to pre-build the indexes that chapters load instead of re-computing
# embeddings on each run.
#
# Output files:
#   data/embeddings/docs.json           -- one record per document (text + metadata)
#   data/embeddings/doc_embeddings.npy  -- float32 array, shape (D, 1536)
#   data/embeddings/chunks.json         -- one record per chunk (text + metadata)
#   data/embeddings/chunk_embeddings.npy -- float32 array, shape (N, 1536)
#
# Chapter 1 loads the doc-level index (whole documents).
# Chapter 2+ loads the chunk-level index.
#
# Usage:
#   python scripts/build_index.py
#   python scripts/build_index.py --corpus data/corpus --chunk-size 512 --overlap 50

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from docx import Document as DocxDocument
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
SUPPORTED = {".pdf", ".md", ".html", ".htm", ".docx", ".txt"}
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_yaml_front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        import yaml
        meta = yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}, text
    return meta, text[match.end():]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._pieces = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr", "section"):
            self._pieces.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._pieces.append(data)

    def get_text(self):
        return "".join(self._pieces).strip()


class _HTMLMetaExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            d = dict(attrs)
            if d.get("name") and d.get("content"):
                self.meta[d["name"]] = d["content"]


def parse_file(path: Path) -> tuple[str, dict]:
    """Return (text, metadata) for any supported file type."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        meta = {"doc_type": "pdf"}

    elif suffix in (".html", ".htm"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta_ext = _HTMLMetaExtractor()
        meta_ext.feed(raw)
        txt_ext = _HTMLTextExtractor()
        txt_ext.feed(raw)
        text = txt_ext.get_text()
        meta = {k: v for k, v in meta_ext.meta.items()
                if k in ("title", "category", "doc_type", "last_updated", "owner", "classification")}
        meta.setdefault("doc_type", "html")

    elif suffix in (".md", ".txt"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, body = _extract_yaml_front_matter(raw)
        text = body
        meta = {k: v for k, v in front_matter.items()
                if k in ("title", "category", "doc_type", "last_updated", "owner", "classification")}
        meta.setdefault("doc_type", "markdown")

    elif suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        meta = {"doc_type": "docx"}

    else:
        raise ValueError(f"Unsupported: {path.suffix}")

    meta["source"] = path.name
    return text, meta


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def recursive_split(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Paragraph-aware recursive splitter."""
    separators = ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part
        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return chunks

    # Hard split fallback
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, retrying on rate-limit errors."""
    for attempt in range(5):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in response.data]
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  Retrying in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def embed_all(client: OpenAI, texts: list[str]) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_embeddings.extend(embed_batch(client, batch))
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_index(corpus_dir: Path, chunk_size: int, overlap: int):
    client = OpenAI()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in corpus_dir.glob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED and not f.name.startswith(".")
    )
    print(f"Found {len(files)} documents in {corpus_dir}/\n")

    all_docs = []
    all_chunks = []

    for path in files:
        print(f"Parsing: {path.name}")
        text, meta = parse_file(path)
        if not text.strip():
            print(f"  Skipped (empty)")
            continue

        all_docs.append({"text": text, **meta})

        chunks = recursive_split(text, chunk_size=chunk_size, overlap=overlap)
        print(f"  {len(chunks)} chunks")
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": meta["source"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **{k: v for k, v in meta.items() if k != "source"},
            })

    print(f"\nEmbedding {len(all_docs)} documents with {EMBED_MODEL}...")
    doc_embeddings = embed_all(client, [d["text"] for d in all_docs])
    with open(EMBEDDINGS_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    np.save(EMBEDDINGS_DIR / "doc_embeddings.npy", doc_embeddings)
    print(f"Saved {len(all_docs)} docs, embeddings {doc_embeddings.shape}")

    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBED_MODEL}...")
    chunk_embeddings = embed_all(client, [c["text"] for c in all_chunks])
    with open(EMBEDDINGS_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    np.save(EMBEDDINGS_DIR / "chunk_embeddings.npy", chunk_embeddings)
    print(f"Saved {len(all_chunks)} chunks, embeddings {chunk_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the corpus embedding index.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    build_index(args.corpus, args.chunk_size, args.overlap)
