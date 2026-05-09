# chunkers.py -- The four chunking strategies from chapter 3, importable from one place.

import re

import numpy as np


def chunk_fixed_size(text, chunk_size=512, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "start": start,
            "end": min(end, len(text)),
            "overlap_start": max(0, start - overlap) if start > 0 else 0,
        })
        start += chunk_size - overlap
    return chunks


def chunk_recursive(text, chunk_size=512, overlap=50, separators=None):
    """Split text recursively, preferring natural boundaries."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text, sep_index):
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        if sep_index >= len(separators):
            return [text[:chunk_size]]

        separator = separators[sep_index]
        if separator == "":
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        parts = text.split(separator)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, sep_index + 1))
                else:
                    current = part
        if current.strip():
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, 0)

    if overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + raw_chunks[i])
        return overlapped
    return raw_chunks


def split_into_sentences(text):
    """Simple sentence splitter using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_similarities(embeddings):
    """Compute cosine similarity between consecutive embeddings."""
    similarities = []
    for i in range(len(embeddings) - 1):
        a = embeddings[i]
        b = embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        similarities.append(cos_sim)
    return similarities


def chunk_semantic(text, model, threshold=0.75, min_chunk_size=100, max_chunk_size=1000):
    """Split text at points where semantic similarity drops."""
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return [text], []

    embeddings = model.encode(sentences)
    similarities = compute_similarities(embeddings)

    chunks = []
    current_chunk = [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

        if len(" ".join(current_chunk)) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks, similarities


def add_context_to_chunk(client, full_document, chunk_text):
    """Generate a context prefix for a chunk using an LLM."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Here is the full document:
<document>
{full_document}
</document>

Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write 1-2 sentences that situate this chunk within the
overall document. Include key context that someone would
need to understand what this chunk is referring to.
Do not summarize the chunk itself, provide the
surrounding context that the chunk lacks."""
        }],
    )
    context_prefix = response.choices[0].message.content
    return f"{context_prefix}\n\n{chunk_text}"


def chunk_markdown(text, max_chunk_size=1500):
    """Split markdown by headers, preserving hierarchy."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        header_match = re.match(r"^(#{1,6})\s+(.+)", section)
        header = header_match.group(2) if header_match else None
        header_level = len(header_match.group(1)) if header_match else 0

        if len(section) <= max_chunk_size:
            chunks.append({
                "text": section,
                "header": header,
                "header_level": header_level,
                "char_count": len(section),
            })
        else:
            subsections = re.split(r"(?=^### )", section, flags=re.MULTILINE)
            for sub in subsections:
                sub = sub.strip()
                if not sub:
                    continue
                sub_header_match = re.match(r"^(#{1,6})\s+(.+)", sub)
                chunks.append({
                    "text": sub,
                    "header": sub_header_match.group(2) if sub_header_match else header,
                    "header_level": (
                        len(sub_header_match.group(1)) if sub_header_match else header_level
                    ),
                    "parent_header": header,
                    "char_count": len(sub),
                })
    return chunks
