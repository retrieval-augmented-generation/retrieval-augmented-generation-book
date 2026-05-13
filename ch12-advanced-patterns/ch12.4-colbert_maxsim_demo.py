# ch12.4-colbert_maxsim_demo.py - Token-level MaxSim worked example.
# A full ColBERT implementation requires a different vector store and is out of
# scope for this chapter. This script demonstrates the scoring idea on a toy
# example so the reader can see how late interaction differs from bi-encoder
# similarity.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embedder import model as embed_model


QUERY = "audit retention"
DOCUMENTS = {
    "A": "audit logs must be retained for seven years",
    "B": "employee records retention policy",
}


def token_vectors(text: str) -> list[tuple[str, np.ndarray]]:
    """Use the bi-encoder model to embed each whitespace token separately. A real
    ColBERT model uses learned per-token contextual vectors; this is a toy proxy
    that lets us illustrate the MaxSim scoring rule without a new dependency."""
    tokens = text.split()
    vectors = embed_model.encode(tokens, show_progress_bar=False)
    return list(zip(tokens, vectors))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Query: {QUERY!r}\n")

    query_tokens = token_vectors(QUERY)

    for doc_name, doc_text in DOCUMENTS.items():
        doc_tokens = token_vectors(doc_text)
        print(f"Document {doc_name}: {doc_text!r}")
        total = 0.0
        for q_tok, q_vec in query_tokens:
            best_score = -1.0
            best_tok = None
            for d_tok, d_vec in doc_tokens:
                score = cosine(q_vec, d_vec)
                if score > best_score:
                    best_score = score
                    best_tok = d_tok
            total += best_score
            print(f"  query token {q_tok!r:<14} best doc match: "
                  f"{best_tok!r:<14} score {best_score:.3f}")
        print(f"  MaxSim total: {total:.3f}")
        print()

    print("The toy MaxSim totals are what late interaction is doing under the")
    print("hood. A real ColBERT system uses contextualised per-token vectors")
    print("from a trained late-interaction model, not the bi-encoder above,")
    print("and its index stores one vector per token rather than one per chunk.")


if __name__ == "__main__":
    main()
