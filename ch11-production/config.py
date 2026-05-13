# config.py - Environment-driven pipeline configuration
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))
CONTEXT_N = int(os.getenv("CONTEXT_N", "5"))

USE_RERANKER = _bool("USE_RERANKER", True)
USE_QUERY_TRANSFORM = _bool("USE_QUERY_TRANSFORM", False)

# Failure injection. Used by ch11.3 to simulate stage outages.
FORCE_RERANKER_FAILURE = _bool("FORCE_RERANKER_FAILURE", False)
FORCE_KEYWORD_FAILURE = _bool("FORCE_KEYWORD_FAILURE", False)


def snapshot() -> dict:
    """Return current effective config. Intended for run-start logging."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "llm_model": LLM_MODEL,
        "candidate_k": CANDIDATE_K,
        "context_n": CONTEXT_N,
        "use_reranker": USE_RERANKER,
        "use_query_transform": USE_QUERY_TRANSFORM,
    }
