# Retrieval-Augmented Generation

**A Hands-On Guide to Grounding LLMs with Your Own Data**

By [Jeroen Herczeg](https://github.com/retrieval-augmented-generation)

This repository contains the code and document corpus for the book *Retrieval-Augmented Generation: A Hands-On Guide to Grounding LLMs with Your Own Data*. Every chapter has a corresponding folder with runnable scripts that build on a single running example — the Acme Corp knowledge base.

## Quick Start

```bash
git clone https://github.com/retrieval-augmented-generation/retrieval-augmented-generation-book.git
cd retrieval-augmented-generation-book
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your API keys
```

## Repository Structure

```
data/
  corpus/              # The Acme Corp document corpus (110 documents)
ch01-the-problem/      # Chapter 1  — The Problem RAG Solves
ch02-embeddings/       # Chapter 2  — Embeddings from First Principles
ch03-chunking/         # Chapter 3  — Chunking Strategies
ch04-vector-storage/   # Chapter 4  — Vector Storage and Indexing
ch05-ingestion/        # Chapter 5  — Building the Ingestion Pipeline
ch06-retrieval/        # Chapter 6  — Retrieval: From Keywords to Semantics
ch07-first-pipeline/   # Chapter 7  — Your First RAG Pipeline
ch08-hybrid-search/    # Chapter 8  — Hybrid Search and Score Fusion
ch09-reranking/        # Chapter 9  — Reranking
ch10-query-transform/  # Chapter 10 — Query Transformation
ch11-evaluation/       # Chapter 11 — Evaluating RAG Systems
ch12-production/       # Chapter 12 — Hardening the Pipeline for Production
ch13-advanced/         # Chapter 13 — Advanced Retrieval Patterns
ch14-agentic/          # Chapter 14 — Agentic RAG
```

Each chapter folder contains the scripts referenced in the book. Chapters build on each other — the ingestion pipeline from chapter 5 creates the index that chapters 6 through 14 retrieve against.

## The Acme Corp Corpus

The running example throughout the book is **Acme Corp**, a fictional 500-employee SaaS company. The `data/corpus/` directory contains ~110 internal documents (HR policies, IT security, operations, compliance, product docs, engineering runbooks) in mixed formats: PDF, Markdown, HTML, DOCX, and scanned PDFs with OCR artifacts.

The corpus is engineered to surface every failure mode the book teaches — exact-reference retrieval targets, cross-document contradictions, vocabulary mismatches, scattered evidence, and questions the corpus deliberately cannot answer.

## Prerequisites

- Python 3.10+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector) extension
- API keys for an LLM provider (OpenAI or Anthropic) and optionally a reranking service (Cohere)

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
DATABASE_URL=postgresql://rag_user:rag_pass@localhost:5432/rag_book
```

## License

The code in this repository is provided for educational purposes to accompany the book. See [LICENSE](LICENSE) for details.
