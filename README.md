# Enterprise RAG & AI Agent Pipeline with Unity Catalog

A production-style Retrieval-Augmented Generation (RAG) system built on **Databricks Free Edition**, showcasing distributed data engineering and governed GenAI infrastructure. This project extends prior RAG work (a local ChromaDB + GPT-4o mini prototype) into a scalable, production-grade pipeline using Unity Catalog, Spark, Vector Search, and MLflow.

## Project Motivation

Most portfolio RAG projects stop at "local script + vector DB." This project instead demonstrates the full production stack a real enterprise GenAI system requires: governed data storage, distributed processing, managed vector search, experiment tracking, and (eventually) an agent layer with tool-calling — all while respecting real infrastructure constraints (serverless-only, CPU-only compute).

## Corpus

**15 SEC 10-K filings** across 5 major tech companies, 3 fiscal years each (2023–2025), sourced directly from SEC EDGAR:

| Company | Ticker | Years |
|---|---|---|
| NVIDIA | NVDA | 2023, 2024, 2025 |
| Apple | AAPL | 2023, 2024, 2025 |
| Microsoft | MSFT | 2023, 2024, 2025 |
| Meta | META | 2023, 2024, 2025 |
| Alphabet (Google) | GOOG | 2023, 2024, 2025 |

Files follow the naming convention `TICKER_YEAR_10K.pdf`. Note: NVIDIA's fiscal year ends in late January rather than December, so `NVDA_2025` corresponds to a period ending January 2025, not calendar year 2025.

## Architecture

```
Raw PDFs (Volume)
      │
      ▼
Bronze: PDF text extraction (pypdf)
      │
      ▼
Silver: Chunking  ──┬── Fixed-size chunking (1000 chars, 150 overlap)
                    └── Section-aware chunking (split on "Item X" headers)
      │
      ▼
Silver: Embeddings (databricks-bge-large-en, 1024-dim)
      │
      ▼
Vector Search: Two Delta Sync Indexes (one per chunking strategy)
      │
      ▼
Retrieval + LLM (Model Serving) → Final Answer
```

Governance is handled throughout via **Unity Catalog** — all tables and volumes live under the `rag_pipeline.main` schema, giving a consistent, permissioned three-level namespace (`catalog.schema.object`) from raw files through to the vector index.

## Tech Stack

- **Compute**: Databricks Free Edition, serverless only (no GPU support)
- **Storage & Governance**: Unity Catalog (Volumes for raw files, Delta tables for structured data)
- **Processing**: PySpark, including `pandas_udf` for distributed embedding calls
- **Embeddings**: Databricks-hosted foundation model endpoint (`databricks-bge-large-en`, 1024 dimensions)
- **Vector Search**: Databricks AI Search (formerly "Vector Search"), Delta Sync Indexes, Hybrid index type
- **Experiment Tracking**: MLflow (chunking strategy comparison)
- **Version Control**: GitHub

## Repository Structure

```
enterprise-rag-unity-catalog/
├── README.md
├── notebooks/
│   ├── 01_ingestion.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_vector_search_setup.ipynb
│   ├── 05_agent.ipynb            (in progress)
│   └── 06_evaluation.ipynb       (in progress)
├── src/
├── config/
├── docs/
└── data/
    └── sample_filings/
```

## Notebook-by-Notebook Breakdown

### `01_ingestion.ipynb` — Bronze Layer
Reads all 15 PDFs from the Unity Catalog Volume `rag_pipeline.main.raw_documents`, extracts raw text using `pypdf`, and parses `company` and `fiscal_year` metadata directly from each filename. Writes the result to `rag_pipeline.main.bronze_10k_filings` — one row per filing, with columns for file path, company, fiscal year, raw extracted text, and character count (used as a sanity check on extraction quality).

**Result**: 15 rows, all filings extracted cleanly (200K–540K characters each, no failures).

### `02_chunking.ipynb` — Silver Layer (Chunking)
Splits each filing's raw text into retrieval-sized passages, implementing **two competing strategies** to compare later:

- **Fixed-size chunking** (`chunking_strategy = "fixed_size_1000"`): mechanically splits text into 1000-character chunks with 150-character overlap (to avoid severing facts at chunk boundaries). Written to `rag_pipeline.main.silver_chunks_fixed_size`.
- **Section-aware chunking** (`chunking_strategy = "section_aware"`): splits first on 10-K "Item X" section headers (regex-based), then sub-chunks any section still over 1000 characters using the same fixed-size logic. Written to `rag_pipeline.main.silver_chunks_section_aware`.

Both strategies apply a `MIN_CHUNK_LENGTH = 30` filter to remove near-empty junk fragments.

**Result**: 6,457 clean fixed-size chunks (avg 998 chars) vs. 7,197 clean section-aware chunks (avg 870 chars). Section-aware chunking produced significantly more noise before filtering — 124 junk fragments (1.7% of raw chunks) caused by the section-header regex false-matching on table-of-contents entries and in-text cross-references (e.g., "see Item 8" appearing mid-sentence), versus only 2 junk fragments (0.03%) for fixed-size.

### `03_embeddings.ipynb` — Silver Layer (Embeddings)
Generates a 1024-dimension vector embedding for every chunk in both Silver tables, using Databricks' hosted `databricks-bge-large-en` foundation model endpoint. Uses a `pandas_udf` to distribute embedding calls across Spark partitions, with an internal batching layer (20 texts per API call) to respect endpoint request limits. Writes results to `rag_pipeline.main.silver_embeddings_fixed_size` and `rag_pipeline.main.silver_embeddings_section_aware`.

**Result**: 6,457 and 7,197 embedded rows respectively, matching chunk counts exactly with no dropped rows.

### `04_vector_search_setup.ipynb` — Vector Search Indexing
Enables Change Data Feed on both embedding tables (a prerequisite for Delta Sync Indexes), then creates two Hybrid-type Delta Sync Indexes on a single shared AI Search endpoint (`rag_pipeline_endpoint`) — one per chunking strategy:

- `rag_pipeline.main.fixed_size_index`
- `rag_pipeline.main.section_aware_index`

Both indexes use pre-computed embeddings (rather than having Databricks recompute them), with `chunk_id` as the primary key and Triggered sync mode (appropriate for this static corpus). Includes retrieval test queries confirming both indexes return relevant, correctly-filtered results.

### `05_evaluation.ipynb` — Chunking Strategy Comparison
Runs a fixed set of 8 test queries — spanning single-document lookup, cross-year comparison, and cross-company comparison query types — against both indexes, logging all retrieved chunks and similarity scores to `rag_pipeline.main.eval_retrieval_results`.

**Key finding**: Fixed-size and section-aware chunking produced nearly identical retrieval quality (avg top-1 similarity: 0.6429 fixed-size vs. 0.6399 section-aware), with fixed-size holding a small, consistent edge across every query type. This is best explained by the fact that most content-dense 10-K sections (Item 1A Risk Factors, Item 7 MD&A) are long enough to be sub-chunked identically by both strategies regardless of the initial split method — meaning the two approaches converge in practice for a document type like this. **Given comparable retrieval quality and meaningfully lower noise, fixed-size chunking was selected as the primary strategy for the production pipeline.**

### `05_agent.ipynb` — Agent & Model Serving *(in progress)*
Will implement the retrieval → prompt construction → LLM generation loop using Databricks Model Serving, built on top of the fixed-size index selected above.

## Setup Notes & Constraints

- Databricks Free Edition is a **permanent** free tier (not a time-limited trial), including one Unity Catalog metastore, one AI Search endpoint (one search unit), CPU-only Model Serving, and serverless-only compute.
- Outbound internet from notebooks is restricted to a trusted-domain allowlist — external documents (e.g., SEC EDGAR filings) must be downloaded locally and uploaded through the UI rather than fetched directly from a notebook.
- Free Edition is intended for non-commercial use, which this portfolio project satisfies.

