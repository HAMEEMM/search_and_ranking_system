# Search Engineering Coding Assessment – Test 6

## Overview

A complete Python implementation of a simplified search platform that processes healthcare-related documents and search logs. The system implements 10 core search engineering components.

## Requirements

- **Python 3.7+** (standard library only — no external dependencies)

## Project Structure

```
engineering_test_6/
├── search_and_ranking_system.py   # Main implementation (all 10 tasks)
├── datasets/
│   ├── documents.csv           # 200 searchable documents
│   ├── ranking_signals.csv     # BM25, freshness, CTR signals
│   ├── query_logs.csv          # 300 query-click pairs
│   ├── query_autocomplete.txt  # 15 historical queries
│   ├── query_synonyms.csv      # 5 synonym mappings
│   ├── document_embeddings.csv # 50 document embeddings (3D)
│   └── rerank_features.csv     # 5 documents with rerank features
└── output/                         # Generated output files
    ├── task1_inverted_index.json
    ├── task2_tf_scores.json
    ├── task3_bm25_ranking.csv
    ├── task4_hybrid_ranking.csv
    ├── task5_autocomplete.json
    ├── task6_analytics.json
    ├── task7_personalized_ranking.csv
    ├── task8_vector_similarity.csv
    ├── task9_query_rewrites.json
    └── task10_reranked.csv
```

## How to Run

```bash
cd "Test 6/engineering_test_6"
```

**Run all tasks:**

```bash
python search_and_ranking_system.py
```

**Run a single task:**

```bash
python search_and_ranking_system.py 3
```

**Run multiple specific tasks:**

```bash
python search_and_ranking_system.py 1 3 5
```

All output files are written to the `output/` directory.

## Tasks Implemented

### 1. Build an Inverted Index

Tokenizes document content (lowercase, whitespace split) and builds a mapping of tokens → document IDs.

- **Run:** `python search_and_ranking_system.py 1`
- **Input dataset:** `documents.csv` (all 200 documents)
- **Output:** `task1_inverted_index.json`

### 2. Compute Term Frequency (TF)

Computes TF for each query term per document: `TF(t, d) = count(t in d) / total_tokens(d)`.

- **Run:** `python search_and_ranking_system.py 2`
- **Input dataset:** `documents.csv` (all 200 documents)
- **Input query:** `"diabetes diet"`
- **Output:** `task2_tf_scores.json` (only documents with non-zero TF)

### 3. Implement BM25 Ranking

Implements the standard BM25 scoring formula with parameters `k1=1.5`, `b=0.75`.

- **Run:** `python search_and_ranking_system.py 3`
- **IDF formula:** `ln((N - df + 0.5) / (df + 0.5) + 1)`
- **Input dataset:** `documents.csv` (all 200 documents)
- **Input query:** `"diabetes diet"`
- **Output:** `task3_bm25_ranking.csv` (all 200 documents, sorted by score descending)

### 4. Hybrid Ranking

Combines precomputed ranking signals using a weighted sum:
`score = 0.5 × BM25 + 0.3 × freshness + 0.2 × CTR`

- **Run:** `python search_and_ranking_system.py 4`
- **Input dataset:** `ranking_signals.csv` (precomputed BM25, freshness, CTR for all 200 documents)
- **Input weights:** `w_bm25=0.5, w_freshness=0.3, w_ctr=0.2`
- **Output:** `task4_hybrid_ranking.csv` (all 200 documents, sorted by score descending)

### 5. Query Autocomplete

Prefix-based matching against historical queries (case-insensitive), sorted alphabetically.

- **Run:** `python search_and_ranking_system.py 5`
- **Input dataset:** `query_autocomplete.txt` (15 historical queries)
- **Input prefixes:** `"heart"`, `"diabetes"`, `"lung"`
- **Output:** `task5_autocomplete.json`

### 6. Search Log Analytics

Computes from query logs:

- **Query frequency** — count of each query
- **Document click frequency** — count of clicks per document
- **Query CTR** — unique docs clicked / total impressions per query
- **Run:** `python search_and_ranking_system.py 6`
- **Input dataset:** `query_logs.csv` (300 query-click pairs)
- **Output:** `task6_analytics.json`

### 7. Personalization Boost

Multiplies hybrid ranking score by a boost factor when a document's tag matches the user's preferred category.

- **Run:** `python search_and_ranking_system.py 7`
- **Input datasets:** `documents.csv` (for tags) + `ranking_signals.csv` (for hybrid scores)
- **Input user tag:** `"cardiology"`
- **Input boost factor:** `1.2`
- **Output:** `task7_personalized_ranking.csv` (all 200 documents, sorted by boosted score descending)

### 8. Vector Similarity Search

Computes cosine similarity between a query embedding and all document embeddings.

- **Run:** `python search_and_ranking_system.py 8`
- **Input dataset:** `document_embeddings.csv` (50 documents with 3D embeddings)
- **Input query embedding:** `[0.5, 0.5, 0.8]`
- **Output:** `task8_vector_similarity.csv` (50 documents, sorted by similarity descending)

### 9. Query Rewriting

Maps queries to canonical forms using the synonym dictionary. Unmatched queries pass through unchanged.

- **Run:** `python search_and_ranking_system.py 9`
- **Input dataset:** `query_synonyms.csv` (5 synonym mappings)
- **Input test queries:** `"heart doctor"`, `"kid fever"`, `"lung doctor"`, `"blood sugar"`, `"diabetes food"`, `"diabetes diet"`
- **Output:** `task9_query_rewrites.json`

### 10. Reranking Pipeline

Applies a weighted reranking formula:
`score = 0.4 × relevance + 0.3 × CTR + 0.2 × authority + 0.1 × freshness`

- **Run:** `python search_and_ranking_system.py 10`
- **Input dataset:** `rerank_features.csv` (5 candidate documents with relevance, CTR, authority, freshness)
- **Input weights:** `w_rel=0.4, w_ctr=0.3, w_auth=0.2, w_fresh=0.1`
- **Output:** `task10_reranked.csv` (5 documents, sorted by rerank score descending)

## Design Decisions

- **No external dependencies:** Uses only Python standard libraries (`csv`, `json`, `math`, `collections`) for maximum portability.
- **Deterministic outputs:** All results are sorted consistently (by score descending, alphabetically where applicable).
- **No hardcoded values:** All outputs are computed from the provided datasets.
- **Modular architecture:** Each task is implemented as an independent, testable function.
