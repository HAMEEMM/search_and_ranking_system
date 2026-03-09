"""
Search Engineering Coding Assessment - Test 6
Digital Product Engineering

A complete implementation of a simplified search platform including:
1.  Build an Inverted Index
2.  Compute Term Frequency (TF)
3.  Implement BM25 Ranking
4.  Hybrid Ranking
5.  Query Autocomplete
6.  Search Log Analytics
7.  Personalization Boost
8.  Vector Similarity Search
9.  Query Rewriting
10. Reranking Pipeline
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# ---------------------------------------------------------------------------
# Utility: Load CSV / TXT helpers
# ---------------------------------------------------------------------------
def load_documents(path):
    """Return list of dicts from documents.csv."""
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["doc_id"] = int(row["doc_id"])
            row["freshness"] = float(row["freshness"])
            docs.append(row)
    return docs


def load_ranking_signals(path):
    """Return dict keyed by doc_id from ranking_signals.csv."""
    signals = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = int(row["doc_id"])
            signals[did] = {
                "bm25": float(row["bm25"]),
                "freshness": float(row["freshness"]),
                "ctr": float(row["ctr"]),
            }
    return signals


def load_query_logs(path):
    """Return list of (query, clicked_doc) tuples."""
    logs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logs.append((row["query"], int(row["clicked_doc"])))
    return logs


def load_autocomplete_queries(path):
    """Return list of historical query strings."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_synonyms(path):
    """Return dict mapping query_term -> rewrite."""
    syns = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            syns[row["query_term"]] = row["rewrite"]
    return syns


def load_embeddings(path):
    """Return dict keyed by doc_id -> list[float]."""
    embs = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = int(row["doc_id"])
            embs[did] = json.loads(row["embedding"])
    return embs


def load_rerank_features(path):
    """Return list of dicts from rerank_features.csv."""
    feats = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats.append({
                "doc_id": int(row["doc_id"]),
                "relevance": float(row["relevance"]),
                "ctr": float(row["ctr"]),
                "authority": float(row["authority"]),
                "freshness": float(row["freshness"]),
            })
    return feats


# ---------------------------------------------------------------------------
# Helper: Tokenize
# ---------------------------------------------------------------------------
def tokenize(text):
    """Lowercase and split on whitespace."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# Task 1: Build an Inverted Index
# ---------------------------------------------------------------------------
def build_inverted_index(documents):
    """
    Tokenize document content and build a mapping of tokens -> set of doc_ids.
    """
    index = defaultdict(set)
    for doc in documents:
        tokens = tokenize(doc["content"])
        for token in tokens:
            index[token].add(doc["doc_id"])
    # Sort for deterministic output
    return {k: sorted(v) for k, v in sorted(index.items())}


# ---------------------------------------------------------------------------
# Task 2: Compute Term Frequency (TF)
# ---------------------------------------------------------------------------
def compute_tf(documents, query):
    """
    For each document, compute TF for every query term.
    TF(t, d) = count(t in d) / total_tokens(d)
    Returns dict: doc_id -> {term: tf_value}
    """
    query_terms = tokenize(query)
    result = {}
    for doc in documents:
        tokens = tokenize(doc["content"])
        total = len(tokens)
        tf = {}
        for term in query_terms:
            count = tokens.count(term)
            tf[term] = count / total if total > 0 else 0.0
        result[doc["doc_id"]] = tf
    return result


# ---------------------------------------------------------------------------
# Task 3: Implement BM25 Ranking
# ---------------------------------------------------------------------------
def compute_bm25(documents, query, k1=1.5, b=0.75):
    """
    Compute BM25 scores for all documents given a query.
    BM25(d, q) = sum over q_terms of IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*(dl/avgdl)))
    IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)   (non-negative variant)
    Returns dict: doc_id -> bm25_score, sorted descending.
    """
    query_terms = tokenize(query)
    N = len(documents)

    # Pre-compute doc lengths and avg doc length
    doc_tokens = {}
    for doc in documents:
        doc_tokens[doc["doc_id"]] = tokenize(doc["content"])

    avgdl = sum(len(t) for t in doc_tokens.values()) / N if N > 0 else 0

    # Document frequency for each query term
    df = {}
    for term in query_terms:
        df[term] = sum(1 for tokens in doc_tokens.values() if term in tokens)

    scores = {}
    for doc in documents:
        did = doc["doc_id"]
        tokens = doc_tokens[did]
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = tokens.count(term)
            if tf == 0:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))
            score += idf * (numerator / denominator)
        scores[did] = round(score, 4)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Task 4: Hybrid Ranking
# ---------------------------------------------------------------------------
def hybrid_ranking(ranking_signals, w_bm25=0.5, w_freshness=0.3, w_ctr=0.2):
    """
    Combine BM25, freshness, and CTR signals using weighted sum.
    hybrid_score = w_bm25 * bm25 + w_freshness * freshness + w_ctr * ctr
    Returns list of (doc_id, score) sorted descending.
    """
    scores = {}
    for did, sig in ranking_signals.items():
        score = (w_bm25 * sig["bm25"]
                 + w_freshness * sig["freshness"]
                 + w_ctr * sig["ctr"])
        scores[did] = round(score, 4)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Task 5: Query Autocomplete
# ---------------------------------------------------------------------------
def autocomplete(prefix, historical_queries):
    """
    Return all historical queries that start with the given prefix (case-insensitive).
    Sorted alphabetically for determinism.
    """
    prefix_lower = prefix.lower()
    matches = [q for q in historical_queries if q.lower().startswith(prefix_lower)]
    return sorted(matches)


# ---------------------------------------------------------------------------
# Task 6: Search Log Analytics
# ---------------------------------------------------------------------------
def search_log_analytics(query_logs):
    """
    Compute:
    - query_frequency: how many times each query appears
    - doc_click_frequency: how many times each doc was clicked
    - query_ctr: for each query, number of unique docs clicked / total impressions
    """
    query_freq = defaultdict(int)
    doc_click_freq = defaultdict(int)
    query_clicks = defaultdict(list)

    for query, doc_id in query_logs:
        query_freq[query] += 1
        doc_click_freq[doc_id] += 1
        query_clicks[query].append(doc_id)

    # CTR per query = unique docs clicked / total impressions (searches) for that query
    query_ctr = {}
    for query, clicks in query_clicks.items():
        unique_clicks = len(set(clicks))
        total = len(clicks)
        query_ctr[query] = round(unique_clicks / total, 4)

    return (
        dict(sorted(query_freq.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(doc_click_freq.items(), key=lambda x: x[1], reverse=True)),
        dict(sorted(query_ctr.items(), key=lambda x: x[1], reverse=True)),
    )


# ---------------------------------------------------------------------------
# Task 7: Personalization Boost
# ---------------------------------------------------------------------------
def personalization_boost(documents, ranking_signals, user_tag, boost=1.2):
    """
    Increase hybrid ranking score by a boost factor when a document's tag
    matches the user's preferred condition/tag.
    Returns list of (doc_id, boosted_score) sorted descending.
    """
    hybrid = hybrid_ranking(ranking_signals)
    boosted = []
    # Build tag lookup
    tag_map = {doc["doc_id"]: doc["tag"] for doc in documents}
    for did, score in hybrid:
        if tag_map.get(did) == user_tag:
            boosted.append((did, round(score * boost, 4)))
        else:
            boosted.append((did, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Task 8: Vector Similarity Search
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def vector_similarity_search(query_embedding, doc_embeddings):
    """
    Compute cosine similarity between query embedding and all document embeddings.
    Returns list of (doc_id, similarity) sorted descending.
    """
    results = []
    for did, emb in doc_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        results.append((did, round(sim, 4)))
    return sorted(results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Task 9: Query Rewriting
# ---------------------------------------------------------------------------
def rewrite_query(query, synonyms):
    """
    If the query matches a synonym key exactly, rewrite it.
    Otherwise return the original query.
    """
    return synonyms.get(query.lower(), query)


# ---------------------------------------------------------------------------
# Task 10: Reranking Pipeline
# ---------------------------------------------------------------------------
def rerank(rerank_features, w_rel=0.4, w_ctr=0.3, w_auth=0.2, w_fresh=0.1):
    """
    Apply reranking formula:
    score = w_rel*relevance + w_ctr*ctr + w_auth*authority + w_fresh*freshness
    Returns list of (doc_id, score) sorted descending.
    """
    results = []
    for feat in rerank_features:
        score = (w_rel * feat["relevance"]
                 + w_ctr * feat["ctr"]
                 + w_auth * feat["authority"]
                 + w_fresh * feat["freshness"])
        results.append((feat["doc_id"], round(score, 4)))
    return sorted(results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_json(data, filename):
    """Write data to a JSON file in the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def write_csv_rows(rows, headers, filename):
    """Write rows (list of dicts or tuples) to CSV."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            if isinstance(row, dict):
                writer.writerow([row[h] for h in headers])
            else:
                writer.writerow(row)
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------
def run_task1():
    """Task 1: Build an Inverted Index."""
    documents = load_documents(os.path.join(DATA_DIR, "documents.csv"))
    print("\n--- Task 1: Build an Inverted Index ---")
    inv_index = build_inverted_index(documents)
    write_json(inv_index, "task1_inverted_index.json")
    for token in list(inv_index.keys())[:5]:
        print(f"  '{token}' -> {inv_index[token][:10]}{'...' if len(inv_index[token]) > 10 else ''}")


def run_task2():
    """Task 2: Compute Term Frequency (TF)."""
    documents = load_documents(os.path.join(DATA_DIR, "documents.csv"))
    print("\n--- Task 2: Compute Term Frequency (TF) ---")
    sample_query = "diabetes diet"
    tf_scores = compute_tf(documents, sample_query)
    tf_nonzero = {
        str(did): tfs for did, tfs in tf_scores.items()
        if any(v > 0 for v in tfs.values())
    }
    write_json(tf_nonzero, "task2_tf_scores.json")
    print(f"  Query: '{sample_query}'")
    print(f"  Documents with non-zero TF: {len(tf_nonzero)}")


def run_task3():
    """Task 3: Implement BM25 Ranking."""
    documents = load_documents(os.path.join(DATA_DIR, "documents.csv"))
    print("\n--- Task 3: Implement BM25 Ranking ---")
    sample_query = "diabetes diet"
    bm25_scores = compute_bm25(documents, sample_query)
    bm25_top10 = list(bm25_scores.items())[:10]
    write_csv_rows(
        [(did, score) for did, score in bm25_scores.items()],
        ["doc_id", "bm25_score"],
        "task3_bm25_ranking.csv"
    )
    print(f"  Top 10 for '{sample_query}':")
    for did, score in bm25_top10:
        print(f"    Doc {did}: {score}")


def run_task4():
    """Task 4: Hybrid Ranking."""
    ranking_signals = load_ranking_signals(os.path.join(DATA_DIR, "ranking_signals.csv"))
    print("\n--- Task 4: Hybrid Ranking ---")
    hybrid_scores = hybrid_ranking(ranking_signals)
    write_csv_rows(
        hybrid_scores,
        ["doc_id", "hybrid_score"],
        "task4_hybrid_ranking.csv"
    )
    print(f"  Top 10:")
    for did, score in hybrid_scores[:10]:
        print(f"    Doc {did}: {score}")


def run_task5():
    """Task 5: Query Autocomplete."""
    autocomplete_queries = load_autocomplete_queries(os.path.join(DATA_DIR, "query_autocomplete.txt"))
    print("\n--- Task 5: Query Autocomplete ---")
    test_prefixes = ["heart", "diabetes", "lung"]
    autocomplete_results = {}
    for prefix in test_prefixes:
        matches = autocomplete(prefix, autocomplete_queries)
        autocomplete_results[prefix] = matches
        print(f"  '{prefix}' -> {matches}")
    write_json(autocomplete_results, "task5_autocomplete.json")


def run_task6():
    """Task 6: Search Log Analytics."""
    query_logs = load_query_logs(os.path.join(DATA_DIR, "query_logs.csv"))
    print("\n--- Task 6: Search Log Analytics ---")
    query_freq, doc_click_freq, query_ctr = search_log_analytics(query_logs)
    analytics = {
        "query_frequency": query_freq,
        "doc_click_frequency": {str(k): v for k, v in doc_click_freq.items()},
        "query_ctr": query_ctr,
    }
    write_json(analytics, "task6_analytics.json")
    print(f"  Top 5 queries by frequency:")
    for q, cnt in list(query_freq.items())[:5]:
        print(f"    '{q}': {cnt}")
    print(f"  Top 5 clicked docs:")
    for did, cnt in list(doc_click_freq.items())[:5]:
        print(f"    Doc {did}: {cnt}")


def run_task7():
    """Task 7: Personalization Boost."""
    documents = load_documents(os.path.join(DATA_DIR, "documents.csv"))
    ranking_signals = load_ranking_signals(os.path.join(DATA_DIR, "ranking_signals.csv"))
    print("\n--- Task 7: Personalization Boost ---")
    user_tag = "cardiology"
    boosted = personalization_boost(documents, ranking_signals, user_tag)
    write_csv_rows(
        boosted,
        ["doc_id", "boosted_score"],
        "task7_personalized_ranking.csv"
    )
    print(f"  User tag: '{user_tag}'")
    print(f"  Top 10 after boosting:")
    for did, score in boosted[:10]:
        print(f"    Doc {did}: {score}")


def run_task8():
    """Task 8: Vector Similarity Search."""
    doc_embeddings = load_embeddings(os.path.join(DATA_DIR, "document_embeddings.csv"))
    print("\n--- Task 8: Vector Similarity Search ---")
    query_embedding = [0.5, 0.5, 0.8]
    sim_results = vector_similarity_search(query_embedding, doc_embeddings)
    write_csv_rows(
        sim_results,
        ["doc_id", "cosine_similarity"],
        "task8_vector_similarity.csv"
    )
    print(f"  Query embedding: {query_embedding}")
    print(f"  Top 10 similar:")
    for did, sim in sim_results[:10]:
        print(f"    Doc {did}: {sim}")


def run_task9():
    """Task 9: Query Rewriting."""
    synonyms = load_synonyms(os.path.join(DATA_DIR, "query_synonyms.csv"))
    print("\n--- Task 9: Query Rewriting ---")
    test_queries = ["heart doctor", "kid fever", "lung doctor", "blood sugar",
                    "diabetes food", "diabetes diet"]
    rewrite_results = {}
    for q in test_queries:
        rewritten = rewrite_query(q, synonyms)
        rewrite_results[q] = rewritten
        print(f"  '{q}' -> '{rewritten}'")
    write_json(rewrite_results, "task9_query_rewrites.json")


def run_task10():
    """Task 10: Reranking Pipeline."""
    rerank_features = load_rerank_features(os.path.join(DATA_DIR, "rerank_features.csv"))
    print("\n--- Task 10: Reranking Pipeline ---")
    reranked = rerank(rerank_features)
    write_csv_rows(
        reranked,
        ["doc_id", "rerank_score"],
        "task10_reranked.csv"
    )
    print(f"  Reranked results:")
    for did, score in reranked:
        print(f"    Doc {did}: {score}")


# Map of task number -> runner function
TASKS = {
    1: run_task1,
    2: run_task2,
    3: run_task3,
    4: run_task4,
    5: run_task5,
    6: run_task6,
    7: run_task7,
    8: run_task8,
    9: run_task9,
    10: run_task10,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Search and Ranking System — run all or individual tasks (1-10)."
    )
    parser.add_argument(
        "tasks",
        nargs="*",
        type=int,
        help="Task number(s) to run (1-10). Omit to run all tasks.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.tasks:
        for t in args.tasks:
            if t not in TASKS:
                print(f"Error: Unknown task {t}. Valid tasks are 1-10.")
                sys.exit(1)
            TASKS[t]()
    else:
        print("Running all tasks...")
        for t in sorted(TASKS):
            TASKS[t]()

    print(f"\n{'='*60}")
    print(f"Output files written to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
