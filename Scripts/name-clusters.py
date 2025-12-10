"""
phase2_name_clusters.py
-----------------------

Phase 2 = Cluster naming via keyword analysis.

Inputs:
  - assignments.csv (from Phase 1), with columns:
      file_id, message, cluster_id[, confidence]

Outputs (in --out-dir):
  - cluster_keywords.csv     : per-cluster top terms (1–3 grams) with scores
  - cluster_summaries.txt    : readable summaries (size, name, keywords, 5 examples)
  - cluster_names.txt        : "cluster_id,name" pairs (auto names or "TBD")

Key techniques:
  - Strict text cleaning (lowercase, alnum, spaces).
  - c-TF-IDF (class-based TF-IDF): builds one "document" per cluster by concatenating
    its messages, then computes TF-IDF across these cluster-docs to get discriminative terms.
  - Optional auto-naming via a lightweight heuristic dictionary (toggle with --auto-name).
    If disabled, names are "TBD" and you can edit cluster_names.txt manually.

Usage example:
  python phase2_name_clusters.py \
    --assignments ./phase1_clusters/assignments.csv \
    --out-dir       ./phase2_naming \
    --auto-name

"""

import argparse, os, re, sys, json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# ------------------- Cleaning -------------------
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

# Minimal stopwords for naming (NOT used in Phase 1 clustering)
DEFAULT_STOPS = {
    "the","a","an","and","or","for","to","of","in","on","by","with","from",
    "is","are","was","were","be","being","been","this","that","it","as",
    "into","at","over","under","up","down","out","about","across","after",
    "before","again","further","more","most","some","such","no","not"
}

# ------------------- c-TF-IDF -------------------
def compute_c_tf_idf(docs_per_cluster: List[str], ngram_range=(1,3), min_df=1,
                     stopwords: set = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build a c-TF-IDF matrix:
      - One document per cluster (concatenate all cluster messages)
      - CountVectorizer (n-grams)
      - Term Frequency = counts / doc_length
      - Inverse Document Frequency across clusters = log( n_clusters / df )
      - cTFIDF = TF * IDF
    Returns:
      ctfidf (C x V), feature_names
    """
    if stopwords is None:
        stopwords = set()
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=list(stopwords) if stopwords else None
    )
    X = vectorizer.fit_transform(docs_per_cluster)  # C x V
    # TF: normalize rows to term frequency
    tf = normalize(X, norm="l1", axis=1)  # each row sums to 1
    # IDF across clusters
    df = np.asarray((X > 0).sum(axis=0)).ravel()  # V
    n_clusters = X.shape[0]
    idf = np.log((n_clusters + 1) / (df + 1)) + 1.0  # smoothed
    ctfidf = tf.multiply(idf)
    return ctfidf, vectorizer.get_feature_names_out().tolist()

# ------------------- Auto-naming (optional) -------------------
def suggest_name_from_terms(top_terms: List[str]) -> str:
    """
    Lightweight heuristic to propose a label from the top terms.
    You can disable this with --no-auto-name to keep names as "TBD".
    """
    s = " ".join(top_terms[:20])

    # buckets (simple, transparent)
    if any(t in s.split() for t in ["fix","fixed","bug","hotfix","patch","regression","issue"]):
        return "Bug Fix"
    if any(t in s.split() for t in ["refactor","rewrite","cleanup","restructure","rename","extract","simplify"]):
        return "Refactor"
    if any(t in s.split() for t in ["test","tests","unittest","ci","workflow","coverage","assert"]):
        return "Tests / CI"
    if any(t in s.split() for t in ["doc","docs","documentation","readme","changelog","comment","comments"]):
        return "Docs / Comments"
    if any(t in s.split() for t in ["bump","upgrade","dependency","dependencies","deps","pin","vendor"]):
        return "Dependencies / Build"
    if any(t in s.split() for t in ["security","cve","xss","csrf","sqli","sanitize","token","auth"]):
        return "Security"
    if any(t in s.split() for t in ["config","configuration","settings","yaml","json","properties","schema"]):
        return "Config / Settings"
    if any(t in s.split() for t in ["feat","feature","features","add","added","implement","support","enable","introduce"]):
        return "Feature / Implementation"
    if any(t in s.split() for t in ["revert","rollback","backout"]):
        return "Revert / Backout"
    if any(t in s.split() for t in ["style","format","lint","prettier","eslint","pep8","black","isort","whitespace"]):
        return "Style / Formatting"
    # fallback
    return "TBD"

# ------------------- Main -------------------
def main():
    print("Script Started")
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", required=True, help="Path to Phase-1 assignments.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for naming artifacts")
    ap.add_argument("--topk", type=int, default=12, help="Top terms per cluster")
    ap.add_argument("--min-df", type=int, default=1, help="Min doc freq across clusters for n-grams")
    ap.add_argument("--no-stop", action="store_true", help="Disable stopword filtering for naming")
    ap.add_argument("--auto-name", action="store_true", help="Enable auto naming via heuristics")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.assignments)
    needed = {"file_id","message","cluster_id"}
    if not needed.issubset(df.columns):
        sys.exit(f"assignments.csv must contain columns: {sorted(list(needed))}")

    # Clean messages
    df["clean"] = df["message"].astype(str).map(clean_text)

    # Build cluster docs
    groups = df.groupby("cluster_id", sort=True)
    cluster_ids = list(groups.groups.keys())
    docs_per_cluster = [" ".join(g["clean"].tolist()) for _, g in groups]
    sizes = [int(g.shape[0]) for _, g in groups]

    # Stopwords (optional)
    stops = None if args.no_stop else DEFAULT_STOPS

    # Compute c-TF-IDF
    ctfidf, vocab = compute_c_tf_idf(docs_per_cluster, ngram_range=(1,3),
                                     min_df=args.min_df, stopwords=stops)
    ctfidf = ctfidf.toarray()

    # Extract top terms per cluster
    top_terms_per_cluster: Dict[int, List[Tuple[str, float]]] = {}
    for i, cid in enumerate(cluster_ids):
        row = ctfidf[i]
        idxs = np.argsort(-row)[:max(args.topk, 1)]
        top_terms_per_cluster[cid] = [(vocab[j], float(row[j])) for j in idxs if row[j] > 0]

    # Save keywords table
    rows = []
    for cid, sz in zip(cluster_ids, sizes):
        terms = [t for t, _ in top_terms_per_cluster[cid]]
        scores = [s for _, s in top_terms_per_cluster[cid]]
        rows.append({
            "cluster_id": cid,
            "size": sz,
            "top_terms": ", ".join(terms),
            "scores": ", ".join([f"{x:.4f}" for x in scores])
        })
    kw_df = pd.DataFrame(rows).sort_values("cluster_id")
    kw_df.to_csv(os.path.join(args.out_dir, "cluster_keywords.csv"), index=False)

    # Auto names (optional)
    names = {}
    if args.auto_name:
        for cid in cluster_ids:
            terms = [t for t, _ in top_terms_per_cluster[cid]]
            names[cid] = suggest_name_from_terms(terms)
    else:
        for cid in cluster_ids:
            names[cid] = "TBD"

    # Save names
    with open(os.path.join(args.out_dir, "cluster_names.txt"), "w", encoding="utf-8") as f:
        for cid in sorted(cluster_ids):
            f.write(f"{cid},{names[cid]}\n")

    # Summaries (with 5 examples per cluster)
    with open(os.path.join(args.out_dir, "cluster_summaries.txt"), "w", encoding="utf-8") as f:
        for cid, g in groups:
            size = g.shape[0]
            terms = [t for t, _ in top_terms_per_cluster[cid]]
            f.write(f"=== Cluster {cid} | size={size} ===\n")
            f.write(f"Name: {names[cid]}\n")
            f.write(f"Top terms: {', '.join(terms[:args.topk])}\n")
            f.write("Examples:\n")
            ex = g.sample(n=min(5, size), random_state=42)[["file_id","message"]]
            for _, row in ex.iterrows():
                one = " ".join(str(row["message"]).split())
                f.write(f"  - {row['file_id']} :: {one}\n")
            f.write("\n")

    # Small metadata
    meta = {
        "num_clusters": len(cluster_ids),
        "total_commits": int(df.shape[0]),
        "topk": args.topk,
        "min_df": args.min_df,
        "stopwords": "disabled" if args.no_stop else "default_minimal",
        "auto_name": bool(args.auto_name)
    }
    with open(os.path.join(args.out_dir, "naming_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Phase 2 naming complete.")
    print("  - cluster_keywords.csv")
    print("  - cluster_names.txt")
    print("  - cluster_summaries.txt")



if __name__ == "__main__":
    main()