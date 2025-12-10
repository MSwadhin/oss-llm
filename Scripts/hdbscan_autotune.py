#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase1_cluster_commits_with_ids.py
----------------------------------
Phase 1 = Commit Message Clustering (Dict-style JSON input)

  # Basic run (keeps HDBSCAN -1 labels as true noise; no reassignment)
  python hdbscan_autotune.py \
    --input second_commits.json \
    --out-dir ./second_commits_clusters_autotuned

  # Recommended, more permissive clustering + auto-tune epsilon (tries a small grid if noise > 50%)
  python hdbscan_autotune.py \
    --input second_commits.json \
    --out-dir ./second_commits_clusters_autotuned \
    --svd-dims 128 \
    --min-cluster-size 15 \
    --epsilon 0.25 \
    --leaf \
    --auto-tune-epsilon \
    --epsilon-grid "0.20,0.25,0.30,0.35" \
    --per-cluster-files

INPUT FORMAT:
  A JSON file in any of these shapes:
  1) Dict: { "file_id_1": {"message": "...", "sha": "..."}, "file_id_2": {"message": "..."} ... }
  2) List of objects: [ {"message": "...", ...}, {"message": "...", ...}, ... ]
  3) JSONL: one JSON object per line

OUTPUTS (created in --out-dir):
  - assignments.csv       → file_id, message, cluster_id
  - clusters.csv          → cluster_id, size
  - params.json           → reproducibility info (versions, args)
  - clustering_report.json→ summary: counts, noise fraction, final epsilon
  - cluster_*.txt         → (optional, if --per-cluster-files) message previews per cluster

NOTES:
  - Cleaning is STRICT: lowercase + alphanumeric + single spaces.
  - Embedding is HYBRID: SBERT (all-MiniLM-L6-v2 by default) + TF-IDF reduced with SVD.
  - Distance: Euclidean on L2-normalized vectors (≈ cosine).
  - HDBSCAN keeps -1 as noise. We DO NOT reassign noise.
  - Auto-tuner: If initial noise > 50%, it retries with a small epsilon grid and keeps the best
    (lowest noise; ties broken by more clusters).

"""

import argparse, json, os, re, sys, platform, time, random
from typing import List, Dict, Tuple
import numpy as np, pandas as pd

# Reproducibility
random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import hdbscan


# ---------------- Utility ----------------
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


# ---------------- Load messages + IDs ----------------
def read_json_with_ids(path: str, message_field: str) -> Tuple[List[str], List[str]]:
    """
    Reads:
      - JSON dict {file_id: {message: ...}}
      - JSON array/list of objects
      - JSONL lines
    Returns:
      (list_of_ids, list_of_messages)
    """
    items: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        head_stripped = head.lstrip()
        if head_stripped.startswith("{"):
            data = json.load(f)
            if isinstance(data, dict):
                items = data
            else:
                sys.exit("Expected top-level JSON dict.")
        elif head_stripped.startswith("["):
            arr = json.load(f)
            items = {str(i): obj for i, obj in enumerate(arr) if isinstance(obj, dict)}
        else:
            # JSONL fallback
            items = {}
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    items[str(i)] = json.loads(line)

    ids, msgs = [], []
    for k, v in items.items():
        if isinstance(v, dict) and message_field in v and v[message_field]:
            ids.append(k)
            msgs.append(str(v[message_field]))
    return ids, msgs


# ---------------- Dedup (by message) ----------------
def dedupe_messages(ids: List[str], messages: List[str]) -> Tuple[List[str], List[int]]:
    seen: Dict[str, int] = {}
    uniq, back_idx = [], []
    for m in messages:
        if m in seen:
            back_idx.append(seen[m])
        else:
            seen[m] = len(uniq)
            uniq.append(m)
            back_idx.append(len(uniq) - 1)
    return uniq, back_idx


# ---------------- Embeddings ----------------
def embed_sbert(texts: List[str], model_name: str, batch: int, device: str) -> np.ndarray:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    cleaned = [clean_text(t) for t in texts]
    embs = model.encode(
        cleaned, batch_size=batch, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    # float64 for HDBSCAN stability
    return np.ascontiguousarray(embs, dtype=np.float64)


def embed_tfidf_svd(texts: List[str], n_components: int) -> np.ndarray:
    cleaned = [clean_text(s) for s in texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
    X = vec.fit_transform(cleaned)
    k = max(1, min(n_components, X.shape[1] - 1)) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=k, random_state=42)
    Xr = svd.fit_transform(X)
    Xr = normalize(Xr)
    return np.ascontiguousarray(Xr, dtype=np.float64)


def make_hybrid(texts: List[str], model: str, batch: int, device: str,
                svd_dims: int, save_embeddings_path: str = None) -> np.ndarray:
    sbert = embed_sbert(texts, model, batch, device)
    bowsvd = embed_tfidf_svd(texts, n_components=svd_dims)
    hybrid = normalize(np.hstack([sbert, bowsvd]))
    hybrid = np.ascontiguousarray(hybrid, dtype=np.float64)
    if save_embeddings_path:
        np.save(save_embeddings_path, hybrid)
    return hybrid


# ---------------- HDBSCAN ----------------
def run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int,
                epsilon: float, leaf: bool) -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(
        metric="euclidean",  # cosine ≈ euclidean on L2-normalized vectors
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf" if leaf else "eom",
        prediction_data=True,
        approx_min_span_tree=False,  # slightly more accurate tree
    ).fit(X)


def noise_ratio(labels: np.ndarray) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return float(np.sum(labels == -1)) / float(n)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--message-field", default="message")

    # Embeddings
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--svd-dims", type=int, default=128)

    # HDBSCAN (permissive defaults to reduce noise)
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--leaf", action="store_true", default=True)
    ap.add_argument("--no-leaf", dest="leaf", action="store_false")

    # Auto-tuner (kept): if initial noise > 50%, try a small epsilon grid
    ap.add_argument("--auto-tune-epsilon", action="store_true", default=True)
    ap.add_argument("--epsilon-grid", default="0.20,0.25,0.30,0.35")

    # Misc
    ap.add_argument("--save-embeddings", default="")
    ap.add_argument("--per-cluster-files", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Params log
    params = {
        "timestamp_utc": now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "hdbscan": getattr(hdbscan, "__version__", "unknown"),
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Load data
    ids, messages = read_json_with_ids(args.input, args.message_field)
    if not messages:
        sys.exit("No messages found.")
    uniq, back_idx = dedupe_messages(ids, messages)

    # Embedding
    X = make_hybrid(uniq, args.model, args.batch_size, args.device,
                    args.svd_dims, save_embeddings_path=(args.save_embeddings or None))

    # Cluster (initial)
    hdb = run_hdbscan(X, args.min_cluster_size, args.min_samples, args.epsilon, args.leaf)
    uniq_labels = hdb.labels_.astype(int)
    initial_noise = noise_ratio(uniq_labels)

    # Auto-tune epsilon (if requested and too much noise)
    if args.auto_tune_epsilon and initial_noise > 0.50:
        eps_list = [float(e.strip()) for e in args.epsilon_grid.split(",") if e.strip()]
        best_noise = initial_noise
        best_k = len(set(uniq_labels[uniq_labels >= 0]))
        best_eps = args.epsilon
        best_labels = uniq_labels
        best_hdb = hdb

        for eps in eps_list:
            h_try = run_hdbscan(X, args.min_cluster_size, args.min_samples, eps, args.leaf)
            labs = h_try.labels_.astype(int)
            nr = noise_ratio(labs)
            k = len(set(labs[labs >= 0]))
            # prefer lower noise; tie-break by more clusters
            if (nr < best_noise) or (nr == best_noise and k > best_k):
                best_noise, best_k, best_eps = nr, k, eps
                best_labels, best_hdb = labs, h_try

        if best_eps != args.epsilon:
            hdb = best_hdb
            uniq_labels = best_labels

    # Map to full list (undo dedupe)
    labels = np.array([uniq_labels[u] for u in back_idx], dtype=int)

    # Save assignments
    pd.DataFrame({
        "file_id": ids,
        "message": messages,
        "cluster_id": labels
    }).to_csv(os.path.join(args.out_dir, "assignments.csv"), index=False)

    # Cluster summary
    vc = pd.Series(labels).value_counts().sort_index()
    pd.DataFrame({
        "cluster_id": vc.index,
        "size": vc.values
    }).to_csv(os.path.join(args.out_dir, "clusters.csv"), index=False)

    # Report
    n_clusters = len(set(labels))
    noise_frac = float(np.sum(np.array(labels) == -1)) / len(labels)
    report = {
        "timestamp_utc": now_iso(),
        "num_commits": len(messages),
        "num_clusters": n_clusters,
        "epsilon_used": getattr(hdb, "cluster_selection_epsilon", "n/a"),
        "leaf_mode": args.leaf,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "noise_fraction": round(noise_frac, 4)
    }
    with open(os.path.join(args.out_dir, "clustering_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Optional per-cluster previews
    if args.per_cluster_files:
        by_c: Dict[int, List[str]] = {}
        for fid, msg, cid in zip(ids, messages, labels):
            by_c.setdefault(int(cid), []).append(f"{fid} :: {msg}")
        for cid, lines in by_c.items():
            with open(os.path.join(args.out_dir, f"cluster_{cid}.txt"), "w", encoding="utf-8") as f:
                for line in lines[:5000]:
                    f.write(line.replace("\n", " ").strip() + "\n")

    print(f"\n✅ Clustering complete — {n_clusters} clusters created.")
    print(f"Noise fraction: {report['noise_fraction']}")
    print(f"Assignments saved to: {os.path.join(args.out_dir, 'assignments.csv')}")
    print(f"Clusters saved to: {os.path.join(args.out_dir, 'clusters.csv')}")
    print(f"Report saved to: {os.path.join(args.out_dir, 'clustering_report.json')}")

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
