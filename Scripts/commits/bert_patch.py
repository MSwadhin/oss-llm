#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer, models

import hdbscan
import umap

import spacy
from nltk.stem import PorterStemmer


PR_TAIL_RE = re.compile(r"\s*\(#\d+\)\s*$")
HASHNUM_RE = re.compile(r"#\d+\b")

# Patch parsing helpers (unified diff-ish)
RE_FILE = re.compile(r"^\+\+\+\s+b/(.+)$")
RE_FILE_ALT = re.compile(r"^\+\+\+\s+(.+)$")
RE_HUNK = re.compile(r"^@@ ")
RE_ADD = re.compile(r"^\+(?!\+\+\+)")
RE_DEL = re.compile(r"^-(?!---)")

# lightweight keyword boosters for intention
INTENT_KEYWORDS = [
    "log", "logger", "logging", "login", "auth", "authenticate",
    "error", "exception", "fail", "fix", "bug", "crash",
    "test", "ci", "build", "refactor", "perf", "optimize",
]


# ---------------------------
# Config + IO
# ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json_records(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


# ---------------------------
# Preprocessing
# ---------------------------
def build_spacy_if_needed(cfg: Dict[str, Any]):
    nlp_cfg = (cfg.get("preprocess", {}) or {}).get("nlp", {}) or {}
    enabled = bool(nlp_cfg.get("enabled", False))
    lemmatize = bool(nlp_cfg.get("lemmatize", False))
    stem = bool(nlp_cfg.get("stem", False))

    if not enabled or (not lemmatize and not stem):
        return None, None

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stemmer = PorterStemmer() if stem else None
    return nlp, stemmer


def preprocess_text(text: str, cfg: Dict[str, Any], nlp, stemmer) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    pre = cfg.get("preprocess", {}) or {}
    nlp_cfg = (pre.get("nlp", {}) or {})

    if pre.get("lowercase", True):
        s = s.lower()

    if pre.get("strip_pr_issue_refs", True):
        s = PR_TAIL_RE.sub("", s)
        s = HASHNUM_RE.sub("", s)

    s = re.sub(r"\s+", " ", s).strip()
    min_chars = int(pre.get("min_chars", 2))
    if len(s) < min_chars:
        return ""

    enabled = bool(nlp_cfg.get("enabled", False))
    lemmatize = bool(nlp_cfg.get("lemmatize", False))
    stem = bool(nlp_cfg.get("stem", False))

    if (not enabled) or (not lemmatize and not stem) or nlp is None:
        return s

    doc = nlp(s)
    tokens: List[str] = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue

        out = tok.text
        if lemmatize:
            out = tok.lemma_
        if stem:
            out = stemmer.stem(out)  # type: ignore[union-attr]

        out = out.strip()
        if len(out) >= 2:
            tokens.append(out)

    return " ".join(tokens)


# ---------------------------
# Patch features
# ---------------------------
def extract_patch_features(patch: str, cfg: Dict[str, Any]) -> str:
    """
    Extract compact signals from a unified diff patch.
    Controls in config.json:
      patch.enabled: bool
      patch.max_added_lines: int
      patch.max_removed_lines: int
      patch.max_files: int
      patch.include_files: bool
      patch.include_added_removed: bool
      patch.include_keywords: bool
      patch.max_chars: int
    """
    pcfg = cfg.get("patch", {}) or {}
    if not bool(pcfg.get("enabled", False)):
        return ""

    patch = (patch or "").strip()
    if not patch:
        return ""

    include_files = bool(pcfg.get("include_files", True))
    include_ar = bool(pcfg.get("include_added_removed", True))
    include_kw = bool(pcfg.get("include_keywords", True))

    max_added = int(pcfg.get("max_added_lines", 10))
    max_removed = int(pcfg.get("max_removed_lines", 10))
    max_files = int(pcfg.get("max_files", 10))
    max_chars = int(pcfg.get("max_chars", 1500))

    files: List[str] = []
    added: List[str] = []
    removed: List[str] = []

    for line in patch.splitlines():
        line = line.rstrip("\n")

        if include_files:
            m = RE_FILE.match(line) or RE_FILE_ALT.match(line)
            if m:
                fp = m.group(1).strip()
                if fp and fp not in files:
                    files.append(fp)
                if len(files) >= max_files:
                    include_files = False  # stop collecting more
                continue

        if include_ar:
            if RE_ADD.match(line):
                if len(added) < max_added:
                    added.append(line[1:].strip())
                continue
            if RE_DEL.match(line):
                if len(removed) < max_removed:
                    removed.append(line[1:].strip())
                continue

    # keyword signals from patch
    kw_hits: List[str] = []
    if include_kw:
        blob = " ".join(files + added + removed).lower()
        for kw in INTENT_KEYWORDS:
            if kw in blob:
                kw_hits.append(kw)

    parts: List[str] = []
    if files:
        parts.append("FILES: " + " ".join(files[:max_files]))
    if added:
        parts.append("ADDED: " + " ; ".join(added[:max_added]))
    if removed:
        parts.append("REMOVED: " + " ; ".join(removed[:max_removed]))
    if kw_hits:
        parts.append("KEYWORDS: " + " ".join(sorted(set(kw_hits))))

    out = " | ".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out


# ---------------------------
# Data loading
# ---------------------------
def load_documents(cfg: Dict[str, Any]) -> pd.DataFrame:
    input_json = cfg["input_json"]
    text_source = cfg.get("text_source", "message")

    with open(input_json, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    nlp, stemmer = build_spacy_if_needed(cfg)

    rows: List[Dict[str, Any]] = []
    for doc_key, obj in data.items():
        raw = (obj.get(text_source) or "").strip()
        processed = preprocess_text(raw, cfg, nlp, stemmer)
        if not processed:
            continue

        patch_raw = obj.get("patch", "") or ""
        patch_feat = extract_patch_features(patch_raw, cfg)

        # doc_text is what we embed/cluster on
        doc_text = processed
        if patch_feat:
            doc_text = f"{processed} [PATCH] {patch_feat}"

        rows.append(
            {
                "doc_key": doc_key,
                "sha": obj.get("sha", ""),
                "author": obj.get("author", ""),
                "date": obj.get("date", ""),
                "message_raw": raw,
                "message": processed,
                "patch_feat": patch_feat,
                "doc_text": doc_text,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------
# Encoders
# ---------------------------
def build_hf_sentence_transformer(model_name: str, pooling: str) -> SentenceTransformer:
    transformer = models.Transformer(model_name)
    pooling = (pooling or "mean").lower().strip()
    if pooling not in {"mean", "cls"}:
        raise ValueError(f"Unsupported pooling='{pooling}'. Use 'mean' or 'cls'.")

    pool = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=(pooling == "mean"),
        pooling_mode_cls_token=(pooling == "cls"),
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pool])


def build_embedder(cfg: Dict[str, Any]) -> SentenceTransformer:
    enc = cfg.get("encoder", {}) or {}
    enc_type = (enc.get("type", "unixcoder") or "unixcoder").lower().strip()
    pooling = enc.get("pooling", "mean")

    presets = {
        "unixcoder": "microsoft/unixcoder-base",
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
        "codet5": "Salesforce/codet5-base",
        "plbart": "uclanlp/plbart-base",
    }

    if enc_type in presets:
        model_name = enc.get("model_name", presets[enc_type]) or presets[enc_type]
        return build_hf_sentence_transformer(model_name, pooling)

    if enc_type == "hf_transformer":
        model_name = enc["model_name"]
        return build_hf_sentence_transformer(model_name, pooling)

    if enc_type == "sentence_transformer":
        model_name = enc["model_name"]
        return SentenceTransformer(model_name)

    raise ValueError(
        "encoder.type must be one of: unixcoder, codebert, graphcodebert, codet5, plbart, "
        "hf_transformer, sentence_transformer"
    )


# ---------------------------
# BERTopic components
# ---------------------------
def build_vectorizer(cfg: Dict[str, Any]) -> Optional[CountVectorizer]:
    vec_cfg = cfg.get("vectorizer", {}) or {}
    if not bool(vec_cfg.get("enabled", False)):
        return None

    stop_words = vec_cfg.get("stop_words", "english")
    if isinstance(stop_words, str) and stop_words.lower() == "none":
        stop_words = None

    ngram_range = vec_cfg.get("ngram_range", [1, 2])
    if not (isinstance(ngram_range, (list, tuple)) and len(ngram_range) == 2):
        raise ValueError("vectorizer.ngram_range must be [min, max]")

    return CountVectorizer(
        stop_words=stop_words,
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=int(vec_cfg.get("min_df", 2)),
    )


def build_hdbscan(cfg: Dict[str, Any]):
    hcfg = cfg.get("hdbscan", {}) or {}
    if not bool(hcfg.get("enabled", False)):
        return None
    return hdbscan.HDBSCAN(
        min_cluster_size=int(hcfg.get("min_cluster_size", 10)),
        min_samples=int(hcfg.get("min_samples", 2)),
        metric=hcfg.get("metric", "euclidean"),
        cluster_selection_method=hcfg.get("cluster_selection_method", "eom"),
        prediction_data=bool(hcfg.get("prediction_data", True)),
    )


def build_umap(cfg: Dict[str, Any]):
    ucfg = cfg.get("umap", {}) or {}
    if not bool(ucfg.get("enabled", False)):
        return None
    return umap.UMAP(
        n_neighbors=int(ucfg.get("n_neighbors", 15)),
        n_components=int(ucfg.get("n_components", 5)),
        min_dist=float(ucfg.get("min_dist", 0.0)),
        metric=ucfg.get("metric", "cosine"),
        random_state=int(ucfg.get("random_state", 42)),
    )


# ---------------------------
# Dedup + sizes
# ---------------------------
def build_unique_corpus(df: pd.DataFrame, dedup_key: str) -> tuple[pd.DataFrame, Counter]:
    counts = Counter(df[dedup_key].astype(str).tolist())
    df_unique = df.drop_duplicates(subset=[dedup_key]).copy().reset_index(drop=True)
    df_unique["count_total"] = df_unique[dedup_key].astype(str).map(counts).astype(int)
    return df_unique, counts


def compute_topic_sizes(df_unique: pd.DataFrame) -> pd.DataFrame:
    g = df_unique.groupby("topic", dropna=False)
    out = pd.DataFrame({
        "topic": g.size().index.astype(int),
        "size_unique": g.size().values.astype(int),
        "size_total": g["count_total"].sum().values.astype(int),
    }).sort_values("topic").reset_index(drop=True)
    return out


def write_cluster_texts(df_unique: pd.DataFrame, outdir: str, cfg: Dict[str, Any]) -> None:
    c_cfg = cfg.get("clusters_output", {}) or {}
    message_field = c_cfg.get("message_field", "message_raw")
    include_outliers = bool(c_cfg.get("include_outliers", True))

    clusters_dir = os.path.join(outdir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    topics = sorted(df_unique["topic"].unique().tolist(), key=lambda x: (int(x) == -1, int(x)))
    for t in topics:
        t = int(t)
        if t == -1 and not include_outliers:
            continue

        path = os.path.join(clusters_dir, f"topic_{t}.txt")
        msgs = df_unique.loc[df_unique["topic"] == t, message_field].astype(str).tolist()

        with open(path, "w", encoding="utf-8") as f:
            for m in msgs:
                m = m.replace("\n", " ").strip()
                if m:
                    f.write(m + "\n")


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BERTopic commit clustering via config.json (dedup)")
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = cfg.get("outdir", "out_bertopic")
    ensure_outdir(outdir)

    df = load_documents(cfg)
    if df.empty:
        raise SystemExit("No usable commit messages found after preprocessing.")

    # Dedup: default to message_raw
    dedup_key = (cfg.get("dedup", {}) or {}).get("key", "message_raw")
    if dedup_key not in df.columns:
        raise ValueError(f"dedup.key='{dedup_key}' not found. Choose from: {list(df.columns)}")

    df_unique, _ = build_unique_corpus(df, dedup_key)

    # What we embed/cluster on
    docs_unique = df_unique["doc_text"].tolist()

    # Optional prefixing (for E5)
    tp = cfg.get("text_prefix", {}) or {}
    if bool(tp.get("enabled", False)):
        prefix = str(tp.get("prefix", ""))
        docs_unique = [prefix + d for d in docs_unique]

    # Encoder + embeddings
    embedder = build_embedder(cfg)
    enc_cfg = cfg.get("encoder", {}) or {}
    batch_size = int(enc_cfg.get("batch_size", 64))
    normalize_embeddings = bool(enc_cfg.get("normalize_embeddings", True))

    embeddings = embedder.encode(
        docs_unique,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )

    # Vectorizer
    vectorizer = build_vectorizer(cfg)

    # BERTopic
    bt_cfg = cfg.get("bertopic", {}) or {}
    umap_model = build_umap(cfg)
    hdbscan_model = build_hdbscan(cfg)

    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=bt_cfg.get("min_topic_size", 10),
        nr_topics=bt_cfg.get("nr_topics", None),
        calculate_probabilities=bt_cfg.get("calculate_probabilities", True),
        verbose=bt_cfg.get("verbose", True),
    )

    topics, probs = topic_model.fit_transform(docs_unique, embeddings)

    # Optional: reduce topics
    post = cfg.get("postprocess", {}) or {}
    if post.get("reduce_topics_to") is not None:
        topic_model.reduce_topics(docs_unique, nr_topics=int(post["reduce_topics_to"]))
        doc_info = topic_model.get_document_info(docs_unique)
        topics = doc_info["Topic"].astype(int).tolist()

    # Optional: reduce outliers
    if bool(post.get("reduce_outliers", False)):
        doc_info = topic_model.get_document_info(docs_unique)
        strategy = post.get("reduce_outliers_strategy", "probabilities")
        try:
            new_topics = topic_model.reduce_outliers(
                docs_unique,
                doc_info["Topic"].tolist(),
                probabilities=probs,
                strategy=strategy,
            )
        except TypeError:
            new_topics = topic_model.reduce_outliers(
                docs_unique,
                doc_info["Topic"].tolist(),
                strategy=strategy,
            )

        if hasattr(new_topics, "tolist"):
            new_topics = new_topics.tolist()
        topics = [int(t) for t in new_topics]
    else:
        topics = [int(t) for t in topics]

    # Assign to unique df
    df_unique["topic"] = topics
    df_unique["topic_prob"] = probs.max(axis=1) if probs is not None else None

    # Outputs: duplicates + sizes
    topic_sizes_df = compute_topic_sizes(df_unique)
    duplicates_df = df_unique[[dedup_key, "count_total"]].sort_values(
        "count_total", ascending=False
    ).reset_index(drop=True)

    # Save toggles
    save_cfg = cfg.get("save", {}) or {}
    write_csv = bool(save_cfg.get("write_csv", True))
    write_json = bool(save_cfg.get("write_json", True))
    save_model = bool(save_cfg.get("save_model", True))
    model_dirname = save_cfg.get("model_dirname", "bertopic_model")

    # doc_topics (UNIQUE)
    if write_csv:
        df_unique.to_csv(os.path.join(outdir, "doc_topics_unique.csv"), index=False)
        topic_sizes_df.to_csv(os.path.join(outdir, "topic_sizes.csv"), index=False)
        duplicates_df.to_csv(os.path.join(outdir, "duplicates.csv"), index=False)

    if write_json:
        write_json_records(df_unique, os.path.join(outdir, "doc_topics_unique.json"))
        write_json_records(topic_sizes_df, os.path.join(outdir, "topic_sizes.json"))
        write_json_records(duplicates_df, os.path.join(outdir, "duplicates.json"))

    # topic_info (CSV + JSON)
    topic_info_df = topic_model.get_topic_info()
    if write_csv:
        topic_info_df.to_csv(os.path.join(outdir, "topic_info.csv"), index=False)
    if write_json:
        topic_info_df.to_json(
            os.path.join(outdir, "topic_info.json"),
            orient="records",
            indent=2,
        )

    # Save model
    if save_model:
        topic_model.save(os.path.join(outdir, model_dirname))

    # Per-topic commit message files (UNIQUE)
    write_cluster_texts(df_unique, outdir, cfg)

    print(f"Done. Outputs written to: {outdir}")
    print("- doc_topics_unique.(csv/json)   (unique rows + count_total)")
    print("- topic_sizes.(csv/json)         (topic -> size_unique, size_total)")
    print("- duplicates.(csv/json)          (dedup key + count_total)")
    print("- topic_info.(csv/json)")
    print(f"- clusters/: {os.path.join(outdir, 'clusters')}")


if __name__ == "__main__":
    main()
