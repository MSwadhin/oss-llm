#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Embeddings
from sentence_transformers import SentenceTransformer, models

import umap
import hdbscan

# NLP (optional)
import spacy
from nltk.stem import PorterStemmer

PR_TAIL_RE = re.compile(r"\s*\(#\d+\)\s*$")
HASHNUM_RE = re.compile(r"#\d+\b")


# ---------------------------
# Config + IO
# ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Preprocessing (config-driven)
# ---------------------------
def build_spacy_if_needed(cfg: Dict[str, Any]):
    """
    Only load spaCy if NLP is enabled AND at least one of lemmatize/stem is true.
    """
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

    # Basic normalization
    if pre.get("lowercase", True):
        s = s.lower()

    if pre.get("strip_pr_issue_refs", True):
        s = PR_TAIL_RE.sub("", s)
        s = HASHNUM_RE.sub("", s)

    s = re.sub(r"\s+", " ", s).strip()
    min_chars = int(pre.get("min_chars", 2))
    if len(s) < min_chars:
        return ""

    # NLP ops only if enabled and something selected
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

        rows.append(
            {
                "doc_key": doc_key,
                "sha": obj.get("sha", ""),
                "author": obj.get("author", ""),
                "date": obj.get("date", ""),
                "message_raw": raw,
                "message": processed,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------
# Encoders (config-driven)
# ---------------------------
def build_hf_sentence_transformer(model_name: str, pooling: str) -> SentenceTransformer:
    """
    Build SentenceTransformer from *any* HuggingFace encoder + explicit pooling.
    Pooling: mean or cls
    """
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

    # Presets (code/SE-ish HF encoders)
    presets = {
        "unixcoder": "microsoft/unixcoder-base",
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
        "codet5": "Salesforce/codet5-base",
    }

    if enc_type in presets:
        model_name = enc.get("model_name", presets[enc_type]) or presets[enc_type]
        return build_hf_sentence_transformer(model_name, pooling)

    if enc_type == "hf_transformer":
        model_name = enc["model_name"]
        return build_hf_sentence_transformer(model_name, pooling)

    if enc_type == "sentence_transformer":
        # ST-native models (e.g., sentence-transformers/*, bge/e5 often work too)
        model_name = enc["model_name"]
        return SentenceTransformer(model_name)

    raise ValueError(
        "encoder.type must be one of: "
        "unixcoder, codebert, graphcodebert, codet5, hf_transformer, sentence_transformer"
    )


# ---------------------------
# Vectorizer + cluster exports
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


def write_cluster_texts(df: pd.DataFrame, outdir: str, cfg: Dict[str, Any]) -> None:
    c_cfg = cfg.get("clusters_output", {}) or {}
    message_field = c_cfg.get("message_field", "message_raw")
    include_outliers = bool(c_cfg.get("include_outliers", True))

    clusters_dir = os.path.join(outdir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    topics = sorted(df["topic"].unique().tolist(), key=lambda x: (int(x) == -1, int(x)))
    for t in topics:
        t = int(t)
        if t == -1 and not include_outliers:
            continue

        path = os.path.join(clusters_dir, f"topic_{t}.txt")
        msgs = df.loc[df["topic"] == t, message_field].astype(str).tolist()

        with open(path, "w", encoding="utf-8") as f:
            for m in msgs:
                m = m.replace("\n", " ").strip()
                if m:
                    f.write(m + "\n")


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
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BERTopic commit clustering via config.json")
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = cfg.get("outdir", "out_bertopic")
    ensure_outdir(outdir)

    df = load_documents(cfg)
    if df.empty:
        raise SystemExit("No usable commit messages found after preprocessing.")

    docs = df["message"].tolist()

    # Embedder + embeddings
    embedder = build_embedder(cfg)
    enc_cfg = cfg.get("encoder", {}) or {}
    batch_size = int(enc_cfg.get("batch_size", 64))
    normalize_embeddings = bool(enc_cfg.get("normalize_embeddings", True))

    embeddings = embedder.encode(
        docs,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    )

    # Vectorizer (optional)
    vectorizer = build_vectorizer(cfg)

    # BERTopic (defaults: UMAP + HDBSCAN) unless you override these fields
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

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Postprocess
    post = cfg.get("postprocess", {}) or {}
    if post.get("reduce_topics_to") is not None:
        topic_model.reduce_topics(docs, nr_topics=int(post["reduce_topics_to"]))
        doc_info = topic_model.get_document_info(docs)
        topics = doc_info["Topic"].astype(int).tolist()

    if bool(post.get("reduce_outliers", False)):
        doc_info = topic_model.get_document_info(docs)

        new_topics = topic_model.reduce_outliers(
            docs,
            doc_info["Topic"].tolist(),
            probabilities=probs,
            strategy=post.get("reduce_outliers_strategy", "probabilities"),
        )

        # make it a plain Python list
        if hasattr(new_topics, "tolist"):
            new_topics = new_topics.tolist()

        topics = [int(t) for t in new_topics]
    else:
        topics = [int(t) for t in topics]
    
    df["topic"] = topics
    df["topic_prob"] = probs.max(axis=1) if probs is not None else None

    # Save settings
    save_cfg = cfg.get("save", {}) or {}
    write_csv = bool(save_cfg.get("write_csv", True))
    write_json = bool(save_cfg.get("write_json", True))
    save_model = bool(save_cfg.get("save_model", True))
    model_dirname = save_cfg.get("model_dirname", "bertopic_model")

    # doc_topics
    if write_csv:
        df.to_csv(os.path.join(outdir, "doc_topics.csv"), index=False)
    if write_json:
        with open(os.path.join(outdir, "doc_topics.json"), "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)

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

    # Per-topic commit message files
    write_cluster_texts(df, outdir, cfg)

    print(f"Done. Outputs written to: {outdir}")
    print(f"- doc_topics.(csv/json), topic_info.(csv/json)")
    print(f"- per-topic message files: {os.path.join(outdir, 'clusters')}")
    if save_model:
        print(f"- model: {os.path.join(outdir, model_dirname)}")


if __name__ == "__main__":
    main()
