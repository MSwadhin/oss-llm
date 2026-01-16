#!/usr/bin/env python3
"""
BERTopic on commit messages (code-aware embeddings) with config.json
+ optional lemmatization/stemming
+ outputs CSV/JSON/model
+ ALSO writes one text file per topic: commit messages (1 per line)

Files written under outdir:
- doc_topics.csv
- topic_info.csv
- doc_topics.json
- bertopic_model/   (optional)
- clusters/
    - topic_<ID>.txt   (one commit message per line, raw by default)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# NLP (optional)
import spacy
from nltk.stem import PorterStemmer

PR_TAIL_RE = re.compile(r"\s*\(#\d+\)\s*$")
HASHNUM_RE = re.compile(r"#\d+\b")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_nlp(pre_cfg: Dict[str, Any]):
    nlp_cfg = pre_cfg.get("nlp", {})
    if not nlp_cfg.get("enabled", False):
        return None, None
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    stemmer = PorterStemmer()
    return nlp, stemmer


def preprocess_text(text: str, cfg: Dict[str, Any], nlp, stemmer) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    pre = cfg["preprocess"]

    if pre.get("lowercase", True):
        s = s.lower()

    if pre.get("strip_pr_issue_refs", True):
        s = PR_TAIL_RE.sub("", s)
        s = HASHNUM_RE.sub("", s)

    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < int(pre.get("min_chars", 2)):
        return ""

    nlp_cfg = pre.get("nlp", {})
    if not nlp_cfg.get("enabled", False):
        return s

    doc = nlp(s)
    tokens: List[str] = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue

        out = tok.text
        if nlp_cfg.get("lemmatize", False):
            out = tok.lemma_

        if nlp_cfg.get("stem", False):
            out = stemmer.stem(out)

        out = out.strip()
        if len(out) >= 2:
            tokens.append(out)

    return " ".join(tokens)


def load_documents(cfg: Dict[str, Any]) -> pd.DataFrame:
    with open(cfg["input_json"], "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    nlp, stemmer = build_nlp(cfg["preprocess"])

    rows = []
    text_source = cfg.get("text_source", "message")

    for key, obj in data.items():
        raw = (obj.get(text_source) or "").strip()
        proc = preprocess_text(raw, cfg, nlp, stemmer)
        if not proc:
            continue

        rows.append(
            {
                "doc_key": key,
                "sha": obj.get("sha", ""),
                "author": obj.get("author", ""),
                "date": obj.get("date", ""),
                "message_raw": raw,
                "message": proc,
            }
        )

    return pd.DataFrame(rows)


def write_cluster_texts(
    df: pd.DataFrame,
    outdir: str,
    topic_col: str = "topic",
    message_col: str = "message_raw",
    include_outliers: bool = True,
) -> None:
    """
    Writes one file per topic with one message per line.
    Default uses raw messages (best for readability).
    """
    clusters_dir = os.path.join(outdir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    topics = sorted(df[topic_col].unique().tolist(), key=lambda x: (x == -1, x))
    for t in topics:
        if (not include_outliers) and int(t) == -1:
            continue
        fname = f"topic_{int(t)}.txt"
        path = os.path.join(clusters_dir, fname)

        msgs = df.loc[df[topic_col] == t, message_col].astype(str).tolist()
        # keep one per line, preserve order as in df
        with open(path, "w", encoding="utf-8") as f:
            for m in msgs:
                m = m.replace("\n", " ").strip()
                if m:
                    f.write(m + "\n")


def main():
    import argparse

    ap = argparse.ArgumentParser("BERTopic on commit messages (config-driven) + per-cluster txt")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = cfg.get("outdir", "out_bertopic")
    os.makedirs(outdir, exist_ok=True)

    df = load_documents(cfg)
    if df.empty:
        raise SystemExit("No usable commit messages found.")

    docs = df["message"].tolist()

    # Embeddings
    emb_cfg = cfg["embedding"]
    embedder = SentenceTransformer(emb_cfg["model_name"])
    embeddings = embedder.encode(
        docs,
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        show_progress_bar=True,
    )

    # Vectorizer
    vec_cfg = cfg.get("vectorizer", {}) or {}
    vectorizer = None
    if vec_cfg.get("enabled", False):
        vectorizer = CountVectorizer(
            stop_words=vec_cfg.get("stop_words", "english"),
            ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
            min_df=int(vec_cfg.get("min_df", 2)),
        )

    # BERTopic
    bt_cfg = cfg.get("bertopic", {}) or {}
    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        min_topic_size=bt_cfg.get("min_topic_size", 10),
        nr_topics=bt_cfg.get("nr_topics", None),
        calculate_probabilities=bt_cfg.get("calculate_probabilities", True),
        verbose=bt_cfg.get("verbose", True),
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Optional postprocessing
    post = cfg.get("postprocess", {}) or {}
    if post.get("reduce_topics_to") is not None:
        topic_model.reduce_topics(docs, nr_topics=int(post["reduce_topics_to"]))
        # refresh topics after reduce_topics
        doc_info = topic_model.get_document_info(docs)
        topics = doc_info["Topic"].astype(int).tolist()

    if post.get("reduce_outliers", False):
        doc_info = topic_model.get_document_info(docs)
        topics = topic_model.reduce_outliers(
            docs,
            doc_info["Topic"].tolist(),
            strategy=post.get("reduce_outliers_strategy", "nearest"),
        )

    df["topic"] = [int(t) for t in topics]
    df["topic_prob"] = probs.max(axis=1) if probs is not None else None

    save_cfg = cfg.get("save", {}) or {}

    # CSV outputs
    if save_cfg.get("write_csv", True):
        df.to_csv(os.path.join(outdir, "doc_topics.csv"), index=False)
        topic_model.get_topic_info().to_csv(os.path.join(outdir, "topic_info.csv"), index=False)


    # JSON outputs
    if save_cfg.get("write_json", True):
        with open(os.path.join(outdir, "doc_topics.json"), "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
        topic_model.get_topic_info().to_json(
            os.path.join(outdir, "topic_info.json"),
            orient="records",
            indent=2
        )

    # Model
    if save_cfg.get("save_model", True):
        model_dir = os.path.join(outdir, save_cfg.get("model_dirname", "bertopic_model"))
        topic_model.save(model_dir)

    # Per-topic message text files
    clusters_cfg = cfg.get("clusters_output", {}) or {}
    write_cluster_texts(
        df=df,
        outdir=outdir,
        topic_col="topic",
        message_col=clusters_cfg.get("message_field", "message_raw"),  # "message" for processed
        include_outliers=bool(clusters_cfg.get("include_outliers", True)),
    )

    print(f"Done. Outputs written to: {outdir}")
    print("Per-topic commit message files in:", os.path.join(outdir, "clusters"))


if __name__ == "__main__":
    main()
