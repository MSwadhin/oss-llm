#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dawid–Skene EM resolver (self-contained, no CLI).
- Unsupervised DS (no gold labels)
- Optional semi-supervised DS (with gold labels)
- Confidence scores and top-2 margins
- Works with missing labels
"""
from __future__ import annotations

import math
import csv
from collections import Counter
from typing import Dict, List, Tuple, Optional
import re


# ---------------------------------------------------------------------
# Core: Dawid–Skene (unsupervised)
# ---------------------------------------------------------------------
def dawid_skene_unsupervised(
    items: List[str],
    annotators: List[str],
    classes: List[str],
    labels_by_item: Dict[str, List[Tuple[str, str]]],
    max_iters: int = 200,
    tol: float = 1e-7,
    laplace: float = 1e-2,
    diagonal_bias: float = 0.6,   # initial bias on confusion matrix diagonal
    init: str = "mv+priors",      # "mv+priors" | "mv" | "uniform"
    prior_mix_uniform: float = 0.0,  # e.g., 0.2 -> priors = 0.8*empirical + 0.2/K
):
    """
    Returns:
      posteriors: dict item -> dict class -> prob
      hard: dict item -> (pred_label, confidence, top2_margin)
      confusions: dict annotator -> dict true_class -> dict observed_class -> prob
      priors: dict class -> prob
      ll_hist: list of log-likelihoods per iteration
    """
    K = len(classes)
    classes_set = set(classes)

    # --- Priors from empirical vote distribution (smoothed) ---
    vote_counts = Counter()
    for i in items:
        for (_a, l) in labels_by_item.get(i, []):
            if l in classes_set:
                vote_counts[l] += 1
    total_votes = sum(vote_counts.values())
    if total_votes > 0:
        priors = {c: (vote_counts.get(c, 0) + 1.0) / (total_votes + K) for c in classes}
    else:
        priors = {c: 1.0 / K for c in classes}
    if prior_mix_uniform > 0:
        u = 1.0 / K
        priors = {c: (1 - prior_mix_uniform) * priors[c] + prior_mix_uniform * u for c in classes}

    # --- Initialize posteriors q_i(c) ---
    post: Dict[str, Dict[str, float]] = {i: {c: 0.0 for c in classes} for i in items}
    for i in items:
        obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
        if not obs:
            for c in classes:
                post[i][c] = priors[c]
            continue

        if init == "uniform":
            for c in classes:
                post[i][c] = 1.0 / K
        else:
            cnt = Counter(l for (_a, l) in obs)
            total = sum(cnt.values())
            if total == 0:
                for c in classes:
                    post[i][c] = priors[c]
            else:
                if init == "mv+priors":
                    scr = {c: (cnt.get(c, 0) / total) * max(priors[c], 1e-12) for c in classes}
                else:  # "mv"
                    scr = {c: (cnt.get(c, 0) / total) for c in classes}
                s = sum(scr.values()) or 1.0
                for c in classes:
                    post[i][c] = scr[c] / s

    # --- Initialize confusions with diagonal bias ---
    confusions: Dict[str, Dict[str, Dict[str, float]]] = {
        a: {c: {l: 0.0 for l in classes} for c in classes} for a in annotators
    }
    off = (1.0 - diagonal_bias) / max(K - 1, 1)
    for a in annotators:
        for c in classes:
            for l in classes:
                confusions[a][c][l] = diagonal_bias if l == c else off

    def e_step(priors_local, conf_local):
        new_post: Dict[str, Dict[str, float]] = {}
        for i in items:
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            if not obs:
                new_post[i] = dict(priors_local)
                continue
            scores = {}
            for c in classes:
                s = math.log(max(priors_local[c], 1e-12))
                for (a, l) in obs:
                    s += math.log(max(conf_local[a][c][l], 1e-12))
                scores[c] = s
            m = max(scores.values())
            exps = {c: math.exp(scores[c] - m) for c in classes}
            Z = sum(exps.values())
            new_post[i] = {c: exps[c] / Z for c in classes}
        return new_post

    def m_step(posterior):
        # Priors
        pri = {c: 0.0 for c in classes}
        tot = 0.0
        for i in items:
            for c in classes:
                pri[c] += posterior[i][c]
                tot += posterior[i][c]
        if tot > 0:
            for c in classes:
                pri[c] /= tot
        if prior_mix_uniform > 0:
            u = 1.0 / K
            pri = {c: (1 - prior_mix_uniform) * pri[c] + prior_mix_uniform * u for c in classes}

        # Confusions
        conf = {a: {c: {l: laplace for l in classes} for c in classes} for a in annotators}
        for i in items:
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            for (a, l) in obs:
                for c in classes:
                    conf[a][c][l] += posterior[i][c]
        # normalize per (a, true c)
        for a in annotators:
            for c in classes:
                row_sum = sum(conf[a][c].values())
                for l in classes:
                    conf[a][c][l] /= max(row_sum, 1e-12)
        return pri, conf

    def loglik(priors_local, conf_local):
        ll = 0.0
        for i in items:
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            if not obs:
                ll += math.log(sum(priors_local[c] for c in classes))
                continue
            terms = []
            for c in classes:
                s = math.log(max(priors_local[c], 1e-12))
                for (a, l) in obs:
                    s += math.log(max(conf_local[a][c][l], 1e-12))
                terms.append(s)
            m = max(terms)
            ll += m + math.log(sum(math.exp(t - m) for t in terms))
        return ll

    ll_hist = [loglik(priors, confusions)]
    for _ in range(max_iters):
        post_new = e_step(priors, confusions)
        priors, confusions = m_step(post_new)
        post = post_new
        ll_new = loglik(priors, confusions)
        ll_hist.append(ll_new)
        if abs(ll_new - ll_hist[-2]) < tol:
            break

    # Hard labels + margins
    hard = {}
    for i in items:
        dist = post[i]
        best = max(dist, key=dist.get)
        best_p = dist[best]
        top2 = sorted(dist.values(), reverse=True)[:2] + [0.0]
        margin = top2[0] - top2[1]
        hard[i] = (best, best_p, margin)

    return post, hard, confusions, priors, ll_hist


# ---------------------------------------------------------------------
# Optional: Dawid–Skene (semi-supervised, with gold labels)
# ---------------------------------------------------------------------
def dawid_skene_semisupervised(
    items: List[str],
    annotators: List[str],
    classes: List[str],
    labels_by_item: Dict[str, List[Tuple[str, str]]],
    gold_labels: Dict[str, str],
    max_iters: int = 200,
    tol: float = 1e-7,
    laplace: float = 1e-2,
    init: str = "mv+priors",
    prior_mix_uniform: float = 0.0,
):
    K = len(classes)
    classes_set = set(classes)

    if gold_labels:
        cnt = Counter(gold_labels.values())
        total = sum(cnt.values())
        priors = {c: (cnt.get(c, 0) + 1.0) / (total + K) for c in classes}
    else:
        priors = {c: 1.0 / K for c in classes}
    if prior_mix_uniform > 0:
        u = 1.0 / K
        priors = {c: (1 - prior_mix_uniform) * priors[c] + prior_mix_uniform * u for c in classes}

    post: Dict[str, Dict[str, float]] = {i: {c: 0.0 for c in classes} for i in items}
    for i in items:
        if i in gold_labels:
            g = gold_labels[i]
            for c in classes:
                post[i][c] = 1.0 if c == g else 0.0
            continue

        obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
        if not obs:
            for c in classes:
                post[i][c] = priors[c]
            continue

        if init == "uniform":
            for c in classes:
                post[i][c] = 1.0 / K
        else:
            cnt = Counter(l for (_a, l) in obs)
            total = sum(cnt.values())
            if total == 0:
                for c in classes:
                    post[i][c] = priors[c]
            else:
                if init == "mv+priors":
                    scr = {c: (cnt.get(c, 0) / total) * max(priors[c], 1e-12) for c in classes}
                else:
                    scr = {c: (cnt.get(c, 0) / total) for c in classes}
                s = sum(scr.values()) or 1.0
                for c in classes:
                    post[i][c] = scr[c] / s

    confusions: Dict[str, Dict[str, Dict[str, float]]] = {
        a: {c: {l: laplace for l in classes} for c in classes} for a in annotators
    }
    for i in items:
        obs = labels_by_item.get(i, [])
        for (a, l) in obs:
            if l not in classes_set:
                continue
            for c in classes:
                confusions[a][c][l] += post[i][c]
    for a in annotators:
        for c in classes:
            row_sum = sum(confusions[a][c].values())
            for l in classes:
                confusions[a][c][l] /= max(row_sum, 1e-12)

    def e_step(priors_local, conf_local):
        new_post: Dict[str, Dict[str, float]] = {}
        for i in items:
            if i in gold_labels:
                g = gold_labels[i]
                new_post[i] = {c: (1.0 if c == g else 0.0) for c in classes}
                continue
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            if not obs:
                new_post[i] = dict(priors_local)
                continue
            scores = {}
            for c in classes:
                s = math.log(max(priors_local[c], 1e-12))
                for (a, l) in obs:
                    s += math.log(max(conf_local[a][c][l], 1e-12))
                scores[c] = s
            m = max(scores.values())
            exps = {c: math.exp(scores[c] - m) for c in classes}
            Z = sum(exps.values())
            new_post[i] = {c: exps[c] / Z for c in classes}
        return new_post

    def m_step(posterior):
        pri = {c: 0.0 for c in classes}
        tot = 0.0
        for i in items:
            for c in classes:
                pri[c] += posterior[i][c]
                tot += posterior[i][c]
        if tot > 0:
            for c in classes:
                pri[c] /= tot
        if prior_mix_uniform > 0:
            u = 1.0 / K
            pri = {c: (1 - prior_mix_uniform) * pri[c] + prior_mix_uniform * u for c in classes}

        conf = {a: {c: {l: laplace for l in classes} for c in classes} for a in annotators}
        for i in items:
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            for (a, l) in obs:
                for c in classes:
                    conf[a][c][l] += posterior[i][c]
        for a in annotators:
            for c in classes:
                row_sum = sum(conf[a][c].values())
                for l in classes:
                    conf[a][c][l] = conf[a][c][l] / max(row_sum, 1e-12)
        return pri, conf

    def loglik(priors_local, conf_local):
        ll = 0.0
        for i in items:
            obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set]
            if not obs:
                ll += math.log(sum(priors_local[c] for c in classes))
                continue
            terms = []
            for c in classes:
                s = math.log(max(priors_local[c], 1e-12))
                for (a, l) in obs:
                    s += math.log(max(conf_local[a][c][l], 1e-12))
                terms.append(s)
            m = max(terms)
            ll += m + math.log(sum(math.exp(t - m) for t in terms))
        return ll

    ll_hist = [loglik(priors, confusions)]
    for _ in range(max_iters):
        post_new = e_step(priors, confusions)
        priors, confusions = m_step(post_new)
        post = post_new
        ll_new = loglik(priors, confusions)
        ll_hist.append(ll_new)
        if abs(ll_new - ll_hist[-2]) < tol:
            break

    hard = {}
    for i in items:
        dist = post[i]
        best = max(dist, key=dist.get)
        best_p = dist[best]
        top2 = sorted(dist.values(), reverse=True)[:2] + [0.0]
        margin = top2[0] - top2[1]
        hard[i] = (best, best_p, margin)

    return post, hard, confusions, priors, ll_hist


# ---------------------------------------------------------------------
# Inference-only (if you already have confusions + priors)
# ---------------------------------------------------------------------
def infer_annotations(
    items: List[str],
    classes: List[str],
    labels_by_item: Dict[str, List[Tuple[str, str]]],
    confusions: Dict[str, Dict[str, Dict[str, float]]],
    priors: Dict[str, float],
    abstain_thresh: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[str, float, float]]]:
    """
    One-pass inference given trained confusions+priors.
    Returns:
      posteriors: item -> {class: prob}
      hard: item -> (pred_label, confidence, top2_margin)  (or 'ABSTAIN' if threshold set)
    """
    classes_set = set(classes)
    posteriors: Dict[str, Dict[str, float]] = {}
    hard: Dict[str, Tuple[str, float, float]] = {}

    for i in items:
        obs = [(a, l) for (a, l) in labels_by_item.get(i, []) if l in classes_set and a in confusions]
        scores = {}
        for k in classes:
            s = math.log(max(priors.get(k, 0.0), 1e-12))
            for (a, l) in obs:
                s += math.log(max(confusions[a][k].get(l, 0.0), 1e-12))
            scores[k] = s
        m = max(scores.values())
        exps = {k: math.exp(scores[k] - m) for k in classes}
        Z = sum(exps.values())
        post = {k: exps[k] / Z for k in classes}
        posteriors[i] = post

        # hard + margin
        best = max(post, key=post.get)
        best_p = post[best]
        top2 = sorted(post.values(), reverse=True)[:2] + [0.0]
        margin = top2[0] - top2[1]
        if abstain_thresh is not None and best_p < abstain_thresh:
            hard[i] = ("ABSTAIN", best_p, margin)
        else:
            hard[i] = (best, best_p, margin)

    return posteriors, hard


# ---------------------------------------------------------------------
# Helper: write predictions to CSV
# ---------------------------------------------------------------------
def write_predictions_csv(
    path: str,
    items: List[str],
    hard: Dict[str, Tuple[str, float, float]],
    posteriors: Dict[str, Dict[str, float]],
    classes: List[str],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "pred_label", "confidence", "top2_margin"] + [f"p_{c}" for c in classes])
        for i in items:
            pred_label, conf, margin = hard[i]
            row = [i, pred_label, f"{conf:.6f}", f"{margin:.6f}"] + [f"{posteriors[i][c]:.6f}" for c in classes]
            w.writerow(row)





import json
from typing import Dict, List, Tuple

def write_predictions_json(
    path: str,
    items: List[str],
    hard: Dict[str, Tuple[str, float, float]],
    posteriors: Dict[str, Dict[str, float]],
    classes: List[str],
) -> None:
    """
    Write predictions to a JSON file as a list of records.
    Each record: {
      "item_id": ...,
      "pred_label": ...,
      "confidence": float (rounded to 6 dp),
      "top2_margin": float (rounded to 6 dp),
      "posteriors": {class: prob (rounded to 6 dp), ...}
    }
    """
    data = []
    for i in items:
        pred_label, conf, margin = hard[i]
        rec = {
            "item_id": i,
            "pred_label": pred_label,
            "confidence": round(conf, 6)
        }
        data.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Example usage (dummy inputs) — modify this for your data
# ---------------------------------------------------------------------


mistral_output = "DS/FULL-DS-LLM/code_comment_category-output-mistral.csv"
gpt_oss_output = "DS/FULL-DS-LLM/code_comment_category-gpt-oss-output.csv"


mistral_output = "DS/FULL-DS-LLM/rq1b/mistral.csv"
gpt_oss_output = "DS/FULL-DS-LLM/rq1b/gpt-oss.csv"

def get_data():


    def load_comments_and_categories(path):

        dic = {}
        cnt = 0
        with open(path,"r") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                comment = row[3]
                json_output = row[6]
                # print(json_output)
                
                outs = json_output.split(":")
                if len(outs) != 2:
                    cnt += 1
                    continue
                cat = re.sub(r'^[\W_]+|[\W_]+$', '', outs[1])

                if not comment in dic:
                    dic[comment] = cat
                else:
                    if dic[comment] != cat:
                        print(f"Differs for comment: {comment} prev : {dic[comment]}  new : {cat}")
        return dic

    gpts = load_comments_and_categories(gpt_oss_output)
    mistrals = load_comments_and_categories(mistral_output)


    classes = []
    class_seen = {}
    annotators = ["gpt_oss","mistral"]
    labels_by_item = {}
    for k,v in gpts.items():
        if k not in mistrals:
            continue
        if v not in class_seen:
            classes.append(v)
            class_seen[v] =  1
        val = [
            ("gpt_oss",v),
            ("mistral",mistrals[k])
        ]
        labels_by_item[k] = val
    return classes, annotators, labels_by_item




def main():


    classes, annotators, labels_by_item = get_data()
    print(classes)

    items = list(labels_by_item.keys())

    # ---------------- 1) Fit Unsupervised DS ----------------
    post_train, hard_train, conf, priors, ll_hist = dawid_skene_unsupervised(
        items, annotators, classes, labels_by_item,
        max_iters=200, tol=1e-7, laplace=1e-2,
        diagonal_bias=0.6, init="mv+priors", prior_mix_uniform=0.0
    )

    # ---------------- 2) Inference (final annotations) ----------------
    # Here we infer on the same items for simplicity; in practice this can be a new batch.
    post_final, hard_final = infer_annotations(
        items=items,
        classes=classes,
        labels_by_item=labels_by_item,
        confusions=conf,
        priors=priors,
        abstain_thresh=None,  # e.g., set to 0.6 to emit 'ABSTAIN' for low-confidence
    )

    # ---------------- 3) Write to CSV ----------------
    # out_path = "DS/FULL-DS-LLM/dawid-skene-annotations.csv"
    # write_predictions_csv(out_path, items, hard_final, post_final, classes)
    print("Calculations Done!!!!!!")
    out_path_json = "DS/FULL-DS-LLM/rq1b/dawid-skene-annotations.json"
    write_predictions_json(out_path_json, items, hard_final, post_final, classes)

    # (Optional) quick console peek
    # print(f"Wrote final annotations to: {out_path}")
    # for i in items:
    #     y, p, m = hard_final[i]
    #     print(f"{i:8s} -> {y:7s}  p={p:.3f}  margin={m:.3f}")


if __name__ == "__main__":
    main()
