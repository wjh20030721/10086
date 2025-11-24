#!/usr/bin/env python3
"""Compute per-type (Gray Industry Detection/Description/Analysis) precision & recall
from evaluation CSV without external ML libraries.

Usage:
  python compute_pr_by_type.py [results_csv] [annotation_json]

Defaults:
  results_csv     = Results_Qwen2VL_test.csv
  annotation_json = ../Annotation/test.json
"""

import csv
import sys
from pathlib import Path
from collections import Counter
import json

GRAY_TYPES = [
    "Gray Industry Detection",
    "Gray Industry Description",
    "Gray Industry Analysis",
]


def extract_answer(predicted: str) -> str:
    if not predicted or predicted == "N/A":
        return "N/A"
    for ch in predicted:
        if ch.isalpha():
            return ch.upper()
    return predicted


def build_question_type_map(annotation_path: Path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    mapping = {}
    for img_key, entry in data.items():
        for conv in entry.get("conversation", []):
            t = conv.get("type")
            if isinstance(t, str) and t in GRAY_TYPES:
                q = (conv.get("Question", "") or "").strip()
                mapping[(img_key, q)] = t
    return mapping


def compute_basic_metrics(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    TP, FP, FN = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            TP[t] += 1
        else:
            FP[p] += 1
            FN[t] += 1
    per_class = {}
    sum_tp = sum(TP.values())
    sum_fp = sum(FP.values())
    sum_fn = sum(FN.values())
    for c in labels:
        tp, fp, fn = TP[c], FP[c], FN[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        support = tp + fn
        per_class[c] = (prec, rec, support)
    macro_prec = sum(p for p, r, s in per_class.values()) / len(labels) if labels else 0.0
    macro_rec = sum(r for p, r, s in per_class.values()) / len(labels) if labels else 0.0
    micro_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    acc = sum_tp / len(y_true) if y_true else 0.0
    return per_class, macro_prec, macro_rec, micro_prec, micro_rec, acc


def main():
    if len(sys.argv) > 1:
        results_csv = Path(sys.argv[1])
    else:
        results_csv = Path(__file__).resolve().parent / "Results_Qwen2VL_test.csv"
    if len(sys.argv) > 2:
        annotation_json = Path(sys.argv[2])
    else:
        annotation_json = Path(__file__).resolve().parent.parent / "Annotation" / "test.json"

    if not results_csv.exists():
        print(f"[ERROR] Results CSV not found: {results_csv}")
        sys.exit(1)
    if not annotation_json.exists():
        print(f"[ERROR] Annotation JSON not found: {annotation_json}")
        sys.exit(1)

    qmap = build_question_type_map(annotation_json)
    by_type = {t: {"y_true": [], "y_pred": []} for t in GRAY_TYPES}
    unmatched = 0
    total_rows = 0

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            img = row.get("Image Path", "")
            q_raw = (row.get("Question", "") or "").strip()
            pred_clean = extract_answer(row.get("Predicted Answer", ""))
            gt = row.get("Correct Answer", "")
            t = qmap.get((img, q_raw))
            if not t:
                unmatched += 1
                continue
            if pred_clean == "N/A":
                continue
            by_type[t]["y_true"].append(gt)
            by_type[t]["y_pred"].append(pred_clean)

    print("Loaded rows:", total_rows, "/ Gray industry matched rows:", sum(len(v["y_true"]) for v in by_type.values()), "/ Unmatched:", unmatched)

    for t in GRAY_TYPES:
        vals = by_type[t]
        y_true, y_pred = vals["y_true"], vals["y_pred"]
        if not y_true:
            print(f"\n=== {t} ===\n[WARN] No samples.")
            continue
        pc, mp, mr, mip, mir, acc = compute_basic_metrics(y_true, y_pred)
        print(f"\n=== {t} ===")
        print(f"Samples: {len(y_true)} | Accuracy: {acc:.4f}")
        print(f"Precision (macro): {mp:.4f} | Recall (macro): {mr:.4f}")
        print(f"Precision (micro): {mip:.4f} | Recall (micro): {mir:.4f}")
        print("Per-class:")
        for c in sorted(pc.keys()):
            p, r, s = pc[c]
            print(f"  {c}: support={s:4d} | precision={p:.4f} | recall={r:.4f}")


if __name__ == "__main__":
    main()
