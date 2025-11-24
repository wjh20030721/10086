#!/usr/bin/env python3
"""
Compute AUROC and AUPRC per question type using Results_Qwen2VL_test.csv that
includes per-option scores (Score_A..Score_D). This script:
 - Joins rows to Gray Industry types via Annotation/test.json (same as compute_pr_by_type)
 - For Detection (binary: A vs B), treats positive class as 'A' (Yes) by default unless
   options indicate otherwise; you can switch with --pos=B.
 - For multi-class (Description/Analysis), reports macro-average One-vs-Rest AUROC/AUPRC.

Usage:
  python compute_auc_metrics.py [results_csv] [annotation_json] [--pos=A|B]
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import math

GRAY_TYPES = [
    "Gray Industry Detection",
    "Gray Industry Description",
    "Gray Industry Analysis",
]

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

def parse_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan

def roc_pr_binary(scores, labels):
    # scores: list of floats (higher -> positive), labels: 0/1
    pairs = [(s, y) for s, y in zip(scores, labels) if not math.isnan(s)]
    if not pairs:
        return math.nan, math.nan
    # Sort by score desc
    pairs.sort(key=lambda t: t[0], reverse=True)
    P = sum(y for _, y in pairs)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return math.nan, math.nan
    tp = 0
    fp = 0
    prev_s = None
    roc_pts = [(0.0, 0.0)]
    pr_pts = []
    for s, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / P
        fpr = fp / N
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tpr
        roc_pts.append((fpr, tpr))
        pr_pts.append((rec, prec))
    # Ensure endpoints
    roc_pts.append((1.0, 1.0))
    pr_pts.append((0.0, P/(P+N)))
    # AUC by trapezoid
    roc_pts_sorted = sorted(roc_pts)
    auroc = 0.0
    for (x1,y1),(x2,y2) in zip(roc_pts_sorted[:-1], roc_pts_sorted[1:]):
        auroc += (x2-x1) * (y1 + y2) / 2
    # PR AUC
    pr_pts_sorted = sorted(pr_pts)
    auprc = 0.0
    for (r1,p1),(r2,p2) in zip(pr_pts_sorted[:-1], pr_pts_sorted[1:]):
        auprc += (r2-r1) * (p1 + p2) / 2
    return auroc, auprc

def ovr_auc(scores_mat, labels, classes):
    # scores_mat: list of dict {class->score}
    # labels: list of gold class labels (e.g., 'A','B','C','D')
    # classes: iterable of classes to include
    aucs = []
    auprs = []
    for c in classes:
        s = [row.get(c, math.nan) for row in scores_mat]
        y = [1 if g == c else 0 for g in labels]
        auroc, auprc = roc_pr_binary(s, y)
        if not math.isnan(auroc):
            aucs.append(auroc)
        if not math.isnan(auprc):
            auprs.append(auprc)
    macro_auroc = sum(aucs)/len(aucs) if aucs else math.nan
    macro_auprc = sum(auprs)/len(auprs) if auprs else math.nan
    return macro_auroc, macro_auprc

def main():
    results_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'Results_Qwen2VL_test.csv'
    annotation_json = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'Annotation' / 'test.json'
    pos_letter = 'A'
    if len(sys.argv) > 3 and sys.argv[3].startswith('--pos='):
        pos_letter = sys.argv[3].split('=',1)[1].strip().upper()

    if not results_csv.exists():
        print(f"[ERROR] Results CSV not found: {results_csv}")
        sys.exit(1)
    if not annotation_json.exists():
        print(f"[ERROR] Annotation JSON not found: {annotation_json}")
        sys.exit(1)

    qmap = build_question_type_map(annotation_json)

    by_type = {t: {"labels": [], "scores": []} for t in GRAY_TYPES}
    total = 0
    skipped_no_type = 0
    skipped_no_scores = 0

    with results_csv.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        has_scores = all(k in reader.fieldnames for k in ['Score_A','Score_B','Score_C','Score_D'])
        if not has_scores:
            print('[ERROR] Results CSV has no Score_* columns. Please re-run Qwen2-VL_test.py to generate scores.')
            sys.exit(2)
        for row in reader:
            total += 1
            img = row.get('Image Path','')
            q = (row.get('Question','') or '').strip()
            t = qmap.get((img, q))
            if not t:
                skipped_no_type += 1
                continue
            label = (row.get('Correct Answer','') or '').strip().upper()
            sA = parse_float(row.get('Score_A',''))
            sB = parse_float(row.get('Score_B',''))
            sC = parse_float(row.get('Score_C',''))
            sD = parse_float(row.get('Score_D',''))
            if all(math.isnan(x) for x in [sA,sB,sC,sD]):
                skipped_no_scores += 1
                continue
            scores = {}
            if not math.isnan(sA): scores['A']=sA
            if not math.isnan(sB): scores['B']=sB
            if not math.isnan(sC): scores['C']=sC
            if not math.isnan(sD): scores['D']=sD
            by_type[t]['labels'].append(label)
            by_type[t]['scores'].append(scores)

    print(f'Total rows: {total} | Skipped (no type): {skipped_no_type} | Skipped (no scores): {skipped_no_scores}')

    for t in GRAY_TYPES:
        labels = by_type[t]['labels']
        scores_list = by_type[t]['scores']
        if not labels:
            print(f"\n=== {t} ===\n[WARN] No samples with scores.")
            continue
        if t == 'Gray Industry Detection':
            # Binary AUROC/AUPRC using pos_letter
            pos = pos_letter
            # choose positive scores, fallback if not present
            pos_scores = [row.get(pos, math.nan) for row in scores_list]
            y = [1 if g == pos else 0 for g in labels]
            auroc, auprc = roc_pr_binary(pos_scores, y)
            print(f"\n=== {t} (pos={pos}) ===")
            print(f"AUROC: {auroc if not math.isnan(auroc) else 'N/A'}")
            print(f"AUPRC: {auprc if not math.isnan(auprc) else 'N/A'}")
        else:
            # Multi-class OVR macro AUCs across present classes in this subset
            classes = sorted({c for row in scores_list for c in row.keys()})
            if not classes:
                print(f"\n=== {t} ===\n[WARN] No scores available.")
                continue
            auroc, auprc = ovr_auc(scores_list, labels, classes)
            print(f"\n=== {t} ===")
            print(f"Macro AUROC (OvR): {auroc if not math.isnan(auroc) else 'N/A'}")
            print(f"Macro AUPRC (OvR): {auprc if not math.isnan(auprc) else 'N/A'}")

if __name__ == '__main__':
    main()
