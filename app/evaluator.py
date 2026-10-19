"""Deterministic exact-span evaluator for development predictions."""
import csv, json
from collections import Counter
from pathlib import Path
from typing import Any
LABELS = ["supported", "contradicted", "unknown"]

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def score(prediction_rows: list[dict[str, Any]], gold_rows: list[dict[str, str]], scheduled_answer_ids: list[str]) -> dict[str, Any]:
    gold_by_key = {(g["answer_id"], int(g["start_char"]), int(g["end_char"])): g for g in gold_rows}
    pred_by_key = {}
    errors = []
    for p in prediction_rows:
        if p.get("status") == "error":
            errors.append({"answer_id": p["answer_id"], "error": p.get("error", "unknown error")})
            continue
        for c in p.get("claims", []):
            key = (p["answer_id"], int(c["start_char"]), int(c["end_char"]))
            if key in pred_by_key:
                raise ValueError(f"Duplicate prediction span: {key}")
            pred_by_key[key] = c
    G = len(gold_rows); P = len(pred_by_key)
    matched = set(pred_by_key) & set(gold_by_key)
    correct = {k for k in matched if pred_by_key[k]["label"] == gold_by_key[k]["expected_label"]}
    M, C = len(matched), len(correct)
    div = lambda a,b: a/b if b else 0.0
    labels_report = {}
    confusion = {r: {c: 0 for c in LABELS + ["missing_or_error"]} for r in LABELS + ["no_gold"]}
    for k, g in gold_by_key.items():
        if k not in pred_by_key:
            confusion[g["expected_label"]]["missing_or_error"] += 1
        else:
            confusion[g["expected_label"]][pred_by_key[k]["label"]] += 1
    for k, p in pred_by_key.items():
        if k not in gold_by_key:
            confusion["no_gold"][p["label"]] += 1
    for label in LABELS:
        tp = sum(1 for k in correct if gold_by_key[k]["expected_label"] == label)
        predicted = sum(1 for p in pred_by_key.values() if p["label"] == label)
        actual = sum(1 for g in gold_rows if g["expected_label"] == label)
        precision, recall = div(tp, predicted), div(tp, actual)
        labels_report[label] = {"precision": precision, "recall": recall, "f1": div(2*precision*recall, precision+recall)}
    missed = [g["claim_id"] for k,g in gold_by_key.items() if k not in pred_by_key]
    extra = [p["claim_id"] for k,p in pred_by_key.items() if k not in gold_by_key]
    wrong = [gold_by_key[k]["claim_id"] for k in matched if k not in correct]
    return {"scheduled_answers": len(scheduled_answer_ids), "answer_errors": len(errors), "answer_error_details": errors,
        "G_gold_claims": G, "P_accepted_predictions": P, "M_exact_span_matches": M, "C_exact_span_and_label_correct": C,
        "extraction_precision": div(M,P), "extraction_recall": div(M,G), "extraction_f1": div(2*M,P+G),
        "joint_precision": div(C,P), "joint_recall": div(C,G), "joint_f1": div(2*C,P+G), "gold_claim_coverage": div(C,G),
        "per_label": labels_report, "macro_f1": sum(v["f1"] for v in labels_report.values())/3,
        "missed_gold_claim_ids": missed, "unmatched_prediction_claim_ids": extra,
        "matched_wrong_label_gold_claim_ids": wrong, "confusion_matrix": confusion}
