"""
evaluation/main.py — format/sanity + cost-sensitive evaluation (ARCHITECTURE.md §8).

Runs against dataset/sample_messages.csv only — own IDs, disjoint from
dataset/messages.csv. This file supplies the only gold labels available
anywhere in the pipeline (messages.csv itself is unlabeled — it's what
main.py produces predictions for). Per §8/AGENTS.md, sample_messages.csv is
used here to measure format shape and cost-sensitive accuracy; the rule
thresholds in rules.py were derived independently from business_accounts.csv
distributions and were never tuned against these labels.

Reports: format/sanity issues, per-field accuracy + F1 (action,
message_type), the cost-sensitive confusion matrix, evidence-ID precision
(both the §8-defined relevance check and a cross-check against this file's
own gold evidence_message_ids), a determinism check (same input run twice,
diffed), and a rules-only / LLM-only / hybrid ablation.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_builder import Dataset, build_context  # noqa: E402
from rules import apply_rules  # noqa: E402
from router import route  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

VALID_ACTIONS = ["notify", "digest", "mute"]

# §8 cost-sensitive confusion matrix. correct=0, mild=1, severe=3 — a severe
# miss (missed-important notify->mute, or a false interruption mute->notify)
# is weighted 3x a mild one. These weights are a stated design choice, not
# fit to data (30 labeled rows isn't enough to fit a cost matrix) — same
# "disclosed, not dressed up as derived" stance as rules.py's own thresholds.
COST = {
    ("notify", "notify"): 0, ("notify", "digest"): 1, ("notify", "mute"): 3,
    ("digest", "notify"): 1, ("digest", "digest"): 0, ("digest", "mute"): 1,
    ("mute", "notify"): 3,   ("mute", "digest"): 1,   ("mute", "mute"): 0,
}


def load_sample_messages() -> list[dict]:
    with open(DATASET_DIR / "sample_messages.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_message_history_index() -> dict[str, dict]:
    with open(DATASET_DIR / "message_history.csv", newline="", encoding="utf-8") as f:
        return {row["message_id"]: row for row in csv.DictReader(f)}


def predict_hybrid(row: dict, ds: Dataset) -> tuple[dict, str]:
    context = build_context(row, ds)
    rule_result = apply_rules(context)
    if rule_result.get("resolved"):
        return rule_result, "rule"
    return route(context, rule_result), "router"


def predict_rules_only(row: dict, ds: Dataset) -> dict | None:
    context = build_context(row, ds)
    rule_result = apply_rules(context)
    return rule_result if rule_result.get("resolved") else None


def predict_llm_only(row: dict, ds: Dataset) -> dict:
    context = build_context(row, ds)
    return route(context, {"resolved": False, "rule": None, "notes": []})


def _prf1(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float, dict]:
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(1 for t in y_true if t == label)
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    scored = [m["f1"] for label, m in metrics.items() if m["support"] > 0]
    macro_f1 = sum(scored) / len(scored) if scored else 0.0
    return accuracy, macro_f1, metrics


def evidence_relevance_precision(
    rows_and_predictions: list[tuple[dict, dict]], history_index: dict[str, dict]
) -> tuple[float | None, int, int]:
    """§8 definition: fraction of cited evidence_message_ids that are
    genuinely relevant (same sender/group/business as the message being
    evaluated) — independent of gold, an objective check against
    message_history.csv."""
    correct = 0
    total = 0
    for row, prediction in rows_and_predictions:
        ev = str(prediction.get("evidence_message_ids") or "none")
        if ev.strip().lower() == "none":
            continue
        for cid in ev.split(";"):
            cid = cid.strip()
            if not cid:
                continue
            total += 1
            hist_row = history_index.get(cid)
            if not hist_row:
                continue
            same_sender = (
                (row.get("group_id") and hist_row.get("group_id") == row.get("group_id"))
                or (row.get("business_id") and hist_row.get("business_id") == row.get("business_id"))
                or (row.get("sender_user_id") and hist_row.get("sender_user_id") == row.get("sender_user_id"))
            )
            if same_sender:
                correct += 1
    precision = correct / total if total else None
    return precision, correct, total


def evidence_gold_overlap(rows_and_predictions: list[tuple[dict, dict]]) -> dict:
    """Cross-check against this file's own gold evidence_message_ids: of the
    31 individual gold citations across 28 rows, how many does the hybrid
    system's predicted evidence_message_ids reproduce (exact ID match)?"""
    gold_total = 0
    predicted_total = 0
    matched = 0
    for row, prediction in rows_and_predictions:
        gold_ev = str(row.get("evidence_message_ids") or "none")
        pred_ev = str(prediction.get("evidence_message_ids") or "none")
        gold_ids = set() if gold_ev.strip().lower() == "none" else {
            e.strip() for e in gold_ev.split(";") if e.strip()
        }
        pred_ids = set() if pred_ev.strip().lower() == "none" else {
            e.strip() for e in pred_ev.split(";") if e.strip()
        }
        gold_total += len(gold_ids)
        predicted_total += len(pred_ids)
        matched += len(gold_ids & pred_ids)
    recall = matched / gold_total if gold_total else None
    return {"gold_total": gold_total, "predicted_total": predicted_total, "matched": matched, "recall": recall}


def format_sanity_check(predictions: list[dict]) -> list[str]:
    issues = []
    for p in predictions:
        if p["action"] not in VALID_ACTIONS:
            issues.append(f"{p['message_id']}: action {p['action']!r} not a valid enum")
        try:
            conf = float(p["confidence"])
            if not (0.0 <= conf <= 1.0):
                issues.append(f"{p['message_id']}: confidence {conf} out of [0,1]")
        except (TypeError, ValueError):
            issues.append(f"{p['message_id']}: confidence {p['confidence']!r} not numeric")
    return issues


def run() -> dict:
    ds = Dataset()
    samples = load_sample_messages()
    history_index = load_message_history_index()
    rows_by_id = {row["message_id"]: row for row in samples}

    # --- hybrid (the real system) ---
    hybrid_predictions = []
    hybrid_sources = Counter()
    for row in samples:
        decision, source = predict_hybrid(row, ds)
        hybrid_predictions.append({"message_id": row["message_id"], **decision})
        hybrid_sources[source] += 1

    sanity_issues = format_sanity_check(hybrid_predictions)

    gold_action = [row["action"] for row in samples]
    pred_action = [p["action"] for p in hybrid_predictions]
    gold_type = [row["message_type"] for row in samples]
    pred_type = [p["message_type"] for p in hybrid_predictions]

    action_acc, action_macro_f1, action_metrics = _prf1(gold_action, pred_action, VALID_ACTIONS)
    type_labels = sorted(set(gold_type) | set(pred_type))
    type_acc, type_macro_f1, type_metrics = _prf1(gold_type, pred_type, type_labels)

    total_cost = sum(COST.get((t, p), 0) for t, p in zip(gold_action, pred_action))
    avg_cost = total_cost / len(samples)
    confusion = Counter(zip(gold_action, pred_action))

    pairs = [(rows_by_id[p["message_id"]], p) for p in hybrid_predictions]
    ev_precision, ev_correct, ev_total = evidence_relevance_precision(pairs, history_index)
    gold_overlap = evidence_gold_overlap(pairs)

    # --- determinism check: rerun hybrid, diff ---
    hybrid_predictions_2 = []
    for row in samples:
        decision, source = predict_hybrid(row, ds)
        hybrid_predictions_2.append({"message_id": row["message_id"], **decision})
    deterministic = hybrid_predictions == hybrid_predictions_2
    diffs = [] if deterministic else [
        (a, b) for a, b in zip(hybrid_predictions, hybrid_predictions_2) if a != b
    ]

    # --- ablation: rules-only ---
    rules_only_results = [(row["message_id"], predict_rules_only(row, ds)) for row in samples]
    rules_only_resolved = [(mid, r) for mid, r in rules_only_results if r is not None]
    rules_only_coverage = len(rules_only_resolved) / len(samples)
    if rules_only_resolved:
        r_gold = [rows_by_id[mid]["action"] for mid, _ in rules_only_resolved]
        r_pred = [r["action"] for _, r in rules_only_resolved]
        rules_only_acc = sum(1 for t, p in zip(r_gold, r_pred) if t == p) / len(rules_only_resolved)
    else:
        rules_only_acc = None

    # --- ablation: LLM-only (bypasses rules.py entirely) ---
    llm_only_predictions = [predict_llm_only(row, ds) for row in samples]
    llm_only_pred_action = [p["action"] for p in llm_only_predictions]
    llm_only_acc = sum(1 for t, p in zip(gold_action, llm_only_pred_action) if t == p) / len(samples)

    return {
        "n": len(samples),
        "sanity_issues": sanity_issues,
        "hybrid_sources": dict(hybrid_sources),
        "gold_action_dist": dict(Counter(gold_action)),
        "pred_action_dist": dict(Counter(pred_action)),
        "gold_type_dist": dict(Counter(gold_type)),
        "pred_type_dist": dict(Counter(pred_type)),
        "action_accuracy": action_acc,
        "action_macro_f1": action_macro_f1,
        "action_metrics": action_metrics,
        "type_accuracy": type_acc,
        "type_macro_f1": type_macro_f1,
        "type_metrics": type_metrics,
        "confusion": confusion,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "evidence_precision": ev_precision,
        "evidence_correct": ev_correct,
        "evidence_total": ev_total,
        "evidence_gold_overlap": gold_overlap,
        "deterministic": deterministic,
        "determinism_diffs": diffs,
        "rules_only_coverage": rules_only_coverage,
        "rules_only_accuracy": rules_only_acc,
        "llm_only_accuracy": llm_only_acc,
        "hybrid_accuracy": action_acc,
    }


def print_report(r: dict) -> None:
    print(f"=== Format/sanity pass ({r['n']} sample_messages.csv rows) ===")
    if r["sanity_issues"]:
        for issue in r["sanity_issues"]:
            print(f"  ISSUE: {issue}")
    else:
        print("  no format/range violations")
    print(f"  decision source: {r['hybrid_sources']}")
    print(f"  gold action distribution:      {r['gold_action_dist']}")
    print(f"  predicted action distribution: {r['pred_action_dist']}")
    print(f"  gold message_type distribution:      {r['gold_type_dist']}")
    print(f"  predicted message_type distribution: {r['pred_type_dist']}")

    print()
    print("=== Per-field accuracy / F1 (hybrid system) ===")
    print(f"  action:       accuracy={r['action_accuracy']:.3f}  macro_f1={r['action_macro_f1']:.3f}")
    for label, m in r["action_metrics"].items():
        print(f"    {label:10s} precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f} support={m['support']}")
    print(f"  message_type: accuracy={r['type_accuracy']:.3f}  macro_f1={r['type_macro_f1']:.3f}")
    for label, m in r["type_metrics"].items():
        if m["support"] > 0:
            print(f"    {label:16s} precision={m['precision']:.2f} recall={m['recall']:.2f} f1={m['f1']:.2f} support={m['support']}")

    print()
    print("=== Cost-sensitive confusion (action) ===")
    print("  true \\ pred    notify  digest  mute")
    for t in VALID_ACTIONS:
        row = "  ".join(f"{r['confusion'].get((t, p), 0):6d}" for p in VALID_ACTIONS)
        print(f"  {t:10s} {row}")
    print(f"  total cost={r['total_cost']}  avg cost/message={r['avg_cost']:.3f}  (weights: correct=0, mild=1, severe=3)")

    print()
    print("=== Evidence-ID precision ===")
    if r["evidence_total"]:
        print(f"  §8 relevance precision: {r['evidence_correct']}/{r['evidence_total']} = {r['evidence_precision']:.3f}")
    else:
        print("  §8 relevance precision: n/a (no non-none evidence cited by hybrid predictions)")
    go = r["evidence_gold_overlap"]
    print(f"  gold overlap: hybrid cited {go['predicted_total']} IDs total, matched {go['matched']}/{go['gold_total']} gold citations"
          f" (recall={go['recall']:.3f})" if go["recall"] is not None else "  gold overlap: n/a")

    print()
    print("=== Determinism check (hybrid run twice) ===")
    print(f"  identical: {r['deterministic']}")
    if not r["deterministic"]:
        for a, b in r["determinism_diffs"]:
            print(f"    DIFF {a['message_id']}: {a} != {b}")

    print()
    print("=== Ablation: rules-only vs LLM-only vs hybrid ===")
    print(f"  rules-only:  coverage={r['rules_only_coverage']:.1%} of messages resolved by rules alone; "
          f"accuracy on that resolved subset={r['rules_only_accuracy']:.3f}" if r["rules_only_accuracy"] is not None
          else "  rules-only: no messages resolved")
    print(f"  LLM-only:    coverage=100%; accuracy={r['llm_only_accuracy']:.3f}")
    print(f"  hybrid:      coverage=100%; accuracy={r['hybrid_accuracy']:.3f}")


if __name__ == "__main__":
    report = run()
    print_report(report)
