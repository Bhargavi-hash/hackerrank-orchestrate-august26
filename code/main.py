"""
main.py — entry point: dataset/ -> output.csv (ARCHITECTURE.md §9).

For every row in dataset/messages.csv: build its MessageContext
(context_builder.py), try the deterministic rule layer (rules.py), and if it
doesn't resolve, fall through to the LLM router (router.py). Validates every
row (validate.py) before writing dataset/output.csv with the exact column
order from §0: message_id,action,message_type,reason,confidence,evidence_message_ids

Usage: python3 main.py
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from context_builder import Dataset, build_context
from rules import apply_rules
from router import route
from validate import load_message_history_ids, validate_output

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("main")

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def build_output_row(message: dict, ds: Dataset) -> tuple[dict, dict, str]:
    """Returns (output_row, context, source) where source is 'rule' or 'router'."""
    context = build_context(message, ds)
    rule_result = apply_rules(context)

    if rule_result.get("resolved"):
        source = "rule"
        decision = rule_result
    else:
        source = "router"
        decision = route(context, rule_result)

    row = {
        "message_id": message["message_id"],
        "action": decision["action"],
        "message_type": decision["message_type"],
        "reason": decision["reason"],
        "confidence": decision["confidence"],
        "evidence_message_ids": decision["evidence_message_ids"],
    }
    return row, context, source


def run(dataset_dir: Path = DATASET_DIR) -> dict:
    ds = Dataset(dataset_dir)
    message_history_ids = load_message_history_ids(dataset_dir)

    rows = []
    contexts_by_id = {}
    rule_count = 0
    router_count = 0

    for message in ds.messages:
        row, context, source = build_output_row(message, ds)
        rows.append(row)
        contexts_by_id[row["message_id"]] = context
        if source == "rule":
            rule_count += 1
        else:
            router_count += 1

    flags = validate_output(rows, contexts_by_id, message_history_ids)

    output_path = dataset_dir / "output.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "total": len(rows),
        "rule_count": rule_count,
        "router_count": router_count,
        "flags": flags,
        "output_path": output_path,
    }


if __name__ == "__main__":
    t0 = time.time()
    summary = run()
    elapsed = time.time() - t0

    print(f"Processed {summary['total']} messages in {elapsed:.2f}s")
    print(f"  resolved via rules:  {summary['rule_count']}")
    print(f"  resolved via router: {summary['router_count']}")
    print(f"  validate.py flags:   {len(summary['flags'])}")
    for flag in summary["flags"]:
        print(f"    - {flag}")
    print(f"Wrote {summary['output_path']}")
