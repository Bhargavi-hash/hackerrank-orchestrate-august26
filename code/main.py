"""
main.py — entry point: dataset/ -> output.csv (ARCHITECTURE.md §9).

For every row in dataset/messages.csv: build its MessageContext
(context_builder.py), try the deterministic rule layer (rules.py), and if it
doesn't resolve, fall through to the LLM router (router.py). Validates every
successfully-produced row (validate.py) and writes dataset/output.csv with
the exact column order from §0:
message_id,action,message_type,reason,confidence,evidence_message_ids

Resilience properties (real, not assumed):
- Incremental writing: each row is written and flushed to disk as soon as
  it's decided, not buffered until the whole run finishes.
- Resumability: router.py and media/*.py both cache by content hash/media_id
  on disk. Re-running main.py after a mid-run failure re-derives the same
  MessageContext for already-completed messages and hits those caches
  instead of re-calling any API — this file doesn't need its own resume
  logic on top of that, just needs to not crash the whole run on one bad row
  (see fault isolation below) and to actually persist progress incrementally
  so a kill -9 mid-run doesn't lose completed rows.
- Per-row fault isolation: a message whose router/media call fails after
  exhausting retries is logged and skipped, not allowed to abort the script.
  Failed message_ids go to a separate dataset/output_failures.csv (message_id,
  error) rather than into output.csv itself — AGENTS.md's dataset contract
  requires action to be exactly one of notify/digest/mute, so a sentinel
  value like "FAILED" would corrupt the output schema; keeping failures in a
  companion file keeps output.csv strictly valid while still making failures
  visible and reproducible (rerun main.py — cache carries forward, failures
  are the only messages that make fresh calls).

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
FAILURE_COLUMNS = ["message_id", "error"]


def build_output_row(message: dict, ds: Dataset) -> tuple[dict, dict, str]:
    """Returns (output_row, context, source) where source is 'rule' or 'router'.
    Raises on failure — the caller (run()) is responsible for catching this
    per-message so one bad row can't abort the whole loop."""
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

    output_path = dataset_dir / "output.csv"
    failures_path = dataset_dir / "output_failures.csv"

    rows: list[dict] = []
    contexts_by_id: dict[str, dict] = {}
    rule_count = 0
    router_count = 0
    failed: list[tuple[str, str]] = []

    with open(output_path, "w", newline="", encoding="utf-8") as out_f, \
         open(failures_path, "w", newline="", encoding="utf-8") as fail_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        out_f.flush()
        fail_writer = csv.DictWriter(fail_f, fieldnames=FAILURE_COLUMNS)
        fail_writer.writeheader()
        fail_f.flush()

        for message in ds.messages:
            message_id = message["message_id"]
            try:
                row, context, source = build_output_row(message, ds)
            except Exception as exc:
                logger.error("message_id=%s FAILED, skipping (will retry on next run): %s", message_id, exc)
                failed.append((message_id, str(exc)))
                fail_writer.writerow({"message_id": message_id, "error": str(exc)})
                fail_f.flush()
                continue

            rows.append(row)
            contexts_by_id[message_id] = context
            if source == "rule":
                rule_count += 1
            else:
                router_count += 1

            writer.writerow(row)
            out_f.flush()

    flags = validate_output(rows, contexts_by_id, message_history_ids) if rows else []

    return {
        "total_messages": len(ds.messages),
        "rows_written": len(rows),
        "rule_count": rule_count,
        "router_count": router_count,
        "failed": failed,
        "flags": flags,
        "output_path": output_path,
        "failures_path": failures_path,
    }


if __name__ == "__main__":
    t0 = time.time()
    summary = run()
    elapsed = time.time() - t0

    print(f"Processed {summary['total_messages']} messages in {elapsed:.2f}s")
    print(f"  rows written:        {summary['rows_written']}")
    print(f"  resolved via rules:  {summary['rule_count']}")
    print(f"  resolved via router: {summary['router_count']}")
    print(f"  failed:              {len(summary['failed'])}")
    for message_id, error in summary["failed"]:
        print(f"    - {message_id}: {error[:150]}")
    print(f"  validate.py flags:   {len(summary['flags'])}")
    for flag in summary["flags"]:
        print(f"    - {flag}")
    print(f"Wrote {summary['output_path']}")
    if summary["failed"]:
        print(f"Wrote {summary['failures_path']} ({len(summary['failed'])} failed rows — rerun main.py to retry them)")
