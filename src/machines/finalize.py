from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.circuit_lab.model_client import generate_json
from src.circuit_lab.validate import answers_agree

from .common import read_jsonl, validate_item, write_jsonl
from .pipeline import BATCH_AUDIT_SYSTEM, judge_item


SELECTED_IDS = [
    "machines-b-generated-001", "machines-b-generated-022", "machines-b-generated-127",
    "machines-b-generated-128", "machines-b-generated-135", "machines-b-generated-161",
    "machines-b-generated-156", "machines-b-generated-157", "machines-b-generated-158",
    "machines-b-generated-159",
    "machines-b-generated-010", "machines-b-generated-118", "machines-b-generated-123",
    "machines-b-generated-129", "machines-b-generated-140", "machines-b-generated-160",
    "machines-b-generated-107", "machines-b-generated-114", "machines-b-generated-115",
    "machines-b-generated-155",
]
CORRECTED_DIFFICULTIES: dict[str, int] = {
    "machines-b-generated-022": 1,
    "machines-b-generated-118": 2,
    "machines-b-generated-138": 2,
    "machines-b-generated-140": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize and revalidate the curated Machines bank")
    parser.add_argument("--run-dir", default="science_olympiad/machines_b/final")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--audit-only", action="store_true",
                        help="Reuse the saved per-item reports and rerun only the whole-bank audit")
    args = parser.parse_args()
    root = Path(args.run_dir)
    # The curated file intentionally omits some sound introductory items while adding
    # replacement IDs. Merge both inventories so the final bank can restore a
    # previously validated basic item without losing the targeted replacements.
    all_items = {row["id"]: row for row in read_jsonl(root / "generated" / "items.jsonl")}
    all_items.update({row["id"]: row for row in read_jsonl(root / "generated" / "items_curated.jsonl")})
    polished_path = root / "generated" / "items_polished.jsonl"
    if polished_path.exists():
        all_items.update({row["id"]: row for row in read_jsonl(polished_path)})
    corpus = {row["id"]: row for row in read_jsonl(root / "corpus" / "items.jsonl")}
    items = [all_items[item_id] for item_id in SELECTED_IDS]
    saved_report_path = root / "validation" / "item_reports_final.jsonl"
    saved_reports = ({row["id"]: row for row in read_jsonl(saved_report_path)}
                     if args.audit_only and saved_report_path.exists() else {})
    reports = []
    for index, item in enumerate(items):
        if item["id"] in CORRECTED_DIFFICULTIES:
            item["difficulty"] = CORRECTED_DIFFICULTIES[item["id"]]
            (item.get("generation") or {}).get("plan", {})["difficulty"] = item["difficulty"]
        if args.audit_only:
            report = saved_reports.get(item["id"])
            if not report or report.get("valid") is not True:
                raise RuntimeError(f"No valid saved item report for {item['id']}; omit --audit-only")
            reports.append(report)
            continue
        errors = validate_item(item)
        generation = item.get("generation") or {}
        plan = generation.get("plan") or {}
        anchor = corpus.get(generation.get("anchor_id")) or {}
        judge = None
        if not errors:
            judge = judge_item(item, plan, anchor, args.model, args.provider, 31000 + index)
            judge["answer_agrees"] = answers_agree(item, judge.get("independent_answer"))
        required = ("solvable", "science_correct", "division_b_appropriate", "machines_relevant",
                    "difficulty_match", "competition_faithful", "style_faithful", "novel", "answer_agrees")
        valid = not errors and judge is not None and all(judge.get(key) is True for key in required)
        reports.append({"id": item["id"], "valid": valid, "deterministic_errors": errors, "judge": judge})
        print(f"Revalidated {index + 1}/{len(items)}: {item['id']} valid={valid}", flush=True)
    final_items = root / "generated" / "items_final.jsonl"
    final_reports = saved_report_path
    write_jsonl(final_items, items)
    write_jsonl(final_reports, reports)

    all_calibration_rows = list(corpus.values())
    source_distribution = {
        difficulty: sum(
            int(row.get("difficulty") or 2) == difficulty for row in all_calibration_rows
        )
        for difficulty in range(1, 6)
    }
    calibration_rows = (
        [row for row in all_calibration_rows if int(row.get("difficulty") or 2) == 1][::8][:6]
        + [row for row in all_calibration_rows if int(row.get("difficulty") or 2) == 2][::5][:6]
        + [row for row in all_calibration_rows if int(row.get("difficulty") or 2) >= 3]
    )
    calibration = [{"response_type": row.get("response_type"), "difficulty": row.get("difficulty"),
                    "prompt": row.get("prompt"), "choices": row.get("choices"),
                    "skills": (row.get("analysis") or {}).get("skills"),
                    "reasoning_graph": (row.get("analysis") or {}).get("reasoning_graph")}
                   for row in calibration_rows]
    payload = []
    for row in items:
        plan = ((row.get("generation") or {}).get("plan") or {})
        payload.append({"id": row["id"], "response_type": row["response_type"],
                        "difficulty": row["difficulty"], "topics": row["topics"],
                        "target_skill": plan.get("target_skill"),
                        "reasoning_graph": plan.get("reasoning_graph"),
                        "prompt": row["prompt"], "choices": row.get("choices")})
    generated_distribution = {
        difficulty: sum(row["difficulty"] == difficulty for row in items)
        for difficulty in range(1, 6)
    }
    prompt = ("REAL 2026 SOURCE DIFFICULTY COUNTS:\n" +
              json.dumps(source_distribution, ensure_ascii=False) +
              "\nGENERATED BANK DIFFICULTY COUNTS:\n" +
              json.dumps(generated_distribution, ensure_ascii=False) +
              "\nREAL 2026 WRITTEN-TEST CALIBRATION EXAMPLES, STRATIFIED THROUGH THE HARD SECTION:\n" +
              json.dumps(calibration, ensure_ascii=False) +
              f"\nThe generated bank is one representative {len(items)}-item set, not a concatenation of all three source tests. "
              "Judge whether its breadth is proportionate to its size; do not require every rare source-test niche.\n"
              "FINAL GENERATED QUESTION SET:\n" + json.dumps(payload, ensure_ascii=False))
    audit = generate_json(args.model, prompt, provider=args.provider, system=BATCH_AUDIT_SYSTEM,
                          temperature=0, max_output_tokens=5000, seed=39999)
    (root / "validation" / "batch_audit_final.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Final: {sum(report['valid'] for report in reports)}/{len(items)} individually valid; "
          f"batch audit overall_good={audit.get('overall_good')}")


if __name__ == "__main__":
    main()
