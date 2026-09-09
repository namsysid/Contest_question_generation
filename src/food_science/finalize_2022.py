from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.circuit_lab.model_client import embed_texts
from src.circuit_lab.validate import answers_agree

from .common import graph_text, read_jsonl, validate_item, verification_error, write_jsonl
from .pipeline import create_item, judge_item, normalized, retrieve_exemplars, source_calibrated_audit


TARGET_SPECS: list[dict[str, Any]] = [
    {
        "response_type": "multiple_choice",
        "topic": "analytical_tests",
        "secondary_topics": ["proteins_enzymes", "carbohydrates_sweeteners", "food_chemistry_reactions"],
        "difficulty": 3,
        "target_skill": "Integrate before-and-after iodine, Benedict's, and Biuret results to identify food molecules and enzyme action.",
        "novel_context": "An unknown food slurry is tested before and after controlled amylase treatment.",
        "fixed_givens": [
            "Iodine turns blue-black with starch; Biuret turns violet with protein; heated Benedict's turns orange with reducing sugar.",
            "Amylase hydrolyzes starch to smaller reducing sugars but does not digest protein.",
            "Before amylase, the slurry is iodine blue-black, Biuret violet, and Benedict's blue.",
            "After amylase at 37 °C, it is iodine amber, Biuret violet, and Benedict's orange.",
        ],
        "verification": {
            "expected_answer": "The original contained starch and protein but no detectable reducing sugar; amylase converted starch to reducing sugar while protein remained.",
            "calculation": "Initial iodine and Biuret positives establish starch and protein. Initial Benedict's is negative. Loss of iodine plus gain of Benedict's after amylase shows starch hydrolysis; unchanged Biuret shows protein remains.",
        },
        "distractor_mechanisms": [
            "claim amylase digested protein",
            "reverse the meanings of iodine and Benedict's results",
            "ignore the unchanged Biuret result",
        ],
    },
    {
        "response_type": "multiple_choice",
        "topic": "food_safety_preservation",
        "difficulty": 3,
        "target_skill": "Apply both water-activity and pH growth limits to a hurdle-preserved food.",
        "novel_context": "A dried, acidified sauce is checked against four organisms' stated growth limits.",
        "fixed_givens": [
            "The finished sauce has water activity 0.88 and pH 4.2.",
            "Growth requires satisfying both the stated minimum water activity and stated pH range.",
            "Organism A: minimum aw 0.95, pH 4.5–9.0; B: minimum aw 0.86, pH 2.0–8.0.",
            "Organism C: minimum aw 0.90, pH 4.0–8.0; D: minimum aw 0.80, pH 5.0–9.0.",
            "Treat these values as classroom data, not commercial safety guidance.",
        ],
        "verification": {
            "expected_answer": "Only Organism B meets both conditions and could grow under the stated classroom model.",
            "calculation": "A fails aw and pH; B has 0.88≥0.86 and pH 4.2 within 2.0–8.0; C fails aw; D fails pH. Only B meets both hurdles.",
        },
        "distractor_mechanisms": [
            "check only water activity",
            "check only pH",
            "reverse the minimum-water-activity inequality",
        ],
    },
    {
        "response_type": "short_answer",
        "topic": "carbohydrates_sweeteners",
        "difficulty": 1,
        "target_skill": "Recall the two monosaccharides released by hydrolysis of lactose.",
        "novel_context": "Lactase hydrolysis of lactose is described in a single direct question.",
        "fixed_givens": [
            "Lactase hydrolyzes the glycosidic bond in lactose.",
        ],
        "verification": {
            "expected_answer": "glucose and galactose",
            "calculation": "Lactose is the disaccharide glucose-galactose, so hydrolysis releases glucose and galactose.",
        },
        "distractor_mechanisms": [
            "confuse lactose with sucrose and answer glucose plus fructose",
            "confuse lactose with maltose and answer two glucose molecules",
            "name the enzyme rather than the hydrolysis products",
        ],
    },
    {
        "response_type": "multiple_choice",
        "topic": "food_safety_preservation",
        "difficulty": 2,
        "target_skill": "Interpret microbial counts to distinguish pasteurization from sterilization and compare vegetative-cell and spore heat resistance.",
        "novel_context": "A milk heat-treatment trial reports before-and-after counts for vegetative cells and bacterial spores.",
        "fixed_givens": [
            "Before heating, milk contains 1,000,000 vegetative cells/mL and 10,000 bacterial spores/mL.",
            "After heating, 100 vegetative cells/mL and 10,000 spores/mL remain.",
            "Sterilization would leave no viable cells or spores.",
        ],
        "verification": {
            "expected_answer": "The treatment greatly reduced vegetative cells but did not sterilize the milk because the spores survived.",
            "calculation": "Vegetative cells fell by four orders of magnitude, while the spore count did not change. Surviving spores mean the treatment was not sterilization.",
        },
        "distractor_mechanisms": [
            "call any large vegetative-cell reduction sterilization",
            "claim spores were less heat resistant despite unchanged counts",
            "compare only the final absolute counts and ignore before-to-after survival",
        ],
    },
]

BASE_SELECTION = [
    "food-science-b-generated-104",
    "food-science-b-generated-005",
    "food-science-b-generated-006",
]

REQUIRED_JUDGE_KEYS = (
    "solvable", "science_correct", "division_b_appropriate", "food_science_relevant",
    "difficulty_match", "reasoning_depth_match", "source_level_calibrated",
    "source_scope_faithful", "single_unambiguous_answer",
    "competition_faithful", "style_faithful",
    "concise", "distractors_plausible", "safety_appropriate", "plan_faithful", "novel",
    "answer_agrees",
)


def plan_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {"id": f"g{index}", "type": "Given", "label": text}
        for index, text in enumerate(spec["fixed_givens"], 1)
    ]
    nodes.extend([
        {"id": "law", "type": "Law", "label": spec["target_skill"]},
        {"id": "target", "type": "Target", "label": spec["verification"]["expected_answer"]},
    ])
    edges = [
        {"src": node["id"], "dst": "law", "type": "supports"} for node in nodes[:-2]
    ] + [{"src": "law", "dst": "target", "type": "derived_from"}]
    plan = {**spec, "reasoning_graph": {"nodes": nodes, "edges": edges},
            "audit": {"valid": True, "method": "human-vetted solution-first specification"}}
    error = verification_error(plan)
    if error:
        raise RuntimeError(error)
    return plan


def choose_anchor(corpus: list[dict[str, Any]], topic: str, offset: int) -> dict[str, Any]:
    matches = [row for row in corpus if topic in row.get("topics", [])]
    return (matches or corpus)[offset % len(matches or corpus)]


def deterministic_audit(
    items: list[dict[str, Any]], corpus: list[dict[str, Any]],
    item_vectors: list[list[float]], source_vectors: list[list[float]],
    plan_vectors: list[list[float]], source_plan_vectors: list[list[float]],
) -> dict[str, Any]:
    imat, smat = normalized(item_vectors), normalized(source_vectors)
    pmat, spmat = normalized(plan_vectors), normalized(source_plan_vectors)
    pair_max = 0.0
    pair_ids: list[str] = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            score = float(imat[left] @ imat[right])
            if score > pair_max:
                pair_max, pair_ids = score, [items[left]["id"], items[right]["id"]]
    source_q_max = float((imat @ smat.T).max())
    source_s_max = float((pmat @ spmat.T).max())
    difficulties = {level: sum(item["difficulty"] == level for item in items) for level in range(1, 6)}
    word_limits = {1: 30, 2: 90, 3: 140}
    deterministic_errors = {item["id"]: validate_item(item) for item in items if validate_item(item)}
    length_errors = {item["id"]: len(re.findall(r"\w+", item["prompt"])) for item in items
                     if len(re.findall(r"\w+", item["prompt"])) > word_limits[item["difficulty"]]}
    checks = {
        "five_items": len(items) == 5,
        "difficulty_1_3_1": difficulties[1] == 1 and difficulties[2] == 3 and difficulties[3] == 1,
        "mcq_and_constructed_response": {item["response_type"] for item in items} >= {"multiple_choice", "short_answer"},
        "five_distinct_primary_topics": len({item["topics"][0] for item in items}) == 5,
        "schema_scope_and_word_caps": not deterministic_errors and not length_errors,
        "question_pair_similarity_below_0_875": pair_max < 0.875,
        "source_question_similarity_below_0_94": source_q_max < 0.94,
        "source_structure_similarity_below_0_92": source_s_max < 0.92,
    }
    return {
        "overall_good": all(checks.values()), "checks": checks, "difficulty_counts": difficulties,
        "maximum_pairwise_question_similarity": {"score": pair_max, "ids": pair_ids},
        "maximum_source_question_similarity": source_q_max,
        "maximum_source_structure_similarity": source_s_max,
        "deterministic_errors": deterministic_errors, "length_errors": length_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a strict five-item Food Science 2022 pilot")
    parser.add_argument("--run-dir", default="science_olympiad/food_science_b/season_2022_pilot")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)

    corpus = read_jsonl(root / "corpus" / "items.jsonl")
    q_rows = read_jsonl(root / "enriched" / "question_embeddings.jsonl")
    s_rows = read_jsonl(root / "enriched" / "structure_embeddings.jsonl")
    source_qmat = normalized([row["embedding"] for row in q_rows])
    source_smat = normalized([row["embedding"] for row in s_rows])
    pool = read_jsonl(root / "generated" / "items.jsonl")
    pool_reports = read_jsonl(root / "validation" / "item_reports.jsonl")
    prior_vectors = embed_texts(args.embedding_model, [item["prompt"] for item in pool], provider=args.provider)

    targeted_path = root / "generated" / "targeted_items.jsonl"
    targeted_reports_path = root / "validation" / "targeted_reports.jsonl"
    targeted = read_jsonl(targeted_path) if targeted_path.exists() else []
    targeted_reports = read_jsonl(targeted_reports_path) if targeted_reports_path.exists() else []
    for index, spec in enumerate(TARGET_SPECS):
        if index < len(targeted):
            continue
        plan = plan_from_spec(spec)
        anchor = choose_anchor(corpus, spec["topic"], index)
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        item, report, vector = create_item(
            anchor, exemplars, plan, pool + targeted, source_qmat, source_smat, corpus,
            prior_vectors, args.embedding_model, args.model, args.provider, 120000 + index * 1000,
            101 + index,
        )
        targeted.append(item)
        targeted_reports.append(report)
        prior_vectors.append(vector)
        write_jsonl(targeted_path, targeted)
        write_jsonl(targeted_reports_path, targeted_reports)
        print(f"Accepted targeted {index + 1}/{len(TARGET_SPECS)}: {spec['topic']}", flush=True)

    all_candidates = pool + targeted
    for index, item in enumerate(targeted):
        secondary = TARGET_SPECS[index].get("secondary_topics") or []
        item["generation"]["plan"]["secondary_topics"] = secondary
        item["topics"] = list(dict.fromkeys([item["generation"]["plan"]["topic"], *secondary]))
    write_jsonl(root / "generated" / "candidate_pool.jsonl", all_candidates)
    by_id = {item["id"]: item for item in all_candidates}
    # The analytical integration item is the primary d3 choice. The hurdle item
    # remains in the pool as an alternate and is never forced into the bank.
    selected = [by_id[item_id] for item_id in
                ["food-science-b-generated-103", *BASE_SELECTION, "food-science-b-generated-101"]]

    prior_final_reports_path = root / "validation" / "item_reports_final.jsonl"
    prior_final_reports = read_jsonl(prior_final_reports_path) if prior_final_reports_path.exists() else []
    prior_report_by_id = {report["id"]: report for report in prior_final_reports}
    fresh_reports = []
    source_by_id = {item["id"]: item for item in corpus}
    for index, item in enumerate(selected):
        if args.audit_only and item["id"] in prior_report_by_id:
            fresh_reports.append(prior_report_by_id[item["id"]])
            continue
        errors = validate_item(item)
        generation = item.get("generation") or {}
        judge = None
        if not errors:
            judge = judge_item(item, generation.get("plan") or {},
                               source_by_id.get(generation.get("anchor_id")) or {},
                               args.model, args.provider, 130000 + index)
            judge["answer_agrees"] = answers_agree(item, judge.get("independent_answer"))
        valid = not errors and judge is not None and all(judge.get(key) is True for key in REQUIRED_JUDGE_KEYS)
        fresh_reports.append({"id": item["id"], "valid": valid, "deterministic_errors": errors, "judge": judge})
        print(f"Fresh validation {index + 1}/5: {item['id']} valid={valid}", flush=True)
    if not all(report["valid"] for report in fresh_reports):
        raise RuntimeError("A selected item failed fresh validation")

    item_vectors = embed_texts(args.embedding_model, [item["prompt"] for item in selected], provider=args.provider)
    plan_texts = [graph_text({"response_type": item["response_type"], "topics": item["topics"],
                              "analysis": {"reasoning_graph": item["generation"]["plan"]["reasoning_graph"]}})
                  for item in selected]
    plan_vectors = embed_texts(args.embedding_model, plan_texts, provider=args.provider)
    audit = deterministic_audit(selected, corpus, item_vectors, [row["embedding"] for row in q_rows],
                                plan_vectors, [row["embedding"] for row in s_rows])
    audit_path = root / "validation" / "deterministic_audit_final.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not audit["overall_good"]:
        raise RuntimeError("Final selection failed deterministic audit")

    batch = source_calibrated_audit(selected, corpus, args.model, args.provider, 140000)
    (root / "validation" / "batch_audit_final.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(root / "generated" / "items_final.jsonl", selected)
    write_jsonl(root / "validation" / "item_reports_final.jsonl", fresh_reports)
    print(f"Final five: deterministic={audit['overall_good']} holistic={batch.get('overall_good')}")


if __name__ == "__main__":
    main()
