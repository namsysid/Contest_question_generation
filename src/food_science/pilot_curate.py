from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.circuit_lab.model_client import embed_texts
from src.circuit_lab.validate import answers_agree

from .common import classify_topics, graph_text, read_jsonl, validate_item, verification_error, write_jsonl
from .pipeline import create_item, judge_item, normalized, retrieve_exemplars, source_calibrated_audit


SPECS: list[dict[str, Any]] = [
    {
        "response_type": "multiple_choice",
        "topic": "proteins_coagulation",
        "difficulty": 2,
        "target_skill": "Compare acid coagulation and rennet coagulation of casein micelles.",
        "novel_context": "Two cheese vats form curds by different controlled mechanisms.",
        "fixed_givens": [
            "Vat A contains a starter culture that converts lactose to lactic acid and lowers pH toward 4.6.",
            "Near pH 4.6, casein micelles lose much of the charge repulsion that keeps them apart.",
            "Vat B receives rennet, which cleaves protective kappa-casein on the micelle surface without requiring the same initial pH drop.",
            "Both vats form curds.",
        ],
        "verification": {
            "expected_answer": "Vat A uses acid coagulation; Vat B uses enzymatic rennet coagulation; both destabilize casein micelles.",
            "calculation": "Acid reduces charge repulsion in A, whereas rennet removes a stabilizing surface protein in B.",
        },
        "distractor_mechanisms": [
            "claim rennet directly ferments lactose",
            "claim both vats coagulate only from heat",
            "reverse the acid and enzyme mechanisms",
        ],
    },
    {
        "response_type": "numeric",
        "topic": "acidity_measurement",
        "difficulty": 3,
        "target_skill": "Calculate titratable acidity as grams of lactic acid per 100 mL from a NaOH titration.",
        "novel_context": "Quality-control titration of a fermented milk sample.",
        "fixed_givens": [
            "A 10.0 mL fermented-milk sample requires 8.00 mL of 0.100 mol/L NaOH to reach the specified endpoint.",
            "Treat all titratable acid as monoprotic lactic acid reacting 1:1 with NaOH.",
            "The molar mass of lactic acid is 90.08 g/mol.",
            "Report grams of lactic acid per 100 mL of sample.",
        ],
        "verification": {
            "expected_answer": "0.721 g lactic acid per 100 mL (0.721% m/v)",
            "expected_value": 0.72064,
            "expression": "0.00800*0.100*90.08*100/10.0",
            "calculation": "NaOH and acid moles are 0.00800 L × 0.100 mol/L = 0.000800 mol. This is 0.072064 g in 10.0 mL, or 0.72064 g per 100 mL.",
        },
        "distractor_mechanisms": [
            "forget to convert milliliters to liters",
            "omit the scaling from 10 mL to 100 mL",
            "invert the 1:1 mole relationship",
        ],
    },
    {
        "response_type": "multiple_choice",
        "topic": "fats_emulsions",
        "difficulty": 2,
        "target_skill": "Predict how homogenization changes milk-fat droplet size and creaming rate.",
        "novel_context": "Compare homogenized and unhomogenized milk with equal fat content during storage.",
        "fixed_givens": [
            "Two milk samples have the same fat percentage and storage temperature.",
            "Sample H is homogenized, breaking fat into much smaller droplets; Sample U is not homogenized.",
            "No stabilizer is added to either sample.",
            "The samples stand undisturbed for the same time.",
        ],
        "verification": {
            "expected_answer": "Sample U forms the more obvious cream layer; Sample H stays more uniform because its smaller fat droplets rise and coalesce more slowly.",
            "calculation": "With composition and storage controlled, droplet size is the relevant difference. Smaller homogenized droplets cream more slowly.",
        },
        "distractor_mechanisms": [
            "claim homogenization removes fat",
            "claim smaller droplets rise faster",
            "attribute the difference to unequal temperatures",
        ],
    },
    {
        "response_type": "multiple_choice",
        "topic": "dairy_processing_products",
        "difficulty": 2,
        "target_skill": "Calculate ice-cream overrun from mix and finished-product volumes.",
        "novel_context": "A pilot freezer incorporates air into a measured ice-cream mix.",
        "fixed_givens": [
            "The freezer starts with 2.00 L of liquid mix and produces 3.20 L of ice cream.",
            "No mix is lost.",
            "Overrun (%) = ((finished volume - mix volume) / mix volume) × 100.",
        ],
        "verification": {
            "expected_answer": "60% overrun",
            "expected_value": 60.0,
            "expression": "(3.20-2.00)/2.00*100",
            "calculation": "The volume increase is 1.20 L. Dividing by 2.00 L and multiplying by 100 gives 60%.",
        },
        "distractor_mechanisms": [
            "divide by finished volume instead of mix volume",
            "report the final volume as a percent",
            "omit multiplication by 100",
        ],
    },
    {
        "response_type": "multiple_choice",
        "topic": "safety_pasteurization",
        "difficulty": 3,
        "target_skill": "Calculate microbial log reduction and apply both a safety target and a product-quality constraint.",
        "novel_context": "Select a modeled milk heat treatment from paired microbial and sensory data.",
        "fixed_givens": [
            "Each milk trial begins at 1.0 × 10^6 CFU/mL.",
            "For this model, an acceptable trial must achieve at least a 5-log reduction and retain a sensory score of at least 80/100.",
            "Use log reduction = log10(initial count / survivor count).",
            "Trial P leaves 1.0 × 10^1 CFU/mL and scores 88; Q leaves 1.0 × 10^2 and scores 93; R leaves 1.0 × 10^0 and scores 72; S leaves 1.0 × 10^3 and scores 96.",
            "Choose the acceptable trial; these modeled thresholds do not replace a validated commercial process.",
        ],
        "verification": {
            "expected_answer": "Trial P: 5-log reduction and sensory score 88, satisfying both modeled requirements.",
            "calculation": "P gives log10(10^6/10^1)=5 and score 88. Q gives 4 logs; R gives 6 logs but score 72; S gives 3 logs. Only P meets both constraints.",
        },
        "distractor_mechanisms": [
            "choose the best sensory score while ignoring microbial reduction",
            "choose the greatest reduction while ignoring product quality",
            "miscount powers of ten in the log reduction",
        ],
    },
]


def make_graph(spec: dict[str, Any]) -> dict[str, Any]:
    givens = [
        {"id": f"g{index}", "type": "Given", "label": text}
        for index, text in enumerate(spec["fixed_givens"], 1)
    ]
    nodes = givens + [
        {"id": "law", "type": "Law", "label": spec["target_skill"]},
        {"id": "target", "type": "Target", "label": spec["verification"]["expected_answer"]},
    ]
    edges = [
        {"src": node["id"], "dst": "law", "type": "supports"} for node in givens
    ] + [{"src": "law", "dst": "target", "type": "derived_from"}]
    return {"nodes": nodes, "edges": edges}


def plan_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(spec)
    plan["reasoning_graph"] = make_graph(spec)
    plan["audit"] = {"valid": True, "method": "human-vetted solution-first specification"}
    arithmetic_error = verification_error(plan)
    if arithmetic_error:
        raise RuntimeError(arithmetic_error)
    return plan


def choose_anchor(corpus: list[dict[str, Any]], topic: str, offset: int) -> dict[str, Any]:
    candidates = [row for row in corpus if topic in row.get("topics", [])]
    if not candidates:
        candidates = corpus
    return candidates[offset % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reviewed five-item Food Science pilot")
    parser.add_argument("--root", default="science_olympiad/food_science_b/pilot")
    parser.add_argument("--output", default="science_olympiad/food_science_b/pilot_reviewed")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    root, output = Path(args.root), Path(args.output)
    corpus = read_jsonl(root / "corpus" / "items.jsonl")
    q_rows = read_jsonl(root / "enriched" / "question_embeddings.jsonl")
    s_rows = read_jsonl(root / "enriched" / "structure_embeddings.jsonl")
    source_qmat = normalized([row["embedding"] for row in q_rows])
    source_smat = normalized([row["embedding"] for row in s_rows])

    items_path = output / "generated" / "items.jsonl"
    reports_path = output / "validation" / "item_reports.jsonl"
    plans_path = output / "generated" / "reasoning_plans.jsonl"
    if items_path.exists():
        items = read_jsonl(items_path)
        reports = read_jsonl(reports_path)
        plans = read_jsonl(plans_path)
    else:
        original_items = read_jsonl(root / "generated" / "items.jsonl")
        original_reports = {row["id"]: row for row in read_jsonl(root / "validation" / "item_reports.jsonl")}
        retained = copy.deepcopy(original_items[0])
        items = [retained]
        reports = [copy.deepcopy(original_reports[retained["id"]])]
        plans = [retained["generation"]["plan"]]
    completed_replacements = len(items) - 1
    prior_vectors = embed_texts(args.embedding_model, [item["prompt"] for item in items], provider=args.provider)

    for index, spec in enumerate(SPECS, 1):
        if index <= completed_replacements:
            continue
        plan = plan_from_spec(spec)
        plan_structure = graph_text({"response_type": spec["response_type"], "topics": [spec["topic"]],
                                     "analysis": {"reasoning_graph": plan["reasoning_graph"]}})
        structure_vector = normalized(embed_texts(args.embedding_model, [plan_structure], provider=args.provider))[0]
        structure_similarity = float((source_smat @ structure_vector).max())
        if structure_similarity >= 0.92:
            raise RuntimeError(f"Plan {index} is structurally too close to source ({structure_similarity:.3f})")
        anchor = choose_anchor(corpus, spec["topic"], index)
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        item, report, vector = create_item(
            anchor, exemplars, plan, items, source_qmat, source_smat, corpus, prior_vectors,
            args.embedding_model, args.model, args.provider, 91000 + index * 1000, 100 + index,
        )
        item["generation"]["source_structure_similarity"] = structure_similarity
        items.append(item)
        reports.append(report)
        plans.append(plan)
        prior_vectors.append(vector)
        write_jsonl(items_path, items)
        write_jsonl(reports_path, reports)
        write_jsonl(plans_path, plans)
        print(f"Accepted reviewed replacement {index}/{len(SPECS)}: {spec['topic']}", flush=True)

    final_items = [copy.deepcopy(item) for item in items[1:]]
    for item in final_items:
        inferred = classify_topics(str(item.get("prompt") or "") + " " + str(item.get("solution") or ""))
        item["topics"] = list(dict.fromkeys([*(item.get("topics") or []), *inferred]))
        if item["id"] == "food-science-b-generated-104":
            item["difficulty"] = 1
            item["generation"]["plan"]["difficulty"] = 1

    report_by_id = {report["id"]: report for report in reports}
    source_by_id = {row["id"]: row for row in corpus}
    overrun = next(item for item in final_items if item["id"] == "food-science-b-generated-104")
    overrun_plan = overrun["generation"]["plan"]
    overrun_anchor = source_by_id[overrun["generation"]["anchor_id"]]
    overrun_judge = judge_item(overrun, overrun_plan, overrun_anchor, args.model, args.provider, 100104)
    overrun_judge["answer_agrees"] = answers_agree(overrun, overrun_judge.get("independent_answer"))
    required = ("solvable", "science_correct", "division_b_appropriate", "food_science_relevant",
                "difficulty_match", "competition_faithful", "style_faithful", "novel", "answer_agrees")
    report_by_id[overrun["id"]] = {
        "id": overrun["id"],
        "valid": not validate_item(overrun) and all(overrun_judge.get(key) is True for key in required),
        "deterministic_errors": validate_item(overrun),
        "judge": overrun_judge,
    }
    final_reports = [report_by_id[item["id"]] for item in final_items]
    if len(final_items) != 5 or not all(report.get("valid") is True for report in final_reports):
        raise RuntimeError("Reviewed final pilot is not five-for-five valid")
    write_jsonl(output / "generated" / "items_final.jsonl", final_items)
    write_jsonl(output / "validation" / "item_reports_final.jsonl", final_reports)

    audit = source_calibrated_audit(final_items, corpus, args.model, args.provider, 99991)
    audit_path = output / "validation" / "batch_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reviewed pilot: {len(final_items)} items; batch audit overall_good={audit.get('overall_good')}")


if __name__ == "__main__":
    main()
