from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from src.circuit_lab.model_client import embed_texts, generate_json
from src.circuit_lab.validate import answers_agree
from .common import read_jsonl, validate_item, write_jsonl
from .pipeline import (BATCH_SYSTEM, DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL, JUDGE_SYSTEM,
                       blind_answer_agrees, normalized, public_item, deterministic_bank_audit, summarize_usage)


CHOICES = {
    3: {"A":"Yes; a strong association proves the smoothies caused the outbreak.",
        "B":"Yes; case-control studies are experiments that prove cause and effect.",
        "C":"No; the study shows association, but more evidence is needed to call the smoothies the definite source.",
        "D":"No; an exposure cannot be a possible source unless a laboratory test is positive."},
    4: {"A":"Fruit salad","B":"Whole-grain bread","C":"Grilled fish","D":"Tomato soup"},
    5: {"A":"0.15","B":"0.25","C":"1.0","D":"4.0"},
    7: {"A":"Contaminated food at the shared Day 1 lunch","B":"Contaminated water brought from home",
        "C":"Person-to-person respiratory-droplet spread during close indoor contact","D":"Mosquito-borne transmission"},
    8: {"A":"Bacteria","B":"Parasites","C":"Viruses","D":"Chemical toxins"},
    9: {"A":"Offer stomach medicine at check-in","B":"Move the touchscreen to a different room",
        "C":"Require hand cleaning beside the snack table before children take food",
        "D":"Ask children to be cleaner without specific instructions"},
    10:{"A":"A real rise that keeps increasing after May","B":"Improved reporting after a surveillance-system change",
        "C":"A new ongoing air-pollution problem beginning in May","D":"A diagnostic change beginning in May"},
}


def strip_embedded_choices(prompt: str) -> str:
    prompt = re.split(r"(?mi)^\s*(?:Choose the best answer:\s*)?A[.)]\s+", prompt, maxsplit=1)[0].rstrip()
    prompt = re.sub(r"(?i)\n?Which of the following is the correct risk ratio[^?]*\?\s*$",
                    "\n\nWhich risk ratio compares the first group with the second?", prompt)
    return prompt


def replacement_six() -> dict[str, Any]:
    plan = {"response_type":"multiple_choice","topic":"data_interpretation","difficulty":2,
        "target_skill":"calculate and interpret an odds ratio from a 2-by-2 case-control table",
        "novel_context":"fictional juice exposure at a museum overnight program",
        "fixed_givens":["18 exposed cases","6 unexposed cases","8 exposed controls","16 unexposed controls"],
        "reasoning_graph":{"nodes":[
            {"id":"g1","type":"Given","label":"2-by-2 exposure counts for cases and controls"},
            {"id":"l1","type":"Law","label":"OR = (exposed cases × unexposed controls)/(unexposed cases × exposed controls)"},
            {"id":"t1","type":"Target","label":"odds ratio for juice exposure"}],
            "edges":[{"src":"g1","dst":"t1","type":"supports"},{"src":"l1","dst":"t1","type":"derived_from"}]},
        "verification":{"expected_answer":"C (6.0)","expected_value":6.0,"expression":"(18*16)/(6*8)",
                        "calculation":"(18×16)/(6×8)=288/48=6.0"},
        "distractor_mechanisms":["invert the odds ratio","compare raw exposed counts","divide only the cases"]}
    return {"id":"disease-detectives-b-generated-006","competition":"Science Olympiad",
        "event":"Disease Detectives","division":"B","response_type":"multiple_choice",
        "prompt":("At a fictional museum overnight program, investigators conduct a case-control study of stomach "
                  "illness and a fruit-juice exposure. Among ill participants (cases), 18 drank the juice and 6 did "
                  "not. Among well participants (controls), 8 drank the juice and 16 did not. Calculate the odds "
                  "ratio for illness associated with drinking the juice. Round to one decimal place."),
        "choices":{"A":"0.2","B":"1.5","C":"6.0","D":"12.0"},"answer":"C",
        "solution":"For a case-control table, OR = (18 × 16)/(6 × 8) = 288/48 = 6.0, so C is correct.",
        "points":1,"difficulty":2,"topics":["data_interpretation"],
        "generation":{"pipeline":"disease_detectives_b_graph_rag_curated","anchor_id":"2017-sample-test-III-16","plan":plan,
                      "replacement_reason":"Removed repeated highest-OR selection task flagged by batch audit."}}


def curate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = []
    for serial, item in enumerate(raw, 1):
        if serial == 6:
            final.append(replacement_six()); continue
        value = dict(item)
        if serial in CHOICES:
            value["response_type"] = "multiple_choice"
            value["choices"] = CHOICES[serial]
            value["answer"] = str(value["answer"]).strip()[0]
            value["prompt"] = strip_embedded_choices(str(value["prompt"]))
            value["points"] = 1
            value["generation"] = dict(value.get("generation") or {}, format_normalization="MCQ options moved to choices field")
        final.append(value)
    return final


def write_novelty_audit(root: Path, items: list[dict[str, Any]], embedding_model: str, provider: str,
                        validation_root: Path | None = None) -> dict[str, Any]:
    source_vectors = read_jsonl(root/"enriched"/"question_embeddings.jsonl")
    source_ids = [row["id"] for row in source_vectors]
    source_matrix = normalized([row["embedding"] for row in source_vectors])
    generated_vectors = embed_texts(embedding_model, [str(item["prompt"]) for item in items], provider=provider)
    generated_matrix = normalized(generated_vectors)
    source_matches = []
    for item, vector in zip(items, generated_matrix):
        scores = source_matrix @ vector; index = int(scores.argmax())
        source_matches.append({"id":item["id"],"nearest_source_id":source_ids[index],
                               "cosine_similarity":round(float(scores[index]),6)})
    pair_matches = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pair_matches.append({"id_a":items[i]["id"],"id_b":items[j]["id"],
                                 "cosine_similarity":round(float(generated_matrix[i] @ generated_matrix[j]),6)})
    pair_matches.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    audit = {"valid":max(x["cosine_similarity"] for x in source_matches) < .92 and
                     (not pair_matches or pair_matches[0]["cosine_similarity"] < .87),
             "source_threshold":.92,"generated_pair_threshold":.87,
             "maximum_source_similarity":max(x["cosine_similarity"] for x in source_matches),
             "maximum_generated_pair_similarity":pair_matches[0] if pair_matches else None,
             "source_matches":source_matches,"top_generated_pairs":pair_matches[:10]}
    output_root = validation_root or root/"validation"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root/"novelty_audit_final.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="science_olympiad/disease_detectives_b")
    parser.add_argument("--provider", choices=["openai","ollama"], default="openai")
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-tag", default="gpt-5-mini")
    parser.add_argument("--reuse-judges", action="store_true")
    args = parser.parse_args(); root = Path(args.run_dir)
    if args.provider == "openai" and args.model != DEFAULT_GENERATION_MODEL:
        raise ValueError(f"Disease Detectives production finalization requires exact model {DEFAULT_GENERATION_MODEL}")
    generated_root, validation_root = root/"generated"/args.output_tag, root/"validation"/args.output_tag
    usage_path = validation_root/"model_usage.jsonl"
    if args.provider == "openai": os.environ["OPENAI_USAGE_LOG"] = str(usage_path)
    items = read_jsonl(generated_root/"items.jsonl")
    corpus = [row for row in read_jsonl(root/"corpus"/"items.jsonl") if not validate_item(row)]
    errors = {x["id"]:validate_item(x) for x in items if validate_item(x)}
    if errors: raise RuntimeError(f"curated bank invalid: {errors}")
    write_jsonl(generated_root/"items_final.jsonl", items)
    novelty = write_novelty_audit(root, items, args.embedding_model, args.provider, validation_root)
    reports = []
    required=("solvable","science_correct","division_b_appropriate","event_relevant","difficulty_match",
              "competition_faithful","answer_agrees")
    prior_reports = {row["id"]:row for row in read_jsonl(validation_root/"item_reports_final.jsonl")} \
        if args.reuse_judges and (validation_root/"item_reports_final.jsonl").exists() else {}
    for item in items:
        if item["id"] in prior_reports:
            judge = prior_reports[item["id"]]["judge"]
        else:
            judge = generate_json(args.model, "RESPONSE_TYPE:" + str(item["response_type"]) +
                                  "\nTARGET TOPIC:" + str((item.get("topics") or [""])[0]) +
                                  "\nTARGET DIFFICULTY:" + str(item["difficulty"]) +
                                  "\nQUESTION:" + json.dumps(public_item(item)), provider=args.provider,
                                  system=JUDGE_SYSTEM, temperature=0, max_output_tokens=900)
            judge["answer_agrees"] = blind_answer_agrees(item, judge.get("independent_answer"))
        reports.append({"id":item["id"],"valid":all(judge.get(k) is True for k in required),
                        "deterministic_errors":validate_item(item),"judge":judge,
                        "judge_model":args.model,"blind_judge":True,"final_rejudge":True,
                        "novelty_gated_separately_by_embedding_audit":True})
    write_jsonl(validation_root/"item_reports_final.jsonl", reports)
    audit=deterministic_bank_audit(items, corpus)
    (validation_root/"deterministic_audit_final.json").write_text(json.dumps(audit,indent=2)+"\n")
    payload=[{k:x.get(k) for k in ("id","response_type","difficulty","topics","prompt","choices")} for x in items]
    batch=generate_json(args.model,json.dumps(payload),provider=args.provider,system=BATCH_SYSTEM,temperature=0,
                        max_output_tokens=2400,seed=92899)
    (validation_root/"batch_audit_final.json").write_text(json.dumps(batch,indent=2)+"\n")
    all_judged = len(reports) == len(items) and all(x.get("valid") and x.get("blind_judge") and x.get("judge_model") == args.model for x in reports)
    summary = {"generation_model":args.model,"judge_model":args.model,"embedding_model":args.embedding_model,
               "item_count":len(items),"all_items_model_generated":all((x.get("generation") or {}).get("generation_model") == args.model for x in items),
               "all_items_blind_judged":all_judged,"deterministic_valid":audit["valid"],
               "coverage_valid":audit["coverage_valid"],"novelty_valid":novelty["valid"],
               "batch_audit_overall_good":batch.get("overall_good") is True,"usage":summarize_usage(usage_path)}
    (validation_root/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Final: deterministic={audit['valid']} novelty={novelty['valid']} blind_judges={all_judged} batch={batch.get('overall_good')}")


if __name__ == "__main__": main()
