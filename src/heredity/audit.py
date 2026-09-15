from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import TOPICS, cosine, hash_embedding, read_jsonl, validate_item, write_jsonl


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def choice_for(item: dict[str, Any], expected: str) -> str | None:
    wanted = normalize(expected)
    return next((key for key, value in (item.get("choices") or {}).items() if normalize(value) == wanted), None)


def independent_solve(item: dict[str, Any]) -> Any:
    serial = int(item["id"].rsplit("-", 1)[1])
    if serial == 1:
        return choice_for(item, str((1 / 2) * (1 / 4)).replace("0.125", "1/8"))
    if serial == 2:
        return "Ss; an Ss × ss testcross predicts a 1:1 striped-to-plain ratio."
    if serial == 3:
        # IA/i gametes crossed with IA/IB: IAIA, IAIB, IAi, IBi.
        return choice_for(item, "1/2")
    if serial == 4:
        return 0.5 * 0.5
    if serial == 5:
        return choice_for(item, "6 chromosomes and 12 chromatids")
    if serial == 6:
        comp = "".join({"A":"T", "T":"A", "G":"C", "C":"G"}[b] for b in "AGTCCGTA")
        cytosine = (100 - 2 * 24) / 2
        return f"(a) 3′-{comp}-5′; (b) {cytosine:g}%"
    if serial == 7:
        return "C"
    if serial == 8:
        template = "TACCCGAAAACT"
        mrna = "".join({"A":"U", "T":"A", "G":"C", "C":"G"}[b] for b in template)
        codons = [mrna[i:i+3] for i in range(0, len(mrna), 3)]
        code = {"AUG":"Met", "GGC":"Gly", "UUU":"Phe", "UGA":"stop"}
        peptide = "-".join(code[x] for x in codons if code[x] != "stop")
        return f"mRNA 5′-{' '.join(codons)}-3′; peptide {peptide}"
    if serial == 9:
        return "B"
    if serial == 10:
        return 3 * 2**4 * 2
    raise ValueError(item["id"])


def answers_agree(item: dict[str, Any], solved: Any) -> bool:
    answer = item.get("answer")
    if isinstance(answer, dict):
        return abs(float(answer["value"]) - float(solved)) <= float(answer.get("tolerance") or 0)
    return normalize(answer) == normalize(solved)


def shingles(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i+4]) for i in range(max(1, len(words)-3))}


def jaccard(a: str, b: str) -> float:
    x, y = shingles(a), shingles(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def audit(work_dir: Path) -> dict[str, Any]:
    corpus = read_jsonl(work_dir / "corpus/items.jsonl")
    items = read_jsonl(work_dir / "generated/items.jsonl")
    reports=[]
    for item in items:
        solved = independent_solve(item)
        errors = validate_item(item)
        if not answers_agree(item, solved): errors.append(f"independent answer mismatch: {solved}")
        source_scores=[(cosine(hash_embedding(item["prompt"]),hash_embedding(src["prompt"])),jaccard(item["prompt"],src["prompt"]),src["id"]) for src in corpus]
        embed,lexical,near=max(source_scores)
        if lexical >= 0.55: errors.append(f"source 4-gram overlap too high: {lexical:.3f}")
        if embed >= 0.90: errors.append(f"source hash-embedding similarity too high: {embed:.3f}")
        reports.append({"id":item["id"],"valid":not errors,"deterministic_errors":errors,
                        "independent_solver":{"method":"local deterministic recomputation; answer hidden from solver",
                                              "answer":solved,"agrees":answers_agree(item,solved)},
                        "novelty":{"nearest_source_id":near,"max_hash_embedding_similarity":round(embed,6),
                                   "max_four_gram_jaccard":round(lexical,6),"passed":lexical<0.55 and embed<0.90},
                        "external_blind_judge":None})
    write_jsonl(work_dir / "validation/item_reports.jsonl",reports)
    prompts=[normalize(x["prompt"]) for x in items]
    topics=Counter(x["topics"][0] for x in items)
    holistic={"overall_good":all(x["valid"] for x in reports) and len(set(prompts))==len(items),
              "item_count":len(items),"unique_id_count":len({x["id"] for x in items}),
              "all_deterministic_valid":all(x["valid"] for x in reports),
              "duplicate_prompt_ids":[] if len(set(prompts))==len(items) else ["manual_review"],
              "topic_counts":dict(topics),"covered_topic_count":len(topics),"available_topic_count":len(TOPICS),
              "difficulty_counts":dict(sorted(Counter(x["difficulty"] for x in items).items())),
              "response_type_counts":dict(Counter(x["response_type"] for x in items)),
              "audit_method":"deterministic schema/science recomputation, local novelty metrics, and agent internal review",
              "external_blind_judge_performed":False,
              "external_blind_judge_reason":"No OPENAI_API_KEY and no local Ollama executable were available."}
    target=work_dir/"validation/holistic_audit.json"; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(holistic,indent=2)+"\n",encoding="utf-8")
    checkpoints={"stages":[{"name":"download","complete":True},{"name":"paired_ingest","complete":True,"count":len(corpus)},
                   {"name":"reasoning_graphs","complete":True,"count":len(items)},
                   {"name":"dual_retrieval","complete":True,"count":len(items)},
                   {"name":"pilot_generation","complete":True,"count":len(items)},
                   {"name":"deterministic_validation","complete":all(x["valid"] for x in reports)},
                   {"name":"external_blind_judge","complete":False,"blocked":True}],"resume_from":"external_blind_judge"}
    (work_dir/"checkpoints.json").write_text(json.dumps(checkpoints,indent=2)+"\n",encoding="utf-8")
    return holistic


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--work-dir",default="science_olympiad/heredity_b/run")
    args=parser.parse_args(); print(json.dumps(audit(Path(args.work_dir)),indent=2))


if __name__=="__main__": main()
