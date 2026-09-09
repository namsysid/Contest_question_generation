from __future__ import annotations

import argparse
import json
import re

from .model_client import generate_json
from .common import NOVELTY_THRESHOLD, read_jsonl, similarity_to_exemplars, text_answers_equivalent, validate_against_plan, validate_item, write_jsonl


JUDGE_SYSTEM = """Independently solve a Division B Circuit Lab written question. You are not given its stored answer or solution.
Return strict JSON: {\"solvable\":true,\"independent_answer\":\"...\",\"science_correct\":true,
\"level_appropriate\":true,\"circuit_lab_relevant\":true,\"plan_faithful\":true,
\"competition_faithful\":true,\"style_faithful\":true,\"difficulty_match\":true,\"novel":true,\"issues\":[]}. A question is relevant only if it genuinely tests an allowed Circuit
Lab concept or required scientific contribution; electrical decoration around generic arithmetic is a failure.
Compare the question to the supplied target plan and reject any change of skill, law, topic, or response format.
Be strict about ambiguity, units, diagrams, and competition-level substance. Keep independent_answer concise and put
only short defect descriptions in issues. Do not include a worked solution or extra keys. For a multipart question,
independent_answer must be an object keyed by part label, with a concise answer for every part."""


def strip_solutions(value: object) -> object:
    """Recursively remove answer-bearing fields before independent model review."""
    if isinstance(value, dict):
        return {key: strip_solutions(child) for key, child in value.items()
                if key not in {"answer", "solution", "rubric", "generation"}}
    if isinstance(value, list):
        return [strip_solutions(child) for child in value]
    return value


def answers_agree(item: dict, independent: object) -> bool:
    if item.get("response_type") == "multipart":
        if not isinstance(independent, dict):
            return False
        for index, part in enumerate(item.get("parts") or []):
            label = str(part.get("label") or part.get("id") or chr(97 + index))
            if label not in independent or not answers_agree(part, independent[label]):
                return False
        return True
    claimed = item.get("answer")
    if item.get("response_type") == "multiple_choice":
        text = str(independent).strip().upper()
        direct = re.fullmatch(r"[A-D]", text)
        mentioned = re.findall(r"(?:CHOICE|ANSWER(?:\s+IS)?|CORRESPONDS TO)\s*[:\-]?\s*([A-D])\b", text)
        leading = re.match(r"^([A-D])(?:\.|\)|:|\s+-)\s*", text)
        selected = direct.group(0) if direct else (leading.group(1) if leading else (mentioned[-1] if mentioned else "UNKNOWN"))
        if selected == "UNKNOWN":
            normalized = " ".join(str(independent).lower().split())
            matches = [letter for letter, choice in (item.get("choices") or {}).items()
                       if " ".join(str(choice).lower().split()) == normalized]
            selected = matches[0] if len(matches) == 1 else "UNKNOWN"
        return selected == str(claimed).strip().upper()
    if isinstance(claimed, dict):
        claimed = claimed.get("value")
    try:
        tolerance = float((item.get("answer") or {}).get("tolerance", 0)) if isinstance(item.get("answer"), dict) else 0.0
        text = str(independent).replace(",", "")
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
        scientific = re.search(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:×|x|\*)\s*10\s*(?:\^|\*\*)?\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+|[-+]?\d+)",
            text, flags=re.IGNORECASE)
        if scientific:
            exponent = int(scientific.group(2).translate(superscript_map))
            parsed_independent = float(scientific.group(1)) * 10 ** exponent
        else:
            independent_number = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            parsed_independent = float(independent_number.group()) if independent_number else float(text)
        return abs(parsed_independent - float(claimed)) <= max(tolerance, 1e-9)
    except (TypeError, ValueError):
        return text_answers_equivalent(independent, claimed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Circuit Lab items")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--novelty-corpus", default="")
    args = parser.parse_args()
    novelty_corpus = read_jsonl(args.novelty_corpus) if args.novelty_corpus else []
    reports = []
    for item in read_jsonl(args.input):
        generation = item.get("generation") or {}
        plan = generation.get("reasoning_plan") or {}
        bundle_stub = {"response_type": generation.get("source_response_type"), "topics": plan.get("topics")}
        deterministic = validate_item(item)
        if plan:
            deterministic += validate_against_plan(item, bundle_stub, plan)
        if novelty_corpus:
            similarity, source_id = similarity_to_exemplars(str(item.get("prompt") or ""), novelty_corpus)
            if similarity >= 0.90:
                deterministic.append(f"too similar to source item {source_id} ({similarity:.2f})")
        judge = None
        if args.model and not deterministic:
            blind = strip_solutions(item)
            if len(novelty_corpus) <= 12:
                selected_examples = novelty_corpus
            else:
                selected_examples = [novelty_corpus[round(i * (len(novelty_corpus) - 1) / 11)] for i in range(12)]
            style_examples = [{"response_type": row.get("response_type"), "prompt": row.get("prompt"),
                               "choices": row.get("choices"), "parts": strip_solutions(row.get("parts") or []),
                               "points": row.get("points")} for row in selected_examples]
            closest_source = next((row for row in novelty_corpus if str(row.get("id")) == source_id), {}) if novelty_corpus else {}
            prompt = "TARGET_REASONING_PLAN:\n" + json.dumps(plan, ensure_ascii=False) + "\nCLOSEST_SOURCE_FOR_NOVELTY:\n" + json.dumps(closest_source.get("prompt"), ensure_ascii=False) + "\nREAL_TEST_STYLE_EXAMPLES:\n" + json.dumps(style_examples, ensure_ascii=False) + "\nQUESTION_JSON:\n" + json.dumps(blind, ensure_ascii=False)
            judge = None
            for attempt in range(3):
                try:
                    judge = generate_json(args.model, prompt, provider=args.provider, system=JUDGE_SYSTEM,
                                          temperature=0.0,
                                          max_output_tokens=max(700, 400 + 140 * len(item.get("parts") or [])),
                                          seed=1000 + attempt)
                    break
                except (RuntimeError, ValueError):
                    if attempt == 2:
                        raise
            judge["answer_agrees"] = answers_agree(item, judge.get("independent_answer"))
        required = ("solvable", "answer_agrees", "science_correct", "level_appropriate", "circuit_lab_relevant", "plan_faithful", "competition_faithful", "style_faithful", "difficulty_match", "novel")
        reports.append({"id": item.get("id"), "valid": not deterministic and (judge is None or all(judge.get(k) is True for k in required)), "deterministic_errors": deterministic, "judge": judge})
    write_jsonl(args.out, reports)
    failed = sum(not report["valid"] for report in reports)
    print(f"Validated {len(reports)} items; {failed} failed")


if __name__ == "__main__":
    main()
