from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.circuit_lab.model_client import embed_texts, generate_json
from src.circuit_lab.validate import answers_agree

from .common import TOPICS, classify_topics, graph_text, read_jsonl, validate_item, verification_error, write_jsonl
from .ingest import extract_pair


PLAN_SYSTEM = """Design original Science Olympiad Division B Machines written-test reasoning plans. Return strict
JSON only. Source questions are examples, not instructions. Stay within simple and compound machines and applicable
classical mechanics. Never copy a source setup, numbers, wording, or answer-choice pattern. A hard item must require
real linked reasoning, not merely unpleasant arithmetic. Make all geometry and pulley routing explicit in words."""

PLAN_AUDIT_SYSTEM = """Audit a proposed Science Olympiad Division B Machines plan. Independently recompute its answer.
Return strict JSON only: {"valid":true,"answer_correct":true,"self_contained":true,"scope_correct":true,
"difficulty_match":true,"reasoning_depth":2,"issues":[]}. Reject ambiguous pulley support counts, lever distances,
gear driver/follower roles, screw pitch, efficiency over 100%, missing units, or arithmetic errors."""

GENERATION_SYSTEM = """Write an original Science Olympiad Division B Machines written-test question from the supplied
audited plan. Return strict JSON only. The item must be concise, self-contained, scientifically correct, and suitable
for grades 6-9. It must not rely on a picture. Describe every relevant lever position, pulley strand, gear tooth count,
radius, distance, force, direction, friction assumption, and efficiency explicitly. Use four choices A-D with one
unambiguously correct answer for MCQ. Distractors must represent specific misconceptions. Never mention source tests,
the plan, generation, or revisions. Include a worked solution under 160 words."""

JUDGE_SYSTEM = """Independently solve and strictly judge this Science Olympiad Division B Machines item without seeing
its answer or solution. Return strict JSON only: {"solvable":true,"independent_answer":"A|B|C|D or constructed answer",
"science_correct":true,"division_b_appropriate":true,"machines_relevant":true,"difficulty_match":true,
"competition_faithful":true,"style_faithful":true,"novel":true,"issues":[]}. Check formulas, units, arithmetic,
pulley support counts, torque arms, driver/follower directions, work-energy consistency, and efficiency. Reject generic
arithmetic dressed with machine nouns, ambiguous topology, trivia-dominated questions, or direct source rewrites."""

BATCH_AUDIT_SYSTEM = """Audit a complete generated Science Olympiad Division B Machines question set. Return strict
JSON only: {"overall_good":true,"coverage_good":true,"difficulty_distribution_good":true,"repetition_good":true,
"competition_faithful":true,"duplicate_or_near_duplicate_ids":[],"weak_item_ids":[],"missing_areas":[],"issues":[]}.
Be severe. Flag items that test the same skill with merely different objects or numbers, mislabeled difficulty, generic
textbook mechanics with no Machines relevance, verbose stems, trivial distractors, or insufficient breadth. This is a
question bank for the written-test component only: do not demand build-device scoring rules, impound procedures, or
diagram-dependent tasks. Judge style and depth against the supplied real current-season written-test examples. A small
number of basic warm-ups is appropriate; they become a defect only when they dominate or repeat the same skill.
Real difficulty-1 calibration items are intentionally direct, often testing one fact, definition, or arithmetic
relation. Do not mark a clear difficulty-1 item unfaithful merely for being direct; judge whether its exact target is
already repeated and whether its distractors are meaningful.
Judge difficulty distribution proportionally: project the supplied source counts to the generated set size, round to
whole items, and accept deviations of at most one item per level. Do not call two items near-duplicates merely because
both use a broad quantity such as IMA, torque, or efficiency; they must test substantially the same machine relation
through substantially the same reasoning path. Coverage must be proportionate to set size, so a rare source niche is
not automatically required. Set overall_good true exactly when the four component booleans are all true."""


def normalized(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items()
            if key not in {"answer", "solution", "analysis", "graph_text", "generation"}}


def load_or_ingest(source_dir: Path, corpus_path: Path, model: str, provider: str) -> list[dict[str, Any]]:
    if corpus_path.exists():
        return read_jsonl(corpus_path)
    pairs = [(source_dir / "bullso-test.pdf", source_dir / "bullso-key.pdf"),
             (source_dir / "berkeley-test.pdf", source_dir / "berkeley-key.pdf"),
             (source_dir / "pembroke-test.pdf", source_dir / "pembroke-key.pdf")]
    rows: list[dict[str, Any]] = []
    for test_path, key_path in pairs:
        rows.extend(extract_pair(test_path, key_path, model, provider))
        write_jsonl(corpus_path, rows)
    return rows


def load_or_embed(corpus: list[dict[str, Any]], q_path: Path, s_path: Path, model: str,
                  provider: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if q_path.exists() and s_path.exists():
        return read_jsonl(q_path), read_jsonl(s_path)
    q_text = [str(row.get("prompt") or "") for row in corpus]
    s_text = [str(row.get("graph_text") or graph_text(row)) for row in corpus]
    q_vec = embed_texts(model, q_text, provider=provider)
    s_vec = embed_texts(model, s_text, provider=provider)
    q_rows = [{"id": row["id"], "embedding": vector} for row, vector in zip(corpus, q_vec)]
    s_rows = [{"id": row["id"], "embedding": vector} for row, vector in zip(corpus, s_vec)]
    write_jsonl(q_path, q_rows)
    write_jsonl(s_path, s_rows)
    return q_rows, s_rows


def schedule_anchors(corpus: list[dict[str, Any]], count: int) -> list[tuple[dict[str, Any], str, int]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus:
        for topic in row.get("topics") or []:
            if topic in TOPICS:
                by_topic[topic].append(row)
    available = [topic for topic in TOPICS if by_topic[topic]]
    if not available:
        raise RuntimeError("No classified Machines source anchors")
    offsets = defaultdict(int)
    # Match the empirical source distribution at any requested count. Quantile sampling
    # avoids dropping a small hard tail while keeping the many real warm-up questions.
    source_difficulties = sorted(max(1, min(5, int(row.get("difficulty") or 2))) for row in corpus)
    difficulty_targets = [
        source_difficulties[min(len(source_difficulties) - 1,
                                int((index + 0.5) * len(source_difficulties) / count))]
        for index in range(count)
    ]
    schedule = []
    for index in range(count):
        topic = available[index % len(available)]
        target_difficulty = difficulty_targets[index]
        group = sorted(by_topic[topic], key=lambda row: abs(int(row.get("difficulty") or 2) - target_difficulty))
        anchor = group[offsets[topic] % len(group)]
        offsets[topic] += 1
        schedule.append((anchor, topic, target_difficulty))
    return schedule


def retrieve_exemplars(anchor: dict[str, Any], corpus: list[dict[str, Any]], q_rows: list[dict[str, Any]],
                       s_rows: list[dict[str, Any]], count: int = 4) -> list[dict[str, Any]]:
    ids = [row["id"] for row in corpus]
    pos = {value: index for index, value in enumerate(ids)}
    qmat = normalized([next(row["embedding"] for row in q_rows if row["id"] == value) for value in ids])
    smat = normalized([next(row["embedding"] for row in s_rows if row["id"] == value) for value in ids])
    anchor_index = pos[anchor["id"]]
    candidates = [i for i, row in enumerate(corpus) if i != anchor_index and
                  row.get("response_type") == anchor.get("response_type")]
    ranked = sorted(candidates, key=lambda i: float(0.55 * (qmat[i] @ qmat[anchor_index]) +
                                                           0.45 * (smat[i] @ smat[anchor_index])), reverse=True)
    return [corpus[i] for i in ranked[:count]]


def make_plan(anchor: dict[str, Any], exemplars: list[dict[str, Any]], topic: str, target_difficulty: int,
              prior: list[dict[str, Any]], model: str, provider: str, seed: int,
              required_brief: str = "") -> dict[str, Any]:
    response_type = anchor["response_type"]
    difficulty = target_difficulty
    prompt = f"""Create a novel plan matching this calibration target:
RESPONSE TYPE: {response_type}
REQUIRED TOPIC: {topic}
EXACT DIFFICULTY: {difficulty}/5
REQUIRED TASK BRIEF: {required_brief or 'Choose a novel task within the required topic.'}
ANCHOR (do not copy): {json.dumps(public_item(anchor), ensure_ascii=False)}
RELATED REAL EXAMPLES (do not copy): {json.dumps([public_item(row) for row in exemplars], ensure_ascii=False)}
PRIOR GENERATED SKILLS/CONTEXTS TO AVOID: {json.dumps([{"skill": row.get("target_skill"), "context": row.get("novel_context")} for row in prior[-20:]], ensure_ascii=False)}

Return {{"response_type":"{response_type}","topic":"{topic}","difficulty":{difficulty},
"target_skill":"specific skill","novel_context":"specific new machine setup","fixed_givens":["all exact givens"],
"reasoning_graph":{{"nodes":[{{"id":"g1","type":"Given|Law|Target|Constraint|Trap","label":"..."}}],
"edges":[{{"src":"g1","dst":"t1","type":"supports|depends_on|derived_from|rules_out"}}]}},
"verification":{{"expected_answer":"answer with unit","expected_value":12.3,
"expression":"numbers and + - * / ** parentheses only; omit expected_value and expression for nonnumeric tasks",
"calculation":"independent derivation under 100 words"}},"distractor_mechanisms":["specific error 1","specific error 2","specific error 3"]}}.
For difficulty 1, use recall or one direct inference. Difficulty 2 requires one meaningful application. Difficulty 3+
requires at least two linked decisions/calculations or a non-obvious conceptual distinction. At difficulty 3, combine
at least two relations or machine stages; changing only numbers does not add depth. Pick new values and setup."""
    rejection = ""
    for attempt in range(8):
        plan = generate_json(model, prompt + rejection, provider=provider, system=PLAN_SYSTEM, temperature=0.25,
                             max_output_tokens=1800, seed=seed + attempt)
        plan["response_type"], plan["topic"], plan["difficulty"] = response_type, topic, difficulty
        arithmetic = verification_error(plan)
        if arithmetic:
            rejection = f"\nREJECTED: {arithmetic}. Recalculate and return a corrected complete plan."
            if response_type != "numeric":
                rejection += " If the keyed answer is conceptual, omit both expected_value and expression entirely."
            continue
        graph = plan.get("reasoning_graph") or {}
        nodes, edges = graph.get("nodes") or [], graph.get("edges") or []
        if len(nodes) < difficulty + 2 or len(edges) < difficulty:
            rejection = (f"\nREJECTED: difficulty {difficulty} requires at least {difficulty + 2} meaningful graph "
                         f"nodes and {difficulty} dependency edges; received {len(nodes)} and {len(edges)}.")
            continue
        if not (plan.get("fixed_givens") and (plan.get("verification") or {}).get("expected_answer")):
            rejection = "\nREJECTED: provide complete fixed_givens and an independently derived expected_answer."
            continue
        plan["audit"] = {"valid": True, "method": "deterministic_graph_and_arithmetic"}
        return plan
    raise RuntimeError("plan failed audit: " + rejection)


def generation_prompt(plan: dict[str, Any], exemplars: list[dict[str, Any]], prior: list[dict[str, Any]]) -> str:
    response_type = plan["response_type"]
    if response_type == "multiple_choice":
        schema: dict[str, Any] = {"id": "machines-b-unique-id", "response_type": "multiple_choice",
                                  "prompt": "string", "choices": {key: "string" for key in "ABCD"},
                                  "answer": "A-D", "solution": "worked solution", "points": 1,
                                  "difficulty": plan["difficulty"], "topics": [plan["topic"]]}
    else:
        answer: Any = {"value": 0, "unit": "string", "tolerance": 0.01} if response_type == "numeric" else "concise answer"
        schema = {"id": "machines-b-unique-id", "response_type": response_type, "prompt": "string",
                  "answer": answer, "solution": "worked solution", "rubric": ["credit-bearing step"],
                  "points": 2, "difficulty": plan["difficulty"], "topics": [plan["topic"]]}
    word_limit = 260 if int(plan.get("difficulty") or 1) >= 3 else 90
    return f"""AUDITED PLAN (authoritative):
{json.dumps(plan, ensure_ascii=False)}
REAL STYLE EXAMPLES (style only; do not copy):
{json.dumps([public_item(row) for row in exemplars], ensure_ascii=False)}
PRIOR QUESTIONS (avoid repetition):
{json.dumps([row.get('prompt') for row in prior[-20:]], ensure_ascii=False)}
Return exactly one item with this schema: {json.dumps(schema)}
Use every fixed given exactly. Make the keyed answer and worked solution agree with verification.expected_answer.
Keep the prompt under {word_limit} words. Do not require a diagram or external information."""


def judge_item(item: dict[str, Any], plan: dict[str, Any], anchor: dict[str, Any], model: str,
               provider: str, seed: int) -> dict[str, Any]:
    blind = public_item(item)
    return generate_json(model, "TARGET PLAN:\n" + json.dumps(plan, ensure_ascii=False) +
                         "\nSOURCE ANCHOR FOR NOVELTY ONLY:\n" + json.dumps(public_item(anchor), ensure_ascii=False) +
                         "\nQUESTION:\n" + json.dumps(blind, ensure_ascii=False), provider=provider,
                         system=JUDGE_SYSTEM, temperature=0, max_output_tokens=850, seed=seed)


def create_item(anchor: dict[str, Any], exemplars: list[dict[str, Any]], plan: dict[str, Any],
                prior: list[dict[str, Any]], source_qmat: np.ndarray, source_rows: list[dict[str, Any]],
                prior_vectors: list[list[float]], embedding_model: str, model: str, provider: str,
                seed: int, serial: int) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    retry = ""
    for attempt in range(7):
        item = generate_json(model, generation_prompt(plan, exemplars, prior) + retry, provider=provider,
                             system=GENERATION_SYSTEM, temperature=0.3, max_output_tokens=1800,
                             seed=seed + attempt)
        item["id"] = f"machines-b-generated-{serial:03d}"
        item["response_type"] = plan["response_type"]
        item["difficulty"] = plan["difficulty"]
        item["topics"] = [plan["topic"]]
        item.setdefault("points", 1 if item["response_type"] == "multiple_choice" else 2)
        if item.get("response_type") == "numeric" and isinstance(item.get("answer"), dict):
            value = item["answer"].get("value")
            if isinstance(value, (int, float)):
                item["answer"]["tolerance"] = max(float(item["answer"].get("tolerance") or 0),
                                                   abs(float(value)) * 0.01, 1e-6)
        errors = validate_item(item)
        inferred_topics = classify_topics(str(item.get("prompt") or "") + " " + str(item.get("solution") or ""))
        if plan["topic"] not in inferred_topics:
            errors.append(f"topic drift: expected {plan['topic']}, inferred {inferred_topics}")
        word_limit = 260 if int(plan.get("difficulty") or 1) >= 3 else 90
        if len(re.findall(r"\w+", str(item.get("prompt") or ""))) > word_limit:
            errors.append(f"prompt exceeds {word_limit} words")
        if item.get("response_type") == "multiple_choice" and re.search(r"\n\s*A[.)]", str(item.get("prompt") or "")):
            errors.append("prompt repeats answer choices that belong only in choices")
        if item.get("response_type") in {"short_answer", "numeric"} and re.search(
                r"(?m)^\s*A[.)]\s+.*\n\s*B[.)]\s+", str(item.get("prompt") or "")):
            errors.append("constructed-response prompt must not contain A-D answer choices")
        candidate_vector = embed_texts(embedding_model, [str(item.get("prompt") or "")], provider=provider)[0]
        candidate = normalized([candidate_vector])[0]
        source_similarity = source_qmat @ candidate
        nearest_index = int(np.argmax(source_similarity))
        nearest_score = float(source_similarity[nearest_index])
        if nearest_score >= 0.94:
            errors.append(f"embedding similarity {nearest_score:.3f} to source {source_rows[nearest_index]['id']}")
        if prior_vectors:
            prior_similarity = normalized(prior_vectors) @ candidate
            if float(prior_similarity.max()) >= 0.875:
                errors.append(f"too similar to another generated item ({float(prior_similarity.max()):.3f})")
        judge = None
        if not errors:
            judge = judge_item(item, plan, anchor, model, provider, seed + 200 + attempt)
            judge["answer_agrees"] = answers_agree(item, judge.get("independent_answer"))
            required = ("solvable", "science_correct", "division_b_appropriate", "machines_relevant",
                        "difficulty_match", "competition_faithful", "style_faithful", "novel", "answer_agrees")
            if not all(judge.get(key) is True for key in required):
                failed_checks = [key for key in required if judge.get(key) is not True]
                errors.extend(map(str, judge.get("issues") or
                                  ["independent judge failed: " + ", ".join(failed_checks)]))
        if not errors:
            item["generation"] = {"pipeline": "machines_b_graph_rag", "anchor_id": anchor["id"], "plan": plan,
                                  "source_embedding_similarity": nearest_score}
            return item, {"id": item["id"], "valid": True, "deterministic_errors": [], "judge": judge}, candidate_vector
        retry = ("\nREJECTED DRAFT:\n" + json.dumps(item, ensure_ascii=False) +
                 "\nRepair every issue without discussing revisions: " + "; ".join(errors))
    raise RuntimeError("generation failed: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-first Machines Division B question pipeline")
    parser.add_argument("--source-dir", default="science_olympiad/machines_b/sources")
    parser.add_argument("--corpus", default="", help="Reuse an already extracted Machines source corpus")
    parser.add_argument("--work-dir", default="science_olympiad/machines_b/run")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--generation-model", default="gpt-5.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--seed", type=int, default=7300)
    args = parser.parse_args()

    root = Path(args.work_dir)
    corpus_path = root / "corpus" / "items.jsonl"
    q_path, s_path = root / "enriched" / "question_embeddings.jsonl", root / "enriched" / "structure_embeddings.jsonl"
    plans_path, items_path = root / "generated" / "reasoning_plans.jsonl", root / "generated" / "items.jsonl"
    reports_path = root / "validation" / "item_reports.jsonl"
    if args.corpus:
        corpus = read_jsonl(args.corpus)
        write_jsonl(corpus_path, corpus)
    else:
        corpus = load_or_ingest(Path(args.source_dir), corpus_path, args.generation_model, args.provider)
    source_errors = [(row["id"], validate_item(row)) for row in corpus]
    removed = {item_id for item_id, errors in source_errors if errors}
    corpus = [row for row in corpus if row["id"] not in removed]
    if len(corpus) < 12:
        raise RuntimeError(f"Only {len(corpus)} valid text-safe source questions remain")
    q_rows, s_rows = load_or_embed(corpus, q_path, s_path, args.embedding_model, args.provider)
    source_qmat = normalized([row["embedding"] for row in q_rows])
    plans = read_jsonl(plans_path) if plans_path.exists() else []
    items = read_jsonl(items_path) if items_path.exists() else []
    reports = read_jsonl(reports_path) if reports_path.exists() else []
    prior_vectors = embed_texts(args.embedding_model, [row["prompt"] for row in items], provider=args.provider) if items else []
    schedule = schedule_anchors(corpus, args.count)
    failures = []
    while len(items) < args.count:
        serial = len(items) + 1
        anchor, topic, target_difficulty = schedule[serial - 1]
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        try:
            plan = make_plan(anchor, exemplars, topic, target_difficulty, plans, args.generation_model, args.provider,
                             args.seed + serial * 1000)
            item, report, vector = create_item(anchor, exemplars, plan, items, source_qmat, corpus, prior_vectors,
                                               args.embedding_model, args.generation_model, args.provider,
                                               args.seed + serial * 1000 + 300, serial)
        except (RuntimeError, ValueError) as exc:
            failures.append({"serial": serial, "anchor_id": anchor["id"], "topic": topic,
                             "difficulty": target_difficulty, "error": str(exc)})
            if len(failures) >= 20:
                break
            schedule.append(schedule.pop(serial - 1))
            continue
        plans.append(plan)
        items.append(item)
        reports.append(report)
        prior_vectors.append(vector)
        write_jsonl(plans_path, plans)
        write_jsonl(items_path, items)
        write_jsonl(reports_path, reports)
        print(f"Accepted {len(items)}/{args.count}: {topic} {item['response_type']} difficulty {item['difficulty']}", flush=True)
    write_jsonl(root / "validation" / "generation_failures.jsonl", failures)
    if items:
        audit_payload = [{"id": row["id"], "response_type": row["response_type"],
                          "difficulty": row["difficulty"], "topics": row["topics"],
                          "prompt": row["prompt"], "choices": row.get("choices")} for row in items]
        batch_audit = generate_json(args.generation_model, json.dumps(audit_payload, ensure_ascii=False),
                                    provider=args.provider, system=BATCH_AUDIT_SYSTEM, temperature=0,
                                    max_output_tokens=5000, seed=args.seed + 999999)
        target = root / "validation" / "batch_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(batch_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Batch audit overall_good={batch_audit.get('overall_good')}")
    print(f"Complete: {len(items)} validated items; {len(failures)} abandoned attempts")


if __name__ == "__main__":
    main()
