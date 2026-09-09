from __future__ import annotations

import argparse
import ast
import json
import operator
import re
from pathlib import Path

from .model_client import generate_json
from .common import OUT_OF_SCOPE, UNRELATED_DOMAINS, classify_topics, read_jsonl, similarity_to_exemplars, write_jsonl


SYSTEM = """Create a novel reasoning graph for a Science Olympiad Division B Circuit Lab written question.
Return strict JSON only. Preserve the retrieved graph's depth, dependency pattern, and misconception profile, but do
not copy its context, quantities, wording, or exact sequence. The task must genuinely assess electricity, magnetism,
electrical safety/control, circuit analysis, or the event's required history of scientific contributions. Never use
a scientist's biography as decoration for date or age arithmetic. Do not introduce any unrelated subject,
competition, trivia, or context. Do not require mechanics equations, force-to-acceleration conversion, or mass.
Do not describe a diode as a triangle pointing at a line; use an accurate standard schematic description or no symbol.
For an ordinary silicon rectifier at modest current, use a plausible forward drop near 0.6-0.8 V, not 2 V."""

PLAN_JUDGE_SYSTEM = """Audit a proposed Division B Circuit Lab question plan before question writing. Return strict
JSON only: {"self_contained":true,"unambiguous":true,"scope_compliant":true,"verified_answer_correct":true,
"reasoning_depth":3,"issues":[]}.
Reject fault diagnosis without explicit topology, component values or healthy reference behavior. Reject a claimed
difficulty of 3 or higher if the task is only recall, naming a law/component, or one direct inference. For component
identification, require a distinct observation that uniquely determines every requested label; reject labels that
can be swapped without changing the evidence. Reject topics beyond the stated level. Return at most three issues,
each under 25 words; do not provide a worked solution."""

CONCEPT_MENU = [
    "mystery resistor from an explicitly specified voltage-divider measurement",
    "power and energy in an explicitly specified multi-branch DC resistor circuit",
    "relay and switch logic with named normally-open and normally-closed contacts",
    "net electrostatic force from three collinear point charges with stated positions",
    "equivalent capacitance and stored charge in a stated series-parallel capacitor network",
    "ideal transformer voltage current and power relationships", "electromagnet comparative design",
    "magnetic-force direction using a stated current and field", "motor and generator energy conversion",
]

VERIFIED_CONCEPT_SPECS = {
    CONCEPT_MENU[0]: {"fixed_givens": ["12 V source", "R1 = 2.0 kΩ in series with Rx", "voltage across R1 = 8.0 V"],
                      "expected_answer": "1.0 kΩ", "expected_value": 1.0,
                      "expression": "2.0*(12/8-1)",
                      "calculation": "Rx = R1(Vtotal/VR1 - 1) = 2.0 kΩ(12/8 - 1) = 1.0 kΩ."},
    CONCEPT_MENU[1]: {"fixed_givens": ["24 V source", "R1 = 6 Ω in series with a parallel pair R2 = 12 Ω and R3 = 8 Ω"],
                      "expected_answer": "53.33 W", "expected_value": 53.3333333333333,
                      "expression": "24**2/(6+1/(1/12+1/8))",
                      "calculation": "Rparallel = 1/(1/12+1/8)=4.8 Ω; Rtotal=10.8 Ω; P=24²/10.8=53.33 W."},
    CONCEPT_MENU[2]: {"fixed_givens": ["NO pushbutton S1 and NC pushbutton S2 are in series with a 12 V relay coil",
                                        "relay NO contact controls lamp L1", "relay NC contact controls lamp L2",
                                        "determine both lamps for all four pressed/released combinations of S1 and S2"],
                      "expected_answer": "S1 released/S2 released: L1 OFF, L2 ON; S1 released/S2 pressed: L1 OFF, L2 ON; S1 pressed/S2 released: L1 ON, L2 OFF; S1 pressed/S2 pressed: L1 OFF, L2 ON.",
                      "calculation": "The coil energizes only when NO S1 is pressed (closed) and NC S2 is released (closed). Only then relay NO lights L1 and relay NC opens L2; in all other input states L1 is OFF and L2 is ON."},
    CONCEPT_MENU[3]: {"fixed_givens": ["q1=+2 μC at x=0 m", "q2=-3 μC at x=0.5 m", "q3=+1 μC at x=1.0 m",
                                        "k=8.99×10^9 N·m²/C²"],
                      "expected_answer": "0.10788 N to the left", "expected_value": 0.10788,
                      "expression": "8.99e9*2e-6*3e-6/(0.5**2)-8.99e9*1e-6*3e-6/(0.5**2)",
                      "calculation": "Attraction to q1 is 0.21576 N left; attraction to q3 is 0.10788 N right; net is 0.10788 N left."},
    CONCEPT_MENU[4]: {"fixed_givens": ["24 V source", "C1=6 μF and C2=12 μF in parallel", "that pair is in series with C3=4 μF"],
                      "expected_answer": "78.55 μC on C3", "expected_value": 78.5454545454545,
                      "expression": "(1/(1/(6+12)+1/4))*24",
                      "calculation": "Cparallel=18 μF; Ctotal=1/(1/18+1/4)=3.2727 μF; series charge, including C3, is 3.2727×24=78.55 μC."},
}
VERIFIED_ANCHOR_SPECS = {
    "circuit-lab-b-test-mc-24": {
        "fixed_givens": ["12 V DC source", "indicator forward drop = 2.0 V", "desired series current = 20 mA",
                          "indicator and current-limiting resistor are in one unbranched series loop"],
        "expected_answer": "500 Ω", "expected_value": 500,
        "expression": "(12-2)/0.020",
        "calculation": "The resistor drops 12-2=10 V. R=V/I=10/0.020=500 Ω."
    }
}
PLAN_REPAIR_ARTIFACTS = ("correction:", "let's check", "however", "but the expression")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate_numeric_expression(expression: str) -> float:
    """Evaluate planner arithmetic containing only numeric literals and basic operators."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("verification expression must contain only numbers and + - * / ** parentheses")
    return walk(ast.parse(expression, mode="eval"))


def verification_error(verification: dict) -> str | None:
    expression = str(verification.get("expression") or "").strip()
    expected_value = verification.get("expected_value")
    if not expression:
        return None
    try:
        calculated = evaluate_numeric_expression(expression)
        expected = float(expected_value)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"invalid numeric verification: {exc}"
    if abs(calculated - expected) > max(1e-9, abs(expected) * 0.01):
        return f"expression evaluates to {calculated:g}, not expected_value {expected:g}"
    answer_text = str(verification.get("expected_answer") or "")
    answer_match = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-zμΩ]+(?:/[A-Za-z²]+)?)", answer_text)
    if answer_match and (answer_match.group(2).startswith(("μ", "m", "k", "M")) or
                         answer_match.group(2).lower().startswith(("micro", "milli", "kilo", "mega"))):
        unit = answer_match.group(2)
        calculation = str(verification.get("calculation") or "")
        same_unit = re.findall(rf"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*{re.escape(unit)}\b", calculation)
        if same_unit:
            final_value = float(same_unit[-1])
            if abs(final_value - expected) > max(1e-9, abs(expected) * 0.02):
                return f"calculation ends at {final_value:g} {unit}, not expected_value {expected:g} {unit}"
    return None


def synchronize_expected_value(verification: dict) -> None:
    """Make the sandboxed arithmetic expression authoritative over model mental arithmetic."""
    expression = str(verification.get("expression") or "").strip()
    if not expression:
        return
    try:
        calculated = evaluate_numeric_expression(expression)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return
    verification["expected_value"] = calculated
    answer = str(verification.get("expected_answer") or "")
    replacement = f"{calculated:.6g}"
    number_pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
    unicode_scientific = re.search(r"[-+]?\d+(?:\.\d+)?\s*[×x]\s*10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻+\-]*", answer)
    if unicode_scientific:
        verification["expected_answer"] = answer[:unicode_scientific.start()] + replacement + answer[unicode_scientific.end():]
    else:
        verification["expected_answer"] = re.sub(number_pattern, replacement, answer, count=1)


def abstract_structure(bundle: dict) -> dict:
    """Expose graph shape to the planner without leaking source-question semantics."""
    source = bundle.get("structure_exemplars") or []
    seed_text = str(bundle.get("seed_graph_text") or "")
    node_types = []
    edge_types = []
    for token in ("Given", "Law", "State", "Target", "Constraint", "Trap"):
        node_types.extend([token] * seed_text.count(f":{token}:"))
    for token in ("supports", "depends_on", "derived_from", "rules_out"):
        edge_types.extend([token] * seed_text.count(f"-{token}->"))
    return {
        "response_type": bundle.get("response_type"),
        "topics": bundle.get("topics"),
        "node_types": node_types,
        "edge_types": edge_types,
        "exemplar_complexities": [
            {"response_type": row.get("response_type"), "points": row.get("points"),
             "part_count": len(row.get("parts") or []) or 1,
             "difficulty": (row.get("analysis") or {}).get("difficulty")}
            for row in source
        ],
    }


def make_plan(bundle: dict, model: str, level: str, provider: str, novelty_corpus: list[dict] | None = None,
              difficulty_floor: int = 1, prior_plans: list[dict] | None = None,
              match_source_difficulty: bool = False) -> dict:
    source_format = bundle.get("response_type")
    anchor = bundle.get("anchor_item") or {}
    source_difficulty = int((anchor.get("analysis") or {}).get("difficulty") or difficulty_floor)
    target_difficulty = max(1, min(5, source_difficulty if match_source_difficulty else difficulty_floor))
    source_prompt_words = len(re.findall(r"\w+", str(anchor.get("prompt") or "")))
    required_format = source_format if source_format in {"multiple_choice", "multipart"} else "short_answer or numeric"
    anchor_parts = anchor.get("parts") or []
    part_profile = [{"label": str(part.get("label") or part.get("id") or chr(97 + index)),
                     "response_type": part.get("response_type"), "points": part.get("points", 1)}
                    for index, part in enumerate(anchor_parts)]
    used_concepts = {str(row.get("assigned_concept") or "").lower() for row in (prior_plans or [])}
    forbidden_text = " ".join(map(str, bundle.get("topics") or [])).lower()
    assigned_concept = None if match_source_difficulty else next((concept for concept in CONCEPT_MENU
                             if concept.lower() not in used_concepts and concept.lower() not in forbidden_text), None)
    if assigned_concept is None:
        assigned_concept = ("a novel question testing the same Circuit Lab concept as this source anchor, without "
                            "copying its wording, values, or exact task: " + str(anchor.get("prompt") or ""))
    verified_spec = None if match_source_difficulty else (VERIFIED_ANCHOR_SPECS.get(bundle.get("anchor_id")) or VERIFIED_CONCEPT_SPECS.get(assigned_concept))
    if verified_spec:
        arithmetic_error = verification_error(verified_spec)
        if arithmetic_error:
            raise RuntimeError(f"Invalid vetted specification for {assigned_concept}: {arithmetic_error}")
        givens = [{"id": f"g{i}", "type": "Given", "label": label}
                  for i, label in enumerate(verified_spec["fixed_givens"], 1)]
        law_id = "law1"
        target_id = "target1"
        graph = {"nodes": givens + [
            {"id": law_id, "type": "Law", "label": verified_spec["calculation"]},
            {"id": target_id, "type": "Target", "label": assigned_concept},
        ], "edges": ([{"src": row["id"], "dst": law_id, "type": "supports"} for row in givens]
                      + [{"src": law_id, "dst": target_id, "type": "derived_from"}])}
        response_type = "multiple_choice" if source_format == "multiple_choice" else "short_answer"
        topics = classify_topics(assigned_concept + " " + " ".join(verified_spec["fixed_givens"]))
        return {"topics": topics, "reasoning_graph": graph, "target_skill": assigned_concept,
                "difficulty": target_difficulty, "novel_context": assigned_concept,
                "response_type": response_type,
                "distractor_mechanisms": ["series/parallel confusion", "wrong direction or contact state",
                                          "arithmetic or unit-conversion error"],
                "verification": {**verified_spec, "vetted": True}, "bundle_id": bundle["bundle_id"],
                "assigned_concept": assigned_concept, "level": level, "target_prompt_words": source_prompt_words}
    prompt = f"""LEVEL: {level}
TOPICS: {json.dumps(bundle.get('topics') or [])}
FORBIDDEN SOURCE CONCEPTS: {json.dumps(bundle.get('topics') or [])}
REQUIRED DISTINCT CONCEPT: {assigned_concept}
MANDATORY VERIFIED SPECIFICATION (copy these exact givens into the graph; do not alter them):
{json.dumps(verified_spec, ensure_ascii=False)}
SOURCE FORMAT: {bundle.get('response_type')}
REQUIRED OUTPUT FORMAT: {required_format}
EXACT TARGET DIFFICULTY: {target_difficulty} on a 1-5 scale
SOURCE PROMPT LENGTH: {source_prompt_words} words
SOURCE MULTIPART PROFILE (match part count and approximate scoring/response mix, not content):
{json.dumps(part_profile, ensure_ascii=False)}
ABSTRACT STRUCTURAL PROFILE (contains no source scenario or target answer):
{json.dumps(abstract_structure(bundle), ensure_ascii=False)}

EXISTING SOURCE QUESTIONS (the new plan must not be equivalent to any of these):
{json.dumps([row.get('prompt') for row in (novelty_corpus or [])], ensure_ascii=False)}

CONCEPTS ALREADY PLANNED IN THIS BATCH (must not repeat):
{json.dumps([{"topics": row.get('topics'), "target_skill": row.get('target_skill'), "novel_context": row.get('novel_context')} for row in (prior_plans or [])], ensure_ascii=False)}

Build the plan specifically around REQUIRED DISTINCT CONCEPT. Do not substitute another concept.
Difficulty 1 may use recall or one direct inference. Difficulty 2 should require one calculation or a conceptual link.
Difficulty 3+ must require at least two meaningful reasoning or calculation steps. Any circuit topology must be described unambiguously using named nodes or
parentheses; never claim two components are in series if their shared node has another branch.
Return {{"topics":["chosen allowed Circuit Lab concept"],"reasoning_graph":{{"nodes":[{{"id":"n1","type":"Given|Law|State|Target|Constraint|Trap","label":"..."}}],
"edges":[{{"src":"n1","dst":"n2","type":"supports|depends_on|derived_from|rules_out"}}]}},
"target_skill":"specific Circuit Lab skill","difficulty":1,"novel_context":"electrical/magnetic context","response_type":"{required_format}",
"part_plans":[{{"label":"a","response_type":"numeric|short_answer|multiple_choice","points":1,"depends_on":[],"skill":"specific escalating subskill","expected_answer":"answer with units","calculation":"brief independent derivation"}}],
"distractor_mechanisms":["..."],"verification":{{"fixed_givens":["every exact value, sign, position, topology, or state needed"],
"expected_answer":"independently calculated result with unit/direction or full state answer","expected_value":12.3,
"expression":"numbers and + - * / ** parentheses only, evaluating to expected_value; omit both for nonnumeric tasks",
"calculation":"derivation under 100 words"}}}}.
The verification object is mandatory. Choose all numerical givens now, solve them, and ensure expected_answer follows from calculation; question writing may not alter them.
Difficulty must be exactly {target_difficulty}. If SOURCE FORMAT is multipart, response_type MUST remain multipart,
part_plans must match the source multipart profile's part count, and dependencies may reference only earlier labels.
Use one shared scenario and make the parts form a coherent progression rather than independent mini-questions.
If the source is multiple choice, response_type MUST remain multiple_choice.
Every Given/Law/Target node must refer to actual Circuit Lab knowledge. Do not write the final question."""
    retry = ""
    rejection = "unknown validation failure"
    bundle_seed = sum(ord(char) for char in str(bundle.get("bundle_id") or ""))
    for attempt in range(6):
        try:
            plan_token_cap = max(1800, 900 + 300 * len(part_profile))
            result = generate_json(model, prompt + retry, provider=provider, system=SYSTEM,
                                   temperature=0.3, max_output_tokens=plan_token_cap, seed=1700 + bundle_seed + attempt)
        except (RuntimeError, ValueError) as exc:
            rejection = f"malformed model response: {exc}"
            retry = "\nREJECTED: Return compact complete JSON. Keep verification.calculation under 100 words."
            continue
        if verified_spec:
            result["verification"] = verified_spec
        result["response_type"] = source_format if source_format in {"multiple_choice", "multipart"} else result.get("response_type", "numeric")
        result["difficulty"] = target_difficulty
        if result["response_type"] not in {"multiple_choice", "short_answer", "numeric", "multipart"}:
            result["response_type"] = "numeric" if source_format != "multiple_choice" else "multiple_choice"
        if source_format == "multipart":
            part_plans = result.get("part_plans") or []
            if len(part_plans) != len(part_profile):
                rejection = f"multipart plan requires exactly {len(part_profile)} part plans"
                retry = f"\nREJECTED: Return exactly {len(part_profile)} coherent part_plans matching the supplied profile."
                continue
        if not isinstance(result.get("difficulty"), int) or result["difficulty"] != target_difficulty:
            retry = f"\nREJECTED: difficulty must be exactly {target_difficulty}."
            rejection = retry.strip()
            continue
        verification = result.get("verification") or {}
        plan_dump = json.dumps(result, ensure_ascii=False).lower()
        expected_text = str(verification.get("expected_answer") or "").lower()
        if "magnitude" in expected_text and re.search(r"magnitude\D*-\d", expected_text):
            rejection = "a physical magnitude cannot be negative; express direction separately"
            retry = "\nREJECTED: Use a positive magnitude and state the direction separately."
            continue
        repair_artifacts = [phrase for phrase in PLAN_REPAIR_ARTIFACTS if phrase in plan_dump]
        if repair_artifacts:
            rejection = "unresolved plan-repair narration: " + ", ".join(repair_artifacts)
            retry = "\nREJECTED: Recalculate cleanly and return no correction or draft-repair narration."
            continue
        blocked_topics = [phrase for phrase in OUT_OF_SCOPE + UNRELATED_DOMAINS if phrase in plan_dump]
        if blocked_topics:
            rejection = "out-of-scope plan content: " + ", ".join(blocked_topics)
            retry = ("\nREJECTED PLAN:\n" + json.dumps(result, ensure_ascii=False)
                     + "\nRemove these out-of-scope topics entirely: " + ", ".join(blocked_topics))
            continue
        arithmetic_error = verification_error(verification)
        if arithmetic_error and match_source_difficulty:
            for key in ("expected_answer", "expected_value", "expression"):
                verification.pop(key, None)
            verification["calculation"] = "No precomputed answer is authoritative; solve independently from the fixed givens."
            arithmetic_error = None
        verified = bool(verification.get("calculation") and
                        (target_difficulty == 1 or verification.get("fixed_givens")) and
                        (verification.get("expected_answer") or match_source_difficulty))
        if not verified or arithmetic_error:
            issue = arithmetic_error or ("verification requires expected_answer and calculation; fixed_givens are also required "
                                         "for difficulty 2+")
            retry = ("\nREJECTED PLAN (revise rather than discuss it):\n" + json.dumps(result, ensure_ascii=False)
                     + "\nMACHINE VERIFICATION FAILURE: " + issue
                     + "\nFor numeric tasks, expression must be a bare Python-style arithmetic expression such as "
                       "12**2/(6+1/(1/12+1/8)), with no equals sign, variables, words, or units. Set expected_value to its result.")
            rejection = issue
            continue
        try:
            audit = generate_json(model, "PLAN_JSON:\n" + json.dumps(result, ensure_ascii=False), provider=provider,
                                  system=PLAN_JUDGE_SYSTEM, temperature=0.0, max_output_tokens=1200,
                                  seed=2700 + bundle_seed + attempt)
        except (RuntimeError, ValueError) as exc:
            rejection = f"malformed audit response: {exc}"
            retry = "\nREJECTED: Rebuild a concise, unambiguous plan; the prior audit response could not be parsed."
            continue
        audit_keys = ["self_contained", "unambiguous", "scope_compliant"]
        if verification.get("expected_answer"):
            audit_keys.append("verified_answer_correct")
        if (not all(audit.get(key) is True for key in audit_keys)
                or int(audit.get("reasoning_depth", 0)) < target_difficulty):
            issues = list(map(str, audit.get("issues") or ["insufficient reasoning depth or missing givens"]))
            retry = ("\nREJECTED PLAN (revise rather than discuss it):\n" + json.dumps(result, ensure_ascii=False)
                     + "\nREJECTED BY PLAN AUDIT: " + "; ".join(issues))
            rejection = "; ".join(issues)
            continue
        graph = result.get("reasoning_graph") or {}
        plan_text = " ".join(str(node.get("label") or "") for node in graph.get("nodes") or [] if isinstance(node, dict))
        plan_text += " " + str(result.get("target_skill") or "") + " " + str(result.get("novel_context") or "")
        lower_plan = plan_text.lower()
        if any(word in lower_plan for word in ("diagnos", "which resistor is open", "identify which resistor")):
            has_values = re.search(r"\br\d+\s*=\s*\d", lower_plan) is not None
            has_reference = any(word in lower_plan for word in ("healthy", "normal reading", "expected voltage", "before the fault"))
            if not (has_values or has_reference):
                retry = "\nREJECTED: fault diagnosis is underdetermined. Supply component values and explicit topology, or healthy expected readings for comparison."
                continue
        similarity, source_id = similarity_to_exemplars(plan_text, novelty_corpus or [])
        # Planning text necessarily shares technical vocabulary with the corpus.
        # Catch near-verbatim plans here; the final question receives the stricter final gate.
        if similarity < 0.85:
            chosen_topics = result.get("topics") or []
            if not chosen_topics:
                retry = "\nREJECTED: Choose a concrete allowed Circuit Lab topic."
                continue
            graph_labels = " ".join(str(node.get("label") or "") for node in graph.get("nodes") or [] if isinstance(node, dict))
            canonical_topics = classify_topics(" ".join(map(str, chosen_topics)) + " " + graph_labels)
            if level in {"invitational", "regional"} and "led" in canonical_topics:
                retry = "\nREJECTED: LED operation is not allowed at this tournament level. Choose another allowed concept."
                continue
            prior_rows = [{"id": row.get("bundle_id"), "prompt": str(row.get("target_skill") or "") + " " + str(row.get("novel_context") or "")} for row in (prior_plans or [])]
            prior_similarity, prior_id = similarity_to_exemplars(plan_text, prior_rows)
            if prior_similarity >= 0.75:
                retry = f"\nREJECTED: This repeats the skill/context from prior batch plan {prior_id} ({prior_similarity:.2f}). Choose a different device, law, and task."
                continue
            stored_concept = str(result.get("target_skill") or assigned_concept) if match_source_difficulty else assigned_concept
            return {**result, "bundle_id": bundle["bundle_id"], "assigned_concept": stored_concept,
                    "topics": canonical_topics, "level": level, "target_prompt_words": source_prompt_words}
        retry = f"\nREJECTED: This plan is too similar to existing source {source_id} ({similarity:.2f}). Change the tested phenomenon or task, not merely the wording."
    raise RuntimeError(f"Could not create a novel reasoning plan for {bundle['bundle_id']}: {rejection}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate novel Circuit Lab reasoning plans")
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--novelty-corpus", default="")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--level", choices=["invitational", "regional", "state", "national"], default="regional")
    parser.add_argument("--difficulty-floor", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--match-source-difficulty", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep valid plans already written and continue remaining bundles")
    parser.add_argument("--invalidate-bundles", default="", help="Comma-separated bundle IDs to force replanning")
    args = parser.parse_args()
    bundles = read_jsonl(args.bundles)
    novelty_corpus = read_jsonl(args.novelty_corpus) if args.novelty_corpus else []
    plans = read_jsonl(args.out) if args.resume and Path(args.out).exists() else []
    invalidated = {value.strip() for value in args.invalidate_bundles.split(",") if value.strip()}
    plans = [row for row in plans if row.get("bundle_id") not in invalidated]
    plans = [row for row in plans if not any(phrase in json.dumps(row, ensure_ascii=False).lower()
                                             for phrase in OUT_OF_SCOPE + UNRELATED_DOMAINS + PLAN_REPAIR_ARTIFACTS)]
    plans = [row for row in plans if not re.search(
        r"e[-+]\d+\s+(?:micro|milli|kilo|mega|μ)", str((row.get("verification") or {}).get("expected_answer") or ""), re.I)]
    plans = [row for row in plans if not ("magnitude" in str((row.get("verification") or {}).get("expected_answer") or "").lower()
             and re.search(r"magnitude\D*-\d", str((row.get("verification") or {}).get("expected_answer") or ""), re.I))]
    plans = [row for row in plans if verification_error(row.get("verification") or {}) is None]
    completed = {row.get("bundle_id") for row in plans}
    for bundle in bundles:
        if bundle.get("bundle_id") in completed:
            continue
        plans.append(make_plan(bundle, args.model, args.level, args.provider, novelty_corpus, args.difficulty_floor,
                               plans, args.match_source_difficulty))
        write_jsonl(args.out, plans)
    write_jsonl(args.out, plans)
    print(f"Wrote {len(plans)} reasoning plans")


if __name__ == "__main__":
    main()
