#!/usr/bin/env python3
"""
04_generate_skeletons.py  (STAGE-1 GRAPH GENERATION)

Generates one new typed latent problem graph per retrieval bundle.
Backward compatibility:
- output file still named like generated_skeletons.jsonl
- includes `skeleton_text` alias equal to `graph_text`
"""
from __future__ import annotations
import argparse, difflib, json, math, os, re, sys, time
from typing import Any, Dict, List
from dotenv import load_dotenv
from circuit_lab.model_client import generate_json
from difficulty_rubric import difficulty_instruction

load_dotenv()

NODE_TYPES = ["Given", "Target", "Law", "State", "Constraint", "Auxiliary", "Trap", "Distractor"]
EDGE_TYPES = ["supports", "depends_on", "derived_from", "couples", "rules_out", "produces_distractor"]
MAX_STEM_WORDS = 180
MAX_CHOICE_WORDS = 32
MAX_EXACT_ANSWER_CHARS = 180

BLIND_SOLVE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "issues": {"type": "array", "maxItems": 6,
                   "items": {"type": "string", "maxLength": 500}},
        "independent_derivation": {"type": "array", "items": {"type": "string"}},
        "independent_result": {"type": "string"},
        "matching_choice": {"type": "string"},
        "well_posed": {"type": "boolean"}, "physics_valid": {"type": "boolean"},
        "unique_answer": {"type": "boolean"}, "choice_match": {"type": "boolean"},
    },
    "required": [
        "verdict", "issues", "independent_derivation", "independent_result", "matching_choice",
        "well_posed", "physics_valid", "unique_answer", "choice_match",
    ],
}

DIFFICULTY_AUDIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "actual_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "conceptual_deductions": {"type": "array", "items": {"type": "string"}},
        "non_obvious_decisions": {"type": "array", "items": {"type": "string"}},
        "routine_shortcut_exists": {"type": "boolean"},
        "familiar_template": {"type": "boolean"},
        "architecture_steps_preserved": {"type": "boolean"},
        "missing_architecture_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "verdict", "issues", "actual_difficulty", "conceptual_deductions",
        "non_obvious_decisions", "routine_shortcut_exists", "familiar_template",
        "architecture_steps_preserved", "missing_architecture_steps",
    ],
}

DIFFICULTY_ARCHITECT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "candidate_mechanisms": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "mechanism": {"type": "string"},
                    "standard_laws": {
                        "type": "array", "minItems": 2, "items": {"type": "string"},
                    },
                    "necessary_deductions": {
                        "type": "array", "minItems": 4, "items": {"type": "string"},
                    },
                    "non_obvious_decisions": {
                        "type": "array", "minItems": 1, "items": {"type": "string"},
                    },
                    "closure_requirements": {
                        "type": "array", "minItems": 2, "items": {"type": "string"},
                    },
                    "shortcut_risk": {"type": "string"},
                },
                "required": [
                    "mechanism", "standard_laws", "necessary_deductions",
                    "non_obvious_decisions", "closure_requirements", "shortcut_risk",
                ],
            },
        },
        "selected_mechanism": {"type": "string"},
        "selection_reason": {"type": "string"},
        "solvability_check": {
            "type": "array", "minItems": 3, "items": {"type": "string"},
        },
        "mandatory_solution_structure": {
            "type": "array", "minItems": 4, "items": {"type": "string"},
        },
        "forbidden_shortcuts": {
            "type": "array", "minItems": 2, "items": {"type": "string"},
        },
        "reference_depth_mapping": {
            "type": "array", "minItems": 3, "maxItems": 6,
            "items": {"type": "string", "maxLength": 500},
        },
    },
    "required": [
        "candidate_mechanisms", "selected_mechanism", "selection_reason", "solvability_check",
        "mandatory_solution_structure", "forbidden_shortcuts", "reference_depth_mapping",
    ],
}

ARCHITECTURE_ADHERENCE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "actual_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "physics_closed": {"type": "boolean"},
        "routine_shortcut_exists": {"type": "boolean"},
        "familiar_template": {"type": "boolean"},
        "single_scalar_or_categorical_target": {"type": "boolean"},
        "concise_student_render_feasible": {"type": "boolean"},
        "derived_intermediate_leaked_as_given": {"type": "boolean"},
        "indispensable_decision": {"type": "string", "maxLength": 500},
        "counterfactual_without_decision": {"type": "string", "maxLength": 500},
        "shortest_solution_outline": {
            "type": "array", "minItems": 4, "maxItems": 7,
            "items": {"type": "string", "maxLength": 450},
        },
        "mandatory_step_checks": {
            "type": "array", "minItems": 4, "maxItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "architecture_step": {"type": "string", "maxLength": 350},
                    "construction_evidence": {"type": "string", "maxLength": 500},
                    "indispensable": {"type": "boolean"},
                },
                "required": ["architecture_step", "construction_evidence", "indispensable"],
            },
        },
    },
    "required": [
        "verdict", "issues", "actual_difficulty", "physics_closed",
        "routine_shortcut_exists", "familiar_template",
        "single_scalar_or_categorical_target", "concise_student_render_feasible",
        "derived_intermediate_leaked_as_given",
        "indispensable_decision", "counterfactual_without_decision",
        "shortest_solution_outline", "mandatory_step_checks",
    ],
}

ARCHITECTURE_VALIDITY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "issues": {"type": "array", "maxItems": 6,
                   "items": {"type": "string", "maxLength": 500}},
        "physics_closed": {"type": "boolean"},
        "syllabus_safe": {"type": "boolean"},
        "projected_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "routine_shortcut_exists": {"type": "boolean"},
        "single_scalar_or_categorical_target": {"type": "boolean"},
        "concise_student_render_feasible": {"type": "boolean"},
        "indispensable_decision": {"type": "string", "maxLength": 500},
        "shortest_solution_outline": {
            "type": "array", "minItems": 4, "maxItems": 7,
            "items": {"type": "string", "maxLength": 450},
        },
        "unknown_equation_check": {"type": "array", "minItems": 3, "maxItems": 8,
                                   "items": {"type": "string", "maxLength": 450}},
    },
    "required": [
        "verdict", "issues", "physics_closed", "syllabus_safe", "projected_difficulty",
        "routine_shortcut_exists", "single_scalar_or_categorical_target",
        "concise_student_render_feasible", "indispensable_decision",
        "shortest_solution_outline", "unknown_equation_check",
    ],
}

SOLUTION_CONSTRUCTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "setup": {"type": "string", "minLength": 1, "maxLength": 1400},
        "target": {"type": "string", "minLength": 1, "maxLength": 300},
        "fixed_givens": {"type": "array", "minItems": 4,
                          "maxItems": 10,
                          "items": {"type": "string", "minLength": 1, "maxLength": 300}},
        "governing_equations": {"type": "array", "minItems": 3,
                                "maxItems": 10,
                                "items": {"type": "string", "minLength": 1, "maxLength": 400}},
        "regime_or_constraint_decisions": {
            "type": "array", "minItems": 1, "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 400},
        },
        "derived_intermediates": {
            "type": "array", "minItems": 2, "maxItems": 6,
            "items": {"type": "string", "minLength": 1, "maxLength": 400},
        },
        "preserved_seed_dependencies": {
            "type": "array", "minItems": 3, "maxItems": 6,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "architecture_trace": {
            "type": "array", "minItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "architecture_step": {"type": "string", "minLength": 1, "maxLength": 350},
                    "implemented_by": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["architecture_step", "implemented_by"],
            },
        },
        "derivation_steps": {"type": "array", "minItems": 5, "maxItems": 10,
                             "items": {"type": "string", "minLength": 1, "maxLength": 650}},
        "exact_answer": {"type": "string", "minLength": 1,
                         "maxLength": MAX_EXACT_ANSWER_CHARS},
        "independent_checks": {"type": "array", "minItems": 2, "maxItems": 5,
                               "items": {"type": "string", "minLength": 1, "maxLength": 450}},
        "closure_proof": {"type": "array", "minItems": 3, "maxItems": 8,
                          "items": {"type": "string", "minLength": 1, "maxLength": 400}},
    },
    "required": [
        "setup", "target", "fixed_givens", "governing_equations",
        "regime_or_constraint_decisions", "derived_intermediates",
        "preserved_seed_dependencies", "architecture_trace", "derivation_steps",
        "exact_answer", "independent_checks", "closure_proof",
    ],
}


def generation_schema(target_difficulty: int, choice_count: int) -> Dict[str, Any]:
    """Strict contract for the solution-first graph/blueprint generation call."""
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "problem_graph": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "nodes": {
                        "type": "array", "minItems": 8, "maxItems": 16,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string", "enum": NODE_TYPES},
                                "label": {"type": "string"},
                                "importance": {
                                    "type": "string",
                                    "enum": ["primary", "secondary", "optional"],
                                },
                            },
                            "required": ["id", "type", "label", "importance"],
                        },
                    },
                    "edges": {
                        "type": "array", "minItems": 2, "maxItems": 20,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "src": {"type": "string"}, "dst": {"type": "string"},
                                "type": {"type": "string", "enum": EDGE_TYPES},
                                "note": {"type": "string"},
                            },
                            "required": ["src", "dst", "type", "note"],
                        },
                    },
                },
                "required": ["nodes", "edges"],
            },
            "topic": {"type": ["string", "null"]},
            "difficulty": {"type": "integer", "const": target_difficulty},
            "solution_blueprint": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "canonical_problem": {"type": "string", "maxLength": 1800},
                    "canonical_choices": {
                        "type": "array", "minItems": choice_count, "maxItems": choice_count,
                        "items": {"type": "string", "maxLength": 240},
                    },
                    "novel_setup": {"type": "string"},
                    "fixed_givens": string_array,
                    "interacting_concepts": string_array,
                    "key_insight": {"type": "string"},
                    "derivation_steps": {
                        "type": "array", "minItems": target_difficulty, "maxItems": 12,
                        "items": {"type": "string"},
                    },
                    "routine_shortcut_test": {"type": "string"},
                    "difficulty_justification": {"type": "string"},
                    "case_or_constraint_handling": {"type": "string"},
                    "verification_method": {"type": "string"},
                    "verification_steps": {
                        "type": "array", "minItems": 2, "maxItems": 6,
                        "items": {"type": "string"},
                    },
                    "equation_closure_check": {
                        "type": "array", "minItems": 3, "maxItems": 8,
                        "items": {"type": "string"},
                    },
                    "expected_answer": {"type": "string", "maxLength": 240},
                    "correct_choice": {"type": "string", "maxLength": 240},
                    "distractor_choices": {
                        "type": "array", "minItems": choice_count - 1,
                        "maxItems": choice_count - 1,
                        "items": {"type": "string", "maxLength": 240},
                    },
                },
                "required": [
                    "canonical_problem", "canonical_choices", "novel_setup", "fixed_givens",
                    "interacting_concepts", "key_insight", "derivation_steps",
                    "routine_shortcut_test", "difficulty_justification",
                    "case_or_constraint_handling", "verification_method", "verification_steps",
                    "equation_closure_check",
                    "expected_answer", "correct_choice", "distractor_choices",
                ],
            },
        },
        "required": ["problem_graph", "topic", "difficulty", "solution_blueprint"],
    }

SYSTEM = """You generate solution-first STRUCTURED typed latent problem plans for Olympiad-style STEM problems.

Hard constraints:
- Do NOT copy any exemplar graph verbatim.
- Keep the graph abstract, then provide a concrete solved blueprint with internally consistent values.
- The graph MUST be Olympiad-like and nontrivial, not one-equation.
- Include at least 1 Target node, at least 2 Law/Constraint nodes total, at least 2 State nodes.
- Include at least one genuine nontrivial feature: a hidden state, regime/case decision,
  invariant, coupled constraint, or tempting approach that must be rejected.
- Calculate the solved blueprint before any question wording is written.
- The setup must determine exactly ONE requested result. Reject any setup with a free
  parameter or multiple physically valid answers.
- Do not pick, guess, tune, or assume an unstated intermediate value. Every numerical step
  must follow from the fixed givens and stated constraints.
- Stay within standard competition classical mechanics: point particles, rigid bodies, ideal strings,
  ideal springs, gravity, friction, and explicitly defined impulses or collisions. Do not invent material
  laws or use deformable media, robotics, granular flow, phase changes, viscoelasticity, distributed-wave
  propagation, variable mass, or velocity-dependent restitution as a substitute for reasoning depth.
- In multi-collision mechanics, specify enough initial geometry to determine collision order and
  explicitly verify that order from collision times or relative motion. A collision-free final state
  requires spatially ordered velocities, not generally equal velocities; never replace the actual
  elastic dynamics with an inelastic common-velocity assumption.
- Keep the solution blueprint concise: at most 12 derivation steps.
- Keep the graph compact: 8-16 nodes and at most 20 edges. Keep the full response below
  roughly 3,500 tokens; never use the derivation as a scratchpad or narrate corrections.
- At difficulty 3, require at least three indispensable conceptual deductions and one modeling or
  constraint choice that changes the equations. Reject a single work-energy balance with friction,
  ordinary spring compression, direct slip-to-roll, or a routine collision followed by energy; extra
  algebra, a quadratic, and an after-the-fact range check do not make those templates difficulty 3.
  Before returning JSON, compare at least three distinct candidate mechanisms and discard every
  candidate that reduces to one familiar conservation chain. Do not reveal that comparison.
- At difficulty 4 or 5, do not use an unchanged textbook archetype (ordinary incline rolling,
  wheel-over-curb torque balance, standard Atwood/pulley algebra, direct collision conservation,
  or a single transition-to-rolling calculation). A genuinely new regime, dependency, or case
  structure must materially change the shortest solution.
- For difficulty 4 or 5, internally compare at least three genuinely different candidate mechanisms
  before returning JSON. Discard candidates whose shortest solve is just a familiar conservation-law
  sequence, repeated use of one formula, or algebra after an obvious setup. Return only a candidate
  whose non-obvious modeling decision is indispensable and changes the governing equations.
Return strict JSON only.
"""

SYSTEM_SEED_FAITHFUL = SYSTEM + """

For this hard-item run, generate a PARALLEL FORM of the primary seed instead of inventing a new
difficulty architecture. Preserve the seed's shortest-solution dependency chain, including every hidden
intermediate, coupling, regime decision, and reason a naive approach fails. Do not simplify that chain by
supplying an intermediate result as a given. Change the physical objects, narrative, variable names,
numbers, and visible presentation so the relationship is not apparent to a student. Internally consider
multiple surface realizations and return the clearest one. The final problem must still ask for one compact
result and use concise authentic F=ma choices.

Treat the seed's kinematics and event schedule as locked physics, not as surface details. Preserve, one for
one, its directions of motion, impact location/height, reference frame, temporal scope of each stated
velocity, and recurrence condition. In particular, a body that has a specified velocity only at each
impact must NOT be changed into a body moving at that velocity throughout the flight; an oscillating or
returning boundary must remain oscillating or returning. Before returning JSON, write signed pre-impact,
boundary, relative, and post-impact velocities privately and verify that a nonzero flight returns to the
stated impact geometry. Cosmetic originality never permits changing these invariants.

The retrieved seed graph is a fallible semantic index, not an answer key. Independently solve the
official seed question before transferring its dependency chain. If a graph equation conflicts with
that solution, discard the equation. Use one explicit signed coordinate convention throughout; never
substitute a positive speed magnitude where a signed velocity is required.
"""

SYSTEM_BLIND_SOLVER = """You are a skeptical contest solution checker. You receive only a proposed
student-facing problem and its answer choices, never the author's solution or claimed answer. Solve it
independently from first principles. Reject ambiguous, impossible, underdetermined, or internally
inconsistent setups. Check signs, geometry, domains, limiting cases, units, and every active constraint.
In independent_derivation, explicitly name every necessary model, regime, system-boundary, or pivot
selection before applying it; do not silently compress a genuine conceptual decision into algebra.
For an extremum, verify that the stated constraint actually determines the extremum. Do not choose the
nearest option: an option must match the independently obtained result except for explicitly stated
ordinary rounding. Return strict JSON only.
"""

SYSTEM_DIFFICULTY_AUDIT = """You are a severe contest difficulty calibrator. Judge only the shortest
correct solution supplied to you, not the author's claimed difficulty, prose length, number of listed
steps, arithmetic volume, or graph size. Compare it with the authentic competition examples provided.
For D4/D5, also verify that every locked mandatory architecture step remains indispensable in that
shortest solution; a missing or optional step is an automatic failure regardless of the claimed rating.
Treat restated givens, explicitly signposted formula selection, routine integration/algebra, numerical
substitution, and verification as zero conceptual deductions. An unchanged familiar textbook template
cannot exceed difficulty 3. Difficulty 4 needs a necessary non-obvious modeling decision and several
dependent conceptual deductions. Difficulty 5 needs at least two necessary non-obvious decisions, such
as a genuine regime/case choice plus a separate synthesis. Return strict JSON only.
"""

SYSTEM_CHOICE_REPAIR = """You repair answer choices for an otherwise valid competition problem.
The problem stem is frozen and must not be edited. The independent solver's result is authoritative.
Return exactly five concise choices containing that result exactly once, plus four plausible distractors
caused by specific mistakes in the supplied independent derivation. Do not redo or question the physics,
change the target, or alter any given. Return strict JSON only."""

SYSTEM_DIFFICULTY_ARCHITECT = """You are a contest solution-structure architect. Design reasoning
mechanisms, not question prose and not numerical data. Compare exactly three substantially different
mechanisms, then select one whose shortest correct solution truly meets the requested difficulty.
The requested problem must be conceptually hard but student-facing compact. Commit decisively to a
qualifying mechanism: do not weaken it into an easier textbook template merely because it is difficult.
Evaluate each candidate exactly once, select the strongest valid candidate exactly once, and lock that
selection. Do not reopen the comparison, narrate indecision, or revise a locked selection later.
Reject repeated standard formulas, routine conservation-law chains, algebraic length, and cosmetic
complications. The selected mechanism must contain an indispensable non-obvious modeling decision that
changes the governing equations. Use only standard high-school competition mechanics. Difficulty must
come from discovering and coupling ordinary laws, not from exotic objects or an unstated force/material
model. It must end in exactly one scalar or categorical target. Do not manufacture difficulty by asking
for a state vector, several impulses or velocities, a proof plus a value, or separately graded subparts.
When counting conceptual deductions, do not count restating a given, writing an explicitly signposted
formula, routine integration or algebra, numerical substitution, or verification. Those may support the
solution but cannot be used to claim D4 depth.
mandatory_solution_structure must contain only the indispensable conceptual path to the answer. Keep
validation, limiting cases, sign checks, equation counting, and restated setup conditions in
solvability_check or forbidden_shortcuts; never use them to fill the mandatory-step minimum.
Reject any mechanism unless every unknown is closed by an explicit standard law or a given relation.
Return only the requested strict JSON."""

SYSTEM_SOLUTION_CONSTRUCTOR = """You construct and solve a competition mechanics problem before any
student-facing wording or answer choices exist. Implement every mandatory architecture step. Use only
standard F=ma mechanics and explicit fixed givens. Derive one exact answer from first principles; never
choose numbers to retrofit an answer. Prove the equations close all unknowns and perform at least two
independent checks. The private solution may be sophisticated, but the finished problem must ask for
exactly one compact scalar or categorical result. Prefer fixed numerical givens yielding a short exact
number, simple fraction, or simple radical; do not return a tuple or list of post-event state variables.
Once the architecture is physically valid,
commit to it and carry the difficult derivation through without second-guessing it, simplifying away a
mandatory dependency, or retreating to a routine textbook problem. If the architecture cannot be made physically complete, choose a new standard
instantiation within the same mechanism family. Return strict JSON only. Keep the response under 3,500
tokens and never include scratchpad exploration, discarded derivations, alternatives, or self-corrections.
Use compact mathematical sentences: no field may restate the whole problem or repeat another field.
For loss of contact on a curved surface, explicitly state which side of the surface the body occupies,
derive the signed radial force equation from a declared inward direction, and verify the computed release
point is reached while N >= 0 beforehand. A bead constrained to a two-sided wire cannot leave at N=0.
For a later rigid-body contact, recompute the rotated body-fixed contact vector at the actual event time
before evaluating contact-point velocity. Use an explicitly smooth contact face and declared normal;
never assign an arbitrary normal to a sharp corner."""

SYSTEM_CONSTRUCTION_RENDERER = """You are a lossless renderer for an already solved competition
problem. The supplied construction is authoritative. Preserve its physical objects, geometry, givens,
event sequence, equations, exact answer, and mandatory deductions exactly; do not substitute a new
scenario, simplify away a stage, or re-solve it into a different problem. Your only creative task is to
write concise student-facing wording and plausible answer-choice distractors. Ask for one result only.
Keep the stem at most 180 words and every choice at most 32 words. Choices must look like authentic exam
choices: never explain, annotate, or label the mistake behind a distractor. Each distractor must come
privately from a stated error in the frozen derivation. Return strict JSON only."""

SYSTEM_ARCHITECTURE_VALIDITY = """You are a skeptical mechanics architect reviewing a proposed
solution mechanism before concrete values and final wording exist. Judge prospective closure: assume the
constructor will turn every item explicitly listed under closure_requirements into a fixed given and will
choose nondegenerate values satisfying stated inequalities. Do not reject merely because numerical masses,
positions, normals, times, or the final requested scalar have not yet been chosen. Reject only an inherent
structural defect: an unlisted degree of freedom, a missing law that cannot be supplied within the selected
mechanism, incompatible impact/contact assumptions, a familiar textbook chain, or difficulty based only
on length. Reject state-vector targets, multi-part targets, and mechanisms whose apparent depth vanishes
when only the single requested result is solved. A PASS requires an empty issues list. Be concise and
return strict JSON only."""

SYSTEM_ARCHITECTURE_ADHERENCE = """You are a severe pre-publication mechanics editor. Compare the
approved solution architecture with the concrete solved construction. PASS only if every mandatory
architecture step is implemented by a specific equation or deduction and remains indispensable in the
shortest correct solution. Reject decorative stages, parameters that trivialize a step, familiar-template
shortcuts, vague placeholders, unsupported impact/contact laws, underdetermined impulses, and incorrect
physics. Reconstruct the shortest solution independently of architecture_trace. Identify the one modeling
decision that changes the governing equations and explain what fails without it. Reject multiple requested
outputs and any construction that cannot become a concise F=ma-style stem with short choices. A PASS
verdict requires an empty issues list; any known defect, truncation, placeholder, or
unverified physical claim requires FAIL. Judge the construction itself, not its claimed difficulty or
prose length. Return strict JSON only."""

USER_TMPL = """Domain: {domain}
REQUIRED TOPIC: {required_topic}
{difficulty_contract}

SEED_GRAPH_TEXT (style prior; do not copy; be at least as complex):
{seed_graph}

SEED_QUESTION_TEXT (source audit context; do not copy):
{seed_question}

SOLUTION EXEMPLARS (do not copy):
{solution_exemplars_block}

Allowed node types:
{node_types_block}

Allowed edge types:
{edge_types_block}

TASK:
Generate ONE NEW typed latent problem graph that is contest-faithful and nontrivial.

Return strict JSON with keys:
- problem_graph: object with keys:
  - nodes: list of objects {{id: string, type: string, label: string, importance: "primary"|"secondary"|"optional"}}
  - edges: list of objects {{src: string, dst: string, type: string, note: string}}
- topic: short string or null
- difficulty: integer equal to the required finished-item difficulty
- solution_blueprint: object with canonical_problem, canonical_choices, novel_setup, fixed_givens, interacting_concepts,
  key_insight, derivation_steps, routine_shortcut_test, difficulty_justification,
  case_or_constraint_handling, verification_method, verification_steps,
  equation_closure_check, expected_answer, correct_choice, and {distractor_count} distractor_choices

The blueprint must primarily test REQUIRED TOPIC. It must ask for exactly one scalar or categorical result,
be uniquely solvable from the stated givens, and contain no guessed or freely chosen values.
For numerical problems, use exact values and units and calculate the expected answer independently. Make
correct_choice equal that result. Each distractor must come from a specific plausible error.
For difficulty 3+, interacting_concepts must contain at least two concepts that materially
depend on one another, the key_insight must be necessary, and the shortest solve must contain at
least three meaningful conceptual deductions plus a modeling or constraint choice that changes the
equations. For difficulty 4+, the key_insight must be necessary, not decorative,
and the shortest valid solve must have at least four meaningful deductions. Difficulty 5 must
also require case/constraint handling or a second non-obvious synthesis. routine_shortcut_test
must explicitly explain why no one-formula or direct-recall solution works; if one does, reject
the plan and start over. Include a final substitution/check in derivation_steps proving the result satisfies every
constraint. The later question-writing stage will treat this blueprint as authoritative.
canonical_problem must be a complete, self-contained student-facing stem containing every
given and condition, use at most 180 words, and contain no separately graded subparts. canonical_choices
must contain exactly {choice_count} concise result texts of at most 32 words each, one equal to
correct_choice. Never put comments such as "sign error," "incorrect," or "missing factor" in a choice.
Keep the entire JSON response concise and each derivation step to one sentence.
After the main derivation, verify the result by a genuinely separate route when possible; otherwise
use a complete dimensional, limiting-case, sign/domain, and constraint check. verification_steps
must expose this check explicitly. Never create choices until both calculations agree.
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def llm_json(model: str, system: str, user: str, provider: str, temperature: float=0.25,
             reasoning_effort: str | None = None,
             json_schema: Dict[str, Any] | None = None,
             token_budget_override: int | None = None,
             ollama_think: bool | None = None) -> Dict[str, Any]:
    supports_reasoning = provider == "openai" and model.startswith(
        ("gpt-5", "gpt-oss", "o1", "o3", "o4")
    )
    effective_effort = (reasoning_effort or "low") if supports_reasoning else None
    default_budget = 12000 if provider == "ollama" else 5000
    token_budget = token_budget_override or {"low": 12000, "medium": 24000, "high": 32000}.get(
        effective_effort, default_budget
    )
    return generate_json(model, user, provider=provider, system=system, temperature=temperature,
                         max_output_tokens=token_budget,
                         reasoning_effort=effective_effort, json_schema=json_schema,
                         ollama_think=ollama_think)


def audit_json_with_fallback(
    model: str, system: str, user: str, provider: str, schema: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    # GPT-OSS spends most of FreeToken's response allowance on hidden reasoning at
    # medium effort, often leaving no room for the required JSON.  The audit prompt
    # already contains the complete solved artifact, so low effort is sufficient.
    efforts = (("low",) if model.startswith("gpt-oss") else ("medium", "low")) \
        if provider == "openai" else (None,)
    for effort in efforts:
        try:
            return llm_json(
                model, system, user, provider, temperature=0.0,
                reasoning_effort=effort, json_schema=schema,
            )
        except (ValueError, RuntimeError) as exc:
            errors.append(f"{effort}: {exc}")
    raise RuntimeError("; ".join(errors))


def normalize_graph(graph: Any) -> Dict[str, Any]:
    if not isinstance(graph, dict):
        return {"nodes": [], "edges": []}
    nodes=[]; seen=set()
    for idx, n in enumerate(graph.get("nodes") or [], start=1):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or f"n{idx}").strip()[:40]
        if not nid or nid in seen:
            nid = f"n{idx}"
        seen.add(nid)
        ntype = str(n.get("type") or "State").strip()
        if ntype not in NODE_TYPES:
            ntype = "State"
        label = " ".join(str(n.get("label") or "").split())[:160]
        if not label:
            continue
        importance = str(n.get("importance") or "primary").strip().lower()
        if importance not in {"primary", "secondary", "optional"}:
            importance = "primary"
        nodes.append({"id": nid, "type": ntype, "label": label, "importance": importance})
    valid = {n["id"] for n in nodes}
    edges=[]
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src = str(e.get("src") or "").strip()[:40]
        dst = str(e.get("dst") or "").strip()[:40]
        etype = str(e.get("type") or "supports").strip()
        if src not in valid or dst not in valid:
            continue
        if etype not in EDGE_TYPES:
            etype = "supports"
        row={"src": src, "dst": dst, "type": etype}
        note = " ".join(str(e.get("note") or "").split())[:160]
        if note:
            row["note"] = note
        edges.append(row)
    return {"nodes": nodes, "edges": edges}


def graph_text(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    groups = {t: [] for t in NODE_TYPES}
    labels = {}
    for n in nodes:
        groups[n["type"]].append(f"{n['id']}:{n['label']}")
        labels[n["id"]] = n["label"]
    parts=[]
    for t in NODE_TYPES:
        if groups[t]:
            parts.append(f"{t.upper()}S: " + " | ".join(groups[t]))
    if edges:
        parts.append("EDGES: " + " | ".join(
            f"{e['src']}[{labels.get(e['src'], e['src'])}] -{e['type']}-> {e['dst']}[{labels.get(e['dst'], e['dst'])}]" + (f" ({e['note']})" if e.get("note") else "")
            for e in edges
        ))
    return "\n".join(parts).strip()


def abstract_solution_profile(text: str) -> str:
    """Retain dependency semantics while omitting literal givens and source prose."""
    fields: Dict[str, str] = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {
            "TARGET_SUMMARY", "STATE_EVOLUTION", "HIDDEN_STATES", "COUPLING_POINTS",
            "INSIGHT_TYPE", "TRAP_PROFILE", "WHY_NAIVE_FAILS", "REASONING_DEPTH",
        }:
            fields[key] = value.strip()
        elif key == "GRAPH_METRICS":
            fields[key] = value.strip()
    if not fields:
        return "(abstract profile unavailable)"
    return " | ".join(f"{key}={value}" for key, value in fields.items())


def seed_faithful_profile(text: str) -> str:
    """Keep retrieval concepts but never feed unverified seed equations to the generator."""
    allowed = {
        "TARGET_SUMMARY", "COUPLING_POINTS", "INSIGHT_TYPE", "TRAP_PROFILE",
        "WHY_NAIVE_FAILS", "REASONING_DEPTH",
    }
    fields: Dict[str, str] = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in allowed:
            fields[key] = value.strip()
    if not fields:
        return "(semantic seed profile unavailable; independently solve the official seed question)"
    return " | ".join(f"{key}={value}" for key, value in fields.items())


def bounded_mechanism_library(required_topic: str) -> str:
    topic = str(required_topic or "").casefold()
    # Check rotation before momentum: the rotation topic description includes the
    # phrase "angular momentum" and was previously misrouted into collision-only
    # mechanism families.
    if any(word in topic for word in ("rotation", "torque", "rolling", "rigid-body")):
        return """Choose three distinct, syllabus-safe rotational depth families:
1. ROLLING REGIME WITH A NECESSARY TRANSITION: determine whether/when slipping changes to rolling,
   or whether contact is maintained, then propagate the resulting translational and rotational state
   into a separate requested event. The regime test must change the equations, not merely add a check.
   The speed or time at which ordinary slip first becomes rolling is forbidden as the final target.
2. CONSTRAINED PLANAR RIGID-BODY KINEMATICS: combine center-of-mass motion with a rotating body-fixed
   vector and one holonomic contact/geometry constraint. Use ordinary rods, disks, or linked rigid
   bodies and choose givens that make the event equation explicit and uniquely solvable.
3. TORQUE/ANGULAR-MOMENTUM SYNTHESIS: use a non-obvious pivot or system boundary to determine an
   intermediate angular state, then combine it with energy, linear momentum, or a stability/contact
   condition to obtain the requested scalar. Every stage must remain necessary in the shortest solve.
   Contact reactions, load distribution, friction forces, and intermediate angular states must be
   derived from primitive geometry and constraints, never supplied as numerical givens.
Prefer the same depth scale as the retrieved official F=ma structures. Avoid multi-impact machinery
unless a retrieved structure actually calls for it; difficulty should come from one hidden modeling
choice and a compact chain of dependable equations, not from stacking fragile collision events."""
    if any(word in topic for word in ("momentum", "impulse", "collision")):
        return """Choose exactly one of these syllabus-safe depth families; the three candidates must be
distinct instantiations of these three families, not invented alternatives:
1. OFF-CENTER RIGID-BODY IMPULSE: use linear impulse, angular impulse about the center of mass,
   and restitution at the actual contact point; the indispensable decision is that contact-point
   velocity contains both translation and rotation. Add one subsequent standard-mechanics event
   whose initial state depends on both post-impact motions. Asking only for the immediate angular
   velocity or impulse is forbidden.
   For a later contact, rotate the body-fixed contact vector into its event-time orientation before
   evaluating v_contact = v_CM + omega cross r. Use a smooth face with an explicitly defined normal;
   a point impact at a sharp corner is forbidden because its contact normal is not unique.
2. IMPACT-TO-CONSTRAINT TRANSITION: determine an impulsive post-impact state using momentum or
   angular momentum about the correct point, evolve it on a FIXED smooth track or fixed ideal string
   using energy, and locate a release/slack/contact
   transition from a force constraint such as T=0 or N=0 before computing the requested final result.
   The motion must continue after release and the target must depend on that post-release motion.
   The guide itself must not translate or rotate; do not introduce a second generalized coordinate.
   State whether the body is on the inside or outside of a one-sided track and derive the signed radial
   force balance; do not use a two-sided bead-on-wire constraint if the body is supposed to release.
   The contact force points from the surface into the body: it is radially outward on the outside of a
   convex circle and radially inward on the inside of a concave circle. Check this direction before
   writing the signed radial equation.
   A collision followed only by spring compression, maximum height, or ordinary pendulum swing is
   forbidden. A safe symbolic pattern is: impact fixes speed on a vertical curved constraint; energy
   gives speed versus position; N=0 or T=0 fixes the release point; tangent velocity initializes a
   projectile; the requested landing quantity requires solving that projectile motion.
   At D4/D5, that unchanged circular-track-to-projectile pattern is itself forbidden as a familiar
   textbook chain; an additional indispensable regime decision must alter the governing equations,
   not merely append another calculation.
3. GEOMETRY-DETERMINED MULTI-STAGE INTERACTION: use ordinary ideal collisions or impulses, but infer
   which event occurs next from positions and relative velocities; update the state and use a separate
   invariant or center-of-mass relation to obtain the final requested quantity. At least three events
   must be possible, and collision order must be proved using times or closing speeds rather than stated.
Every body is a point mass or rigid body and every interaction law is standard and explicit."""
    return """Use only point particles, rigid bodies, ideal strings/springs, gravity, friction, and
explicit standard constraints. Build depth from a hidden regime/constraint decision followed by at
least four dependent deductions; never invent a constitutive law or material model."""


def graph_ok(graph: Dict[str, Any], difficulty: int = 3) -> bool:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    counts = {t: 0 for t in NODE_TYPES}
    for n in nodes:
        if isinstance(n, dict) and n.get("type") in counts:
            counts[n["type"]] += 1
    minimum_edges = {1: 2, 2: 3, 3: 5, 4: 6, 5: 7}[difficulty]
    minimum_laws = {1: 1, 2: 2, 3: 3, 4: 3, 5: 4}[difficulty]
    return (
        counts["Target"] >= 1 and
        (counts["Law"] + counts["Constraint"]) >= minimum_laws and
        counts["State"] >= (1 if difficulty == 1 else 2) and
        (difficulty < 3 or counts["Auxiliary"] >= 1 or counts["Trap"] >= 1 or counts["Constraint"] >= 2) and
        len(edges) >= minimum_edges
    )


def graph_from_construction(construction: Dict[str, Any]) -> Dict[str, Any]:
    """Build reliable latent metadata without asking the renderer to reinvent the solved problem."""
    equations = [str(item) for item in (construction.get("governing_equations") or []) if str(item).strip()]
    decisions = [str(item) for item in (construction.get("regime_or_constraint_decisions") or []) if str(item).strip()]
    nodes = [
        {"id": "given", "type": "Given", "label": str(construction.get("setup") or "Fixed givens")[:160], "importance": "primary"},
        {"id": "initial", "type": "State", "label": "Initial state fixed by the stated geometry and data", "importance": "primary"},
        {"id": "intermediate", "type": "State", "label": "Intermediate state produced by the coupled governing equations", "importance": "primary"},
        {"id": "constraint", "type": "Constraint", "label": (decisions[0] if decisions else "Active regime and sign convention")[:160], "importance": "primary"},
        {"id": "aux", "type": "Auxiliary", "label": "Solve the intermediate state before evaluating the requested result", "importance": "primary"},
        {"id": "target", "type": "Target", "label": str(construction.get("target") or "Requested result")[:160], "importance": "primary"},
    ]
    for idx, equation in enumerate(equations[:5], start=1):
        nodes.append({"id": f"law{idx}", "type": "Law", "label": equation[:160], "importance": "primary"})
    edges = [
        {"src": "given", "dst": "initial", "type": "supports", "note": "sets the initial state"},
        {"src": "initial", "dst": "law1", "type": "supports", "note": "supplies initial data"},
        {"src": "constraint", "dst": "law1", "type": "constrains", "note": "fixes the active model"},
    ]
    law_ids = [node["id"] for node in nodes if node["type"] == "Law"]
    for left, right in zip(law_ids, law_ids[1:]):
        edges.append({"src": left, "dst": right, "type": "depends_on", "note": "dependent solution stage"})
    if law_ids:
        edges.extend([
            {"src": law_ids[-1], "dst": "intermediate", "type": "derived_from", "note": "determines the intermediate state"},
            {"src": "intermediate", "dst": "aux", "type": "supports", "note": "provides the final-stage input"},
            {"src": "aux", "dst": "target", "type": "supports", "note": "yields the requested result"},
        ])
    return {"nodes": nodes, "edges": edges}


def graph_from_blueprint(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Recover latent metadata when a useful parallel form has a malformed graph."""
    concepts = [str(item) for item in blueprint.get("interacting_concepts") or [] if str(item).strip()]
    steps = [str(item) for item in blueprint.get("derivation_steps") or [] if str(item).strip()]
    equations = (concepts + steps)[:5]
    pseudo_construction = {
        "setup": blueprint.get("canonical_problem") or "Fixed problem givens",
        "target": "Obtain the single requested result in the student-facing problem.",
        "governing_equations": equations or [
            "Apply the governing law", "Apply the active constraint", "Solve for the result",
        ],
        "regime_or_constraint_decisions": [
            blueprint.get("key_insight") or blueprint.get("case_or_constraint_handling")
            or "Select the active physical constraint"
        ],
    }
    return graph_from_construction(pseudo_construction)


def closest_source_overlap(candidate: str, sources: List[str]) -> tuple[float, str]:
    """Detect copied/paraphrased source stems before an LLM can self-certify novelty."""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "between", "by", "determine",
        "does", "during", "each", "far", "for", "from", "how", "in", "is", "it",
        "of", "on", "or", "that", "the", "then", "this", "to", "what", "when",
        "which", "with",
    }
    normalized = " ".join(str(candidate or "").casefold().split())
    best_score, best_source = 0.0, ""
    for source in sources:
        prior = " ".join(str(source or "").casefold().split())
        if not normalized or not prior:
            continue
        sequence_score = difflib.SequenceMatcher(None, normalized, prior).ratio()
        candidate_tokens = set(re.findall(r"[a-z0-9]+", normalized)) - stopwords
        prior_tokens = set(re.findall(r"[a-z0-9]+", prior)) - stopwords
        containment_score = (
            len(candidate_tokens & prior_tokens) / min(len(candidate_tokens), len(prior_tokens))
            if candidate_tokens and prior_tokens else 0.0
        )
        score = max(sequence_score, containment_score)
        if score > best_score:
            best_score, best_source = score, source
    return best_score, best_source


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_.])")


def _strip_choice_label(value: Any) -> str:
    return re.sub(r"^\s*(?:\(?[A-Ea-e]\)?[.:)]\s*)", "", str(value or "").strip())


def _answer_matches_choice(answer: Any, choice: Any) -> bool:
    if isinstance(answer, bool):
        return False
    if isinstance(answer, (int, float)) and math.isfinite(float(answer)):
        numbers = [float(x) for x in _NUMBER_RE.findall(str(choice))]
        return len(numbers) == 1 and math.isclose(float(answer), numbers[0], rel_tol=1e-8, abs_tol=1e-10)
    normalized_answer = "".join(_strip_choice_label(answer).split()).casefold()
    normalized_choice = "".join(_strip_choice_label(choice).split()).casefold()
    return bool(normalized_answer) and normalized_answer == normalized_choice


def normalize_blueprint_answer_fields(blueprint: Dict[str, Any]) -> None:
    """Remove harmless answer-format drift before physics verification."""
    correct = blueprint.get("correct_choice")
    choices = blueprint.get("canonical_choices") or []
    if isinstance(choices, list) and sum(
        _answer_matches_choice(correct, choice) for choice in choices
    ) == 1:
        blueprint["expected_answer"] = correct


def _word_count(value: Any) -> int:
    return len(re.findall(r"\b[\w'+-]+\b", str(value or ""), flags=re.UNICODE))


def use_architecture_stage(target_difficulty: int, seed_faithful_hard: bool) -> bool:
    return target_difficulty >= 3 and not seed_faithful_hard


def _looks_truncated(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or text[-1] in "[({=,+-*/":
        return True
    final_words = re.findall(r"[A-Za-z]+", text.casefold())
    return bool(final_words and final_words[-1] in {
        "a", "an", "and", "for", "of", "or", "the", "to", "with",
    })


def construction_quality_errors(value: Any) -> List[str]:
    """Reject complexity that cannot become one compact contest target."""
    if not isinstance(value, dict):
        return ["construction is not an object"]
    errors: List[str] = []
    target = str(value.get("target") or "").strip()
    answer = str(value.get("exact_answer") or "").strip()
    if _word_count(target) > 24:
        errors.append("target exceeds 24 words")
    if re.search(r"\([a-e]\)|\b(?:and|plus)\s+(?:find|determine|state)\b", target, re.I):
        errors.append("target contains separately graded subparts")
    if re.search(
        r"\band\s+(?:the\s+)?(?:impulse|velocity|velocities|speed|time|angle|"
        r"acceleration|force|coefficient|ratio|displacement|angular speed|angular velocity)\b",
        target,
        re.I,
    ):
        errors.append("target requests more than one physical result")
    if ";" in answer or answer.count("=") > 1:
        errors.append("exact_answer is a tuple, list, or multi-result derivation")
    if len(answer) > MAX_EXACT_ANSWER_CHARS:
        errors.append(f"exact_answer exceeds {MAX_EXACT_ANSWER_CHARS} characters")
    if _looks_truncated(answer):
        errors.append("exact_answer appears truncated")
    return errors


def student_facing_quality_errors(value: Any) -> List[str]:
    """Enforce F=ma-like presentation independently of conceptual difficulty."""
    if not isinstance(value, dict):
        return ["solution blueprint is not an object"]
    errors: List[str] = []
    stem = str(value.get("canonical_problem") or "").strip()
    choices = value.get("canonical_choices") or []
    if _word_count(stem) > MAX_STEM_WORDS:
        errors.append(f"student-facing stem exceeds {MAX_STEM_WORDS} words")
    if re.search(r"\([a-e]\)", stem, re.I):
        errors.append("student-facing stem contains separately graded subparts")
    if isinstance(choices, list):
        for index, choice in enumerate(choices, start=1):
            if _word_count(choice) > MAX_CHOICE_WORDS:
                errors.append(f"choice {index} exceeds {MAX_CHOICE_WORDS} words")
            if re.search(
                r"\b(?:incorrect|error|wrong|omitted|missing|neglects?|sign error)\b",
                str(choice),
                re.I,
            ):
                errors.append(f"choice {index} exposes its distractor rationale")
            if _looks_truncated(choice):
                errors.append(f"choice {index} appears truncated")
    return errors


def solution_blueprint_ok(value: Any, difficulty: int = 3, distractor_count: int = 4) -> bool:
    if not isinstance(value, dict):
        return False
    required = (
        "canonical_problem", "canonical_choices", "novel_setup", "fixed_givens", "interacting_concepts", "key_insight",
        "derivation_steps", "routine_shortcut_test", "difficulty_justification",
        "expected_answer", "correct_choice", "verification_method", "verification_steps",
        "equation_closure_check",
    )
    if any(not value.get(key) for key in required):
        return False
    if student_facing_quality_errors(value):
        return False
    steps = value.get("derivation_steps")
    concepts = value.get("interacting_concepts")
    canonical_choices = value.get("canonical_choices")
    verification_steps = value.get("verification_steps")
    closure = value.get("equation_closure_check")
    min_steps = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[difficulty]
    if not isinstance(steps, list) or len(steps) < min_steps or len(steps) > 12:
        return False
    if difficulty >= 3 and (not isinstance(concepts, list) or len(concepts) < 2):
        return False
    if not isinstance(canonical_choices, list) or len(canonical_choices) != distractor_count + 1:
        return False
    if not isinstance(verification_steps, list) or not 2 <= len(verification_steps) <= 6:
        return False
    if not isinstance(closure, list) or not 3 <= len(closure) <= 8:
        return False
    if sum(_answer_matches_choice(value.get("correct_choice"), choice) for choice in canonical_choices) != 1:
        return False
    if not _answer_matches_choice(value.get("expected_answer"), value.get("correct_choice")):
        return False
    shortcut = str(value.get("routine_shortcut_test") or "").casefold()
    if difficulty >= 3 and not shortcut:
        return False
    if difficulty == 5 and not str(value.get("case_or_constraint_handling") or "").strip():
        return False
    derivation_text = " ".join(str(step) for step in steps).casefold()
    drafting_markers = (
        "after detailed", "after algebra", "see detailed", "must be adjusted",
        "chosen so that", "guess", "contradiction", "if not, set",
    )
    if any(marker in derivation_text for marker in drafting_markers):
        return False
    distractors = value.get("distractor_choices")
    if isinstance(distractors, dict):
        distractors = list(distractors.values())
    return (
        isinstance(distractors, list)
        and len(distractors) == distractor_count
        and all(str(item).strip() for item in distractors)
    )


def solution_construction_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = (
        "setup", "target", "fixed_givens", "governing_equations",
        "regime_or_constraint_decisions", "derived_intermediates",
        "preserved_seed_dependencies", "architecture_trace", "derivation_steps",
        "exact_answer", "independent_checks", "closure_proof",
    )
    if any(not value.get(key) for key in required):
        return False
    if construction_quality_errors(value):
        return False
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    invalid_markers = (
        "placeholder", "continued_if_needed", "remove_if_used", "do not display",
        "truncated", "to be completed",
    )
    exact_answer = str(value.get("exact_answer") or "").strip()
    if any(marker in serialized for marker in invalid_markers):
        return False
    if not exact_answer or exact_answer[-1] in "[({=,+-*/":
        return False
    return (
        len(value.get("fixed_givens") or []) >= 4
        and len(value.get("governing_equations") or []) >= 3
        and len(value.get("derived_intermediates") or []) >= 2
        and len(value.get("preserved_seed_dependencies") or []) >= 3
        and len(value.get("architecture_trace") or []) >= 4
        and 5 <= len(value.get("derivation_steps") or []) <= 12
        and 2 <= len(value.get("independent_checks") or []) <= 5
        and 3 <= len(value.get("closure_proof") or []) <= 8
    )


def solution_construction_errors(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return ["construction is not an object"]
    errors: List[str] = []
    required = (
        "setup", "target", "fixed_givens", "governing_equations",
        "regime_or_constraint_decisions", "derived_intermediates",
        "preserved_seed_dependencies", "architecture_trace", "derivation_steps",
        "exact_answer", "independent_checks", "closure_proof",
    )
    errors.extend(f"missing or empty {key}" for key in required if not value.get(key))
    errors.extend(construction_quality_errors(value))
    serialized = json.dumps(value, ensure_ascii=False).casefold()
    invalid_markers = (
        "placeholder", "continued_if_needed", "remove_if_used", "do not display",
        "truncated", "to be completed",
    )
    found_markers = [marker for marker in invalid_markers if marker in serialized]
    if found_markers:
        errors.append("draft/truncation markers present: " + ", ".join(found_markers))
    exact_answer = str(value.get("exact_answer") or "").strip()
    if exact_answer and exact_answer[-1] in "[({=,+-*/":
        errors.append("exact_answer ends with an incomplete expression")
    bounds = {
        "fixed_givens": (4, None), "governing_equations": (3, None),
        "derived_intermediates": (2, 6), "preserved_seed_dependencies": (3, 6),
        "architecture_trace": (4, None), "derivation_steps": (5, 12),
        "independent_checks": (2, 5), "closure_proof": (3, 8),
    }
    for key, (minimum, maximum) in bounds.items():
        items = value.get(key)
        if not isinstance(items, list):
            errors.append(f"{key} is not a list")
        elif len(items) < minimum or (maximum is not None and len(items) > maximum):
            errors.append(f"{key} count {len(items)} outside {minimum}..{maximum or 'unbounded'}")
    return errors


def architecture_adherence_passes(audit: Any, target_difficulty: int) -> bool:
    if not isinstance(audit, dict) or audit.get("verdict") != "PASS":
        return False
    checks = audit.get("mandatory_step_checks")
    return (
        audit.get("physics_closed") is True
        and not (audit.get("issues") or [])
        and audit.get("routine_shortcut_exists") is False
        and audit.get("familiar_template") is False
        and audit.get("single_scalar_or_categorical_target") is True
        and audit.get("concise_student_render_feasible") is True
        and audit.get("derived_intermediate_leaked_as_given") is False
        and bool(str(audit.get("indispensable_decision") or "").strip())
        and bool(str(audit.get("counterfactual_without_decision") or "").strip())
        and isinstance(audit.get("actual_difficulty"), int)
        and not isinstance(audit.get("actual_difficulty"), bool)
        and audit.get("actual_difficulty") >= target_difficulty
        and isinstance(checks, list)
        and len(checks) >= 4
        and isinstance(audit.get("shortest_solution_outline"), list)
        and len(audit.get("shortest_solution_outline")) >= 4
        and all(
            isinstance(check, dict)
            and check.get("indispensable") is True
            and bool(str(check.get("architecture_step") or "").strip())
            and bool(str(check.get("construction_evidence") or "").strip())
            for check in checks
        )
    )


def architecture_static_errors(value: Any, target_difficulty: int) -> List[str]:
    """Catch familiar mechanisms that LLM reviewers repeatedly overrate."""
    if not isinstance(value, dict):
        return ["architecture is not an object"]
    errors: List[str] = []
    mapping = value.get("reference_depth_mapping")
    if target_difficulty >= 4 and (
        not isinstance(mapping, list)
        or len([item for item in mapping if str(item).strip()]) < 3
    ):
        errors.append("architecture does not map its depth to the primary hard seed")
    mandatory = [str(item).strip() for item in value.get("mandatory_solution_structure") or []]
    conceptual = [
        item for item in mandatory
        if not re.search(
            r"^\s*(?:\d+[.)]\s*)?(?:state|write|integrate|substitute|check|verify|confirm|count)\b",
            item,
            re.I,
        )
    ]
    if target_difficulty >= 4 and len(conceptual) < 4:
        errors.append("fewer than four mandatory steps are conceptual rather than setup/algebra/checks")
    selected = str(value.get("selected_mechanism") or "").casefold()
    candidates = value.get("candidate_mechanisms") or []
    selected_details = ""
    for candidate in candidates:
        mechanism = str((candidate or {}).get("mechanism") or "")
        if selected and (selected in mechanism.casefold() or mechanism.casefold() in selected):
            selected_details = json.dumps(candidate, ensure_ascii=False).casefold()
            break
    text = " ".join([
        selected,
        selected_details,
        str(value.get("selection_reason") or "").casefold(),
        json.dumps(value.get("mandatory_solution_structure") or [], ensure_ascii=False).casefold(),
    ])
    routine_target_patterns = (
        r"(?:speed|time).{0,80}(?:when|instant).{0,50}(?:pure )?rolling",
        r"(?:slip|sliding).{0,30}(?:to|→).{0,30}(?:pure )?roll(?:ing)? transition",
        r"rod.{0,80}(?:sticks|sticking).{0,50}(?:fixed )?peg",
        r"immediate post.{0,80}rolling-without-slip",
    )
    if target_difficulty >= 4 and any(re.search(pattern, text) for pattern in routine_target_patterns):
        errors.append("selected mechanism is a known routine D1-D3 template, not a D4 architecture")
    return errors


def architecture_validity_passes(audit: Any, target_difficulty: int) -> bool:
    if not isinstance(audit, dict) or audit.get("verdict") != "PASS":
        return False
    checks = [str(item).strip() for item in (audit.get("unknown_equation_check") or []) if str(item).strip()]
    return (
        audit.get("physics_closed") is True
        and not (audit.get("issues") or [])
        and audit.get("syllabus_safe") is True
        and audit.get("routine_shortcut_exists") is False
        and audit.get("single_scalar_or_categorical_target") is True
        and audit.get("concise_student_render_feasible") is True
        and bool(str(audit.get("indispensable_decision") or "").strip())
        and isinstance(audit.get("shortest_solution_outline"), list)
        and len(audit.get("shortest_solution_outline")) >= 4
        and isinstance(audit.get("projected_difficulty"), int)
        and not isinstance(audit.get("projected_difficulty"), bool)
        and audit.get("projected_difficulty") >= target_difficulty
        and isinstance(checks, list)
        and len(checks) >= 2
    )


def build_architecture_adherence_prompt(
    architecture: Dict[str, Any], construction: Dict[str, Any], target_difficulty: int,
) -> str:
    return f"""TARGET DIFFICULTY: {target_difficulty}/5

APPROVED ARCHITECTURE:
{json.dumps(architecture, ensure_ascii=False)}

CONCRETE SOLVED CONSTRUCTION:
{json.dumps(construction, ensure_ascii=False)}

For every entry in approved mandatory_solution_structure, cite the exact construction equation or
deduction implementing it and decide whether removing it would change or prevent the shortest solve.
First reconstruct the shortest solution without trusting architecture_trace. Name the indispensable
modeling decision, state how the governing equations change if it is missed, and reject a familiar
template even if the construction lists many steps. Do not count restated givens, routine algebra or
integration, substitution, or consistency checks toward D4. Also reject multiple outputs or a construction whose
required givens cannot fit a concise student-facing problem. Treat every quantity in derived_intermediates
as something the student must derive: if the setup or fixed_givens supplies one directly, set
derived_intermediate_leaked_as_given=true and FAIL. PASS only if all mandatory steps are concrete
and indispensable and the shortest solve still reaches difficulty {target_difficulty}."""


def independent_solve_passes(audit: Any) -> bool:
    if not isinstance(audit, dict) or audit.get("verdict") != "PASS":
        return False
    return (
        audit.get("well_posed") is True
        and audit.get("physics_valid") is True
        and audit.get("unique_answer") is True
        and audit.get("choice_match") is True
        and isinstance(audit.get("independent_derivation"), list)
        and bool(audit.get("independent_derivation"))
        and bool(str(audit.get("independent_result") or "").strip())
    )


def difficulty_audit_passes(audit: Any, target_difficulty: int) -> bool:
    if not isinstance(audit, dict) or audit.get("verdict") != "PASS":
        return False
    actual = audit.get("actual_difficulty")
    deductions = audit.get("conceptual_deductions")
    decisions = audit.get("non_obvious_decisions")
    min_decisions = 0 if target_difficulty <= 3 else (1 if target_difficulty == 4 else 2)
    return (
        isinstance(actual, int) and not isinstance(actual, bool)
        and actual >= target_difficulty
        and audit.get("routine_shortcut_exists") is False
        and (target_difficulty < 4 or audit.get("familiar_template") is False)
        and (target_difficulty < 4 or audit.get("architecture_steps_preserved") is True)
        and (target_difficulty < 4 or not (audit.get("missing_architecture_steps") or []))
        and isinstance(deductions, list)
        and len(deductions) >= {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[target_difficulty]
        and isinstance(decisions, list)
        and len(decisions) >= min_decisions
    )


def repairable_choice_mismatch(audit: Any) -> bool:
    if not isinstance(audit, dict):
        return False
    result = str(audit.get("independent_result") or "").strip()
    return (
        audit.get("well_posed") is True
        and audit.get("physics_valid") is True
        and audit.get("unique_answer") is True
        and audit.get("choice_match") is False
        and bool(result)
        and result.casefold() != "unknown"
        and isinstance(audit.get("independent_derivation"), list)
        and bool(audit.get("independent_derivation"))
    )


def choice_repair_schema(choice_count: int) -> Dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "canonical_choices": {
                "type": "array", "minItems": choice_count, "maxItems": choice_count,
                "items": {"type": "string"},
            },
            "correct_choice": {"type": "string"},
            "distractor_choices": {
                "type": "array", "minItems": choice_count - 1, "maxItems": choice_count - 1,
                "items": {"type": "string"},
            },
        },
        "required": ["canonical_choices", "correct_choice", "distractor_choices"],
    }


def build_choice_repair_prompt(blueprint: Dict[str, Any], audit: Dict[str, Any]) -> str:
    return f"""FROZEN PROBLEM STEM:
{blueprint.get('canonical_problem', '')}

INDEPENDENT DERIVATION:
{json.dumps(audit.get('independent_derivation') or [], ensure_ascii=False)}

AUTHORITATIVE RESULT (copy this exact text as correct_choice and as exactly one canonical choice):
{audit.get('independent_result', '')}

Create four distinct plausible wrong choices and return the repaired choice set."""


def build_blind_solve_prompt(blueprint: Dict[str, Any]) -> str:
    return f"""PROBLEM:
{blueprint.get('canonical_problem', '')}

ANSWER_CHOICES:
{json.dumps(blueprint.get('canonical_choices') or [], ensure_ascii=False)}

Return exactly these keys. choice_match means exactly one listed choice matches your result:
{{
  "verdict": "PASS"|"FAIL",
  "issues": ["specific issue"],
  "independent_derivation": ["complete independent logical or mathematical step"],
  "independent_result": "result obtained independently or UNKNOWN",
  "matching_choice": "exact matching choice text or UNKNOWN",
  "well_posed": true,
  "physics_valid": true,
  "unique_answer": true,
  "choice_match": true
}}
"""


def build_difficulty_audit_prompt(
    blueprint: Dict[str, Any], independent_solve: Dict[str, Any], exemplars: List[str],
    required_topic: str, architecture: Dict[str, Any] | None = None,
) -> str:
    return f"""COMPETITION EXEMPLARS (calibration only):
{json.dumps(exemplars[:5], ensure_ascii=False)}

REQUIRED_TOPIC: {required_topic}
PROBLEM:
{blueprint.get('canonical_problem', '')}

SHORTEST VERIFIED SOLUTION TO CLASSIFY:
{json.dumps(independent_solve.get('independent_derivation') or [], ensure_ascii=False)}

LOCKED MANDATORY SOLUTION STRUCTURE:
{json.dumps((architecture or {}).get('mandatory_solution_structure') or [], ensure_ascii=False)}

architecture_steps_preserved may be true only if every locked mandatory step is both present and
indispensable in the shortest verified solution; list every absent or optional step.

Return exactly these keys:
{{
  "verdict": "PASS"|"FAIL",
  "issues": ["specific calibration issue"],
  "actual_difficulty": 1,
  "conceptual_deductions": ["necessary conceptual deduction; exclude arithmetic"],
  "non_obvious_decisions": ["necessary non-obvious decision in the shortest solve"],
  "routine_shortcut_exists": false,
  "familiar_template": false,
  "architecture_steps_preserved": true,
  "missing_architecture_steps": []
}}
"""


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", default="./generated/generated_skeletons.jsonl")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--verify-model", default="")
    ap.add_argument("--domain", default="fma")
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--required-topic", default="F=ma mechanics")
    ap.add_argument("--target-difficulty", type=int, choices=range(1, 6), default=3)
    ap.add_argument("--num-choices", type=int, choices=(4, 5), default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument(
        "--seed-faithful-hard", action="store_true",
        help="For D4/D5, generate a parallel form that preserves the hard seed's solution topology.",
    )
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args=ap.parse_args()
    verify_model = args.verify_model or args.model

    rows=read_jsonl(args.bundles)
    if args.limit and args.limit>0:
        rows=rows[:args.limit]

    total=len(rows)
    rejected_path = args.out + ".rejected.jsonl"
    architecture_trace_path = args.out + ".architecture.jsonl"
    construction_trace_path = args.out + ".construction.jsonl"
    with open(rejected_path, "w", encoding="utf-8"):
        pass
    with open(architecture_trace_path, "w", encoding="utf-8"):
        pass
    with open(construction_trace_path, "w", encoding="utf-8"):
        pass

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        failures = 0
        for i, b in enumerate(rows, start=1):
            domain=b.get("domain") or args.domain
            seed_graph=b.get("seed_graph_text") or b.get("seed_skeleton_text") or ""
            seed_question=b.get("seed_question_text") or ""
            sol_ex=b.get("solution_exemplars") or []
            ex_blocks=[]
            for j,ex in enumerate(sol_ex[:8], start=1):
                exemplar_graph = ex.get('graph_text') or ex.get('skeleton_text','')
                if args.target_difficulty >= 4:
                    exemplar_graph = abstract_solution_profile(exemplar_graph)
                ex_blocks.append(f"EX {j}:\n{exemplar_graph}")
            sol_block="\n\n".join(ex_blocks) if ex_blocks else "(none)"
            source_questions = [seed_question] + [
                str(ex.get("question_text") or "") for ex in sol_ex
            ]
            authentic_examples: List[str] = []
            for candidate in [
                seed_question,
                *[str(ex.get("question_text") or "") for ex in (b.get("question_exemplars") or [])],
                *[str(ex.get("question_text") or "") for ex in sol_ex],
            ]:
                cleaned = str(candidate or "").strip()
                if cleaned and cleaned not in authentic_examples:
                    authentic_examples.append(cleaned)

            seed_faithful_hard = args.seed_faithful_hard and args.target_difficulty >= 4
            seed_question_for_prompt = (
                seed_question if args.target_difficulty <= 3 or seed_faithful_hard
                else "(withheld at D4/D5 to prevent scenario copying; use only abstract solution structures)"
            )
            seed_graph_for_prompt = (
                seed_graph if args.target_difficulty <= 3
                else seed_faithful_profile(seed_graph) if seed_faithful_hard
                else abstract_solution_profile(seed_graph)
            )
            user=USER_TMPL.format(
                domain=domain,
                required_topic=args.required_topic,
                difficulty_contract=difficulty_instruction(args.target_difficulty),
                seed_graph=seed_graph_for_prompt,
                seed_question=seed_question_for_prompt,
                solution_exemplars_block=sol_block,
                node_types_block="\n".join(f"- {o}" for o in NODE_TYPES),
                edge_types_block="\n".join(f"- {o}" for o in EDGE_TYPES),
                distractor_count=args.num_choices - 1,
                choice_count=args.num_choices,
            )

            architecture = None
            construction = None
            architecture_history: List[str] = []
            architecture_prompt_base = ""
            if use_architecture_stage(args.target_difficulty, seed_faithful_hard):
                required_conceptual_steps = 3 if args.target_difficulty == 3 else args.target_difficulty
                architecture_prompt_base = f"""Required topic: {args.required_topic}
Required difficulty: {args.target_difficulty}/5

PRIMARY HARD SEED STRUCTURE (transfer its depth topology, never its surface scenario):
{seed_graph_for_prompt}

Retrieved solution structures:
{sol_block}

AUTHENTIC RECENT F=ma QUESTIONS (difficulty/style calibration only; never copy their scenario):
{json.dumps(authentic_examples[:3], ensure_ascii=False)}

BOUNDED MECHANISM LIBRARY:
{bounded_mechanism_library(args.required_topic)}

Propose exactly three different solution mechanisms. For difficulty 4, the selected mechanism needs
at least one indispensable non-obvious decision and four dependent conceptual deductions. For
difficulty 5, it needs at least two non-obvious decisions including genuine case/regime handling.
The four deductions must all be necessary for one final scalar or categorical answer. Intermediate
states may be solved privately, but never expose them as a tuple of requested answers. Prefer a compact
numerical instantiation that can be stated in at most 180 words with choices of at most 32 words.
The primary hard seed outranks easier nearest-neighbor exemplars as the depth reference. Fill
reference_depth_mapping with at least three explicit correspondences from its state evolution, coupling
points, or insight type to the new mechanism. Preserve the dependency topology while changing the
objects, narrative, variables, values, and visible presentation.
Put only the {required_conceptual_steps} or more indispensable conceptual deductions in
mandatory_solution_structure. Setup,
equation counting, routine algebra/integration, substitution, and consistency or limiting checks belong
elsewhere and do not satisfy its minimum length.
The mechanism must not reduce to repeated elastic-collision formulas, direct conservation equations,
or another familiar textbook sequence. Do not choose values, write a stem, or solve arithmetic yet.
The selected mechanism must nevertheless make every modeling commitment before construction: state
whether motion is 1D, planar, or 3D; whether each body is a point mass or rigid body; whether contacts
are smooth or rough; the direction and scalar/vector content of every impulse; whether later targets
are fixed or finite-mass; non-grazing and non-simultaneous conditions; and an existence/uniqueness
condition for each later contact time. Never defer alternatives such as "point mass or rigid particle"
to the constructor. Count every post-impact translational and rotational unknown and name the
independent standard equation that closes it.
Use only standard F=ma mechanics. Ban deformable media, material failure, robotics, granular or
viscoelastic behavior, phase changes, distributed impulse waves, variable mass, and invented force,
damping, or restitution laws. For each mechanism, name the standard laws and the givens needed to close
every unknown. The selected mechanism must include an explicit unknown-versus-equation closure check.
You may transfer an abstract solution mechanism, dependency shape, or comparable target relation from
a retrieved source. To keep the result original on the surface, change the physical objects, narrative,
variable names, numerical/symbolic givens, and presentation, and do not preserve a recognizable phrase
or diagram. A reference-frame change, conservation-law chain, coefficient-of-restitution substitution,
or collision ordering by itself is at most D3; it can appear only as one subordinate step inside a deeper
mechanism."""

            if args.debug:
                bid=b.get("bundle_id","?")
                print(f"[{i}/{total}] bundle_id={bid} calling model={args.model}", file=sys.stderr, flush=True)
                t0=time.time()

            best=None
            last_exc = None
            rejection_feedback = ""
            generation_effort = (
                "low" if args.target_difficulty >= 4 and architecture
                else "medium" if args.target_difficulty >= 3
                else "low"
            )
            for k in range(max(1, args.max_attempts)):
                try:
                    architecture = None
                    architecture_adherence = None
                    if use_architecture_stage(args.target_difficulty, seed_faithful_hard):
                        banned = (
                            "\nPreviously rejected selected mechanisms (do not reuse or rename):\n- "
                            + "\n- ".join(architecture_history)
                            if architecture_history else ""
                        )
                        architecture = llm_json(
                            args.model, SYSTEM_DIFFICULTY_ARCHITECT,
                            architecture_prompt_base + banned + rejection_feedback,
                            args.provider,
                            temperature=0.2 if args.model.startswith("phi4-reasoning") else 0.35,
                            reasoning_effort="medium",
                            json_schema=DIFFICULTY_ARCHITECT_SCHEMA, token_budget_override=12000,
                        )
                        architecture_issues = architecture_static_errors(
                            architecture, args.target_difficulty
                        )
                        if architecture_issues:
                            selected = str(architecture.get("selected_mechanism") or "").strip()
                            if selected:
                                architecture_history.append(selected)
                            raise ValueError(
                                "architecture failed deterministic depth gate: "
                                + "; ".join(architecture_issues)
                            )
                        architecture_validity = None
                        for architecture_revision in range(2):
                            architecture_validity = llm_json(
                                verify_model,
                                SYSTEM_ARCHITECTURE_VALIDITY,
                                f"""TARGET DIFFICULTY: {args.target_difficulty}/5
REQUIRED TOPIC: {args.required_topic}
PROPOSED ARCHITECTURE:
{json.dumps(architecture, ensure_ascii=False)}

Audit the selected pre-numeric mechanism only. Map each symbolic unknown class to an independent standard
equation and confirm closure_requirements names the future fixed givens needed to instantiate it. Do not
demand numerical values or a finished stem here; those belong to the next construction gate. Judge the
shortest prospective solution, not the proposed step count. Explicitly outline that shortest solution,
identify the indispensable decision, and verify that it ends in one compact target. Do not count restated
givens, routine formula writing, algebra/integration, substitution, or checks as conceptual deductions.""",
                                args.provider,
                                temperature=0.0,
                                reasoning_effort="low",
                                json_schema=ARCHITECTURE_VALIDITY_SCHEMA,
                                token_budget_override=12000,
                                ollama_think=False if args.provider == "ollama" else None,
                            )
                            with open(architecture_trace_path, "a", encoding="utf-8") as trace_f:
                                write_jsonl_line(trace_f, {
                                    "bundle_id": b.get("bundle_id"),
                                    "attempt": k + 1,
                                    "architecture_revision": architecture_revision,
                                    "architecture": architecture,
                                    "architecture_validity": architecture_validity,
                                })
                            if architecture_validity_passes(
                                architecture_validity, args.target_difficulty
                            ):
                                break
                            if architecture_revision == 0:
                                architecture = llm_json(
                                    args.model,
                                    SYSTEM_DIFFICULTY_ARCHITECT,
                                    f"""Revise the proposed architecture in place using the audit below.
Do not change to an easier mechanism and do not write the final stem or choose numerical values.
Commit explicitly to the missing dimensionality, body types, contact directions, event geometry,
nondegenerate parameter conditions, and unknown-to-equation closure. Return the complete architecture
schema, retaining exactly three candidates but correcting the selected mechanism and its mandatory steps.

PROPOSED ARCHITECTURE:
{json.dumps(architecture, ensure_ascii=False)}

VALIDATOR AUDIT:
{json.dumps(architecture_validity, ensure_ascii=False)}""",
                                    args.provider,
                                    temperature=0.1,
                                    reasoning_effort="medium",
                                    json_schema=DIFFICULTY_ARCHITECT_SCHEMA,
                                    token_budget_override=12000,
                                )
                        selected = str(architecture.get("selected_mechanism") or "").strip()
                        if selected:
                            architecture_history.append(selected)
                        if not architecture_validity_passes(
                            architecture_validity, args.target_difficulty
                        ):
                            raise ValueError(
                                "architecture failed pre-construction physics gate: "
                                + json.dumps(architecture_validity, ensure_ascii=False)[:1200]
                            )
                        construction_prompt_base = f"""REQUIRED TOPIC: {args.required_topic}
REQUIRED DIFFICULTY: {args.target_difficulty}/5
APPROVED SOLUTION ARCHITECTURE:
{json.dumps(architecture, ensure_ascii=False)}

Construct and fully solve one concrete problem now. Do not write answer choices. Do not omit the
dependent event after the initial impulse/collision. Ask only for the final scalar or categorical target;
all intermediate states remain solution work. Favor fixed numerical givens and a compact exact answer.
Avoid verbose coordinate bookkeeping when an ordinary diagram-free geometric description or declared
positive direction suffices. List the quantities that create the hard dependency chain under
derived_intermediates; none may appear as a supplied value in setup or fixed_givens. Use
preserved_seed_dependencies to show how at least three semantic dependencies from the primary seed remain
necessary in this construction. Each architecture_trace entry must point to a specific equation or
derivation step. exact_answer must be derived, not selected or tuned."""
                        construction_effort = "low" if args.model.startswith("gpt-oss") else "medium"
                        construction_feedback = ""
                        construction_failure_detail: Any = None
                        for construction_attempt in range(2):
                            construction = llm_json(
                                args.model, SYSTEM_SOLUTION_CONSTRUCTOR,
                                construction_prompt_base + construction_feedback,
                                args.provider, temperature=0.2,
                                reasoning_effort=construction_effort,
                                json_schema=SOLUTION_CONSTRUCTION_SCHEMA,
                                token_budget_override=12000,
                            )
                            if not solution_construction_ok(construction):
                                construction_issues = solution_construction_errors(construction)
                                construction_failure_detail = construction_issues
                                with open(construction_trace_path, "a", encoding="utf-8") as trace_f:
                                    write_jsonl_line(trace_f, {
                                        "bundle_id": b.get("bundle_id"),
                                        "attempt": k + 1,
                                        "construction_attempt": construction_attempt + 1,
                                        "construction": construction,
                                        "structural_issues": construction_issues,
                                        "architecture_adherence": None,
                                    })
                                construction_feedback = (
                                    "\n\nThe previous construction was structurally incomplete. Keep the "
                                    "locked architecture and reconstruct it with every required field. "
                                    "Specific defects: " + json.dumps(construction_issues)
                                )
                                continue
                            architecture_adherence = llm_json(
                                verify_model,
                                SYSTEM_ARCHITECTURE_ADHERENCE,
                                build_architecture_adherence_prompt(
                                    architecture, construction, args.target_difficulty
                                ),
                                args.provider,
                                temperature=0.0,
                                reasoning_effort="low",
                                json_schema=ARCHITECTURE_ADHERENCE_SCHEMA,
                                token_budget_override=5000,
                                ollama_think=False if args.provider == "ollama" else None,
                            )
                            with open(construction_trace_path, "a", encoding="utf-8") as trace_f:
                                write_jsonl_line(trace_f, {
                                    "bundle_id": b.get("bundle_id"),
                                    "attempt": k + 1,
                                    "construction_attempt": construction_attempt + 1,
                                    "construction": construction,
                                    "architecture_adherence": architecture_adherence,
                                })
                            construction_failure_detail = architecture_adherence
                            if architecture_adherence_passes(
                                architecture_adherence, args.target_difficulty
                            ):
                                break
                            construction_feedback = (
                                "\n\nThe previous concrete construction failed the locked-architecture "
                                "adherence audit. Preserve the selected mechanism but replace the setup "
                                "and values so every mandatory step is physically valid and indispensable. "
                                "Do not merely assert complexity. Audit findings:\n"
                                + json.dumps(architecture_adherence, ensure_ascii=False)
                            )
                        else:
                            raise ValueError(
                                "solution construction failed architecture-adherence checks: "
                                + json.dumps(construction_failure_detail, ensure_ascii=False)[:1200]
                            )
                        generation_user = (
                            f"REQUIRED TOPIC: {args.required_topic}\n"
                            f"REQUIRED DIFFICULTY: {args.target_difficulty}/5\n"
                            f"NUMBER OF CHOICES: {args.num_choices}\n\n"
                            "LOCKED MANDATORY SOLUTION STRUCTURE:\n"
                            + json.dumps(architecture.get("mandatory_solution_structure") or [], ensure_ascii=False)
                            + "\n\nFROZEN SOLVED CONSTRUCTION:\n"
                            + json.dumps(construction, ensure_ascii=False)
                            + "\n\nRender only this construction into problem_graph and solution_blueprint. "
                              "canonical_problem must preserve every fixed given and event. Copy the frozen "
                              "derivation faithfully but ask only for its single final target. Keep the stem "
                              f"at most {MAX_STEM_WORDS} words and every choice at most {MAX_CHOICE_WORDS} words. "
                              "Choices must contain only answer content, never explanations of their errors. "
                              "Make expected_answer and correct_choice byte-for-byte "
                              "equal to FROZEN SOLVED CONSTRUCTION.exact_answer. Create exactly "
                            + str(args.num_choices - 1)
                            + " distractors and exactly " + str(args.num_choices) + " canonical choices."
                        )
                    else:
                        generation_user = user + rejection_feedback
                    js=llm_json(
                        args.model,
                        (
                            SYSTEM_CONSTRUCTION_RENDERER if construction is not None
                            else SYSTEM_SEED_FAITHFUL if seed_faithful_hard else SYSTEM
                        ),
                        generation_user, args.provider, temperature=0.1 if construction is not None else 0.25,
                        reasoning_effort=(
                            "medium" if construction is not None and args.target_difficulty >= 4
                            else "low" if construction is not None else generation_effort
                        ),
                        json_schema=generation_schema(args.target_difficulty, args.num_choices),
                        ollama_think=False if args.provider == "ollama" else None,
                    )
                except (ValueError, RuntimeError) as exc:
                    last_exc = exc
                    if args.debug:
                        print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} retry {k+1} model_error={exc}", file=sys.stderr, flush=True)
                    time.sleep(0.6 * (k + 1))
                    continue
                blueprint = js.get("solution_blueprint") or {}
                normalize_blueprint_answer_fields(blueprint)
                graph = normalize_graph(js.get("problem_graph") or {})
                if not graph_ok(graph, args.target_difficulty):
                    fallback_graph = (
                        graph_from_construction(construction)
                        if construction is not None else graph_from_blueprint(blueprint)
                    )
                    graph = normalize_graph(fallback_graph)
                gt = graph_text(graph)
                graph_valid = graph_ok(graph, args.target_difficulty)
                blueprint_valid = solution_blueprint_ok(
                    blueprint, args.target_difficulty, args.num_choices - 1
                )
                presentation_issues = student_facing_quality_errors(blueprint)
                difficulty_exact = js.get("difficulty") == args.target_difficulty
                overlap_score, overlap_source = closest_source_overlap(
                    str(blueprint.get("canonical_problem") or ""), source_questions
                )
                originality_valid = overlap_score < 0.65
                construction_match = (
                    construction is None or _answer_matches_choice(
                        construction.get("exact_answer"), blueprint.get("expected_answer")
                    )
                )
                static_ok = (
                    graph_valid and blueprint_valid and difficulty_exact
                    and originality_valid and construction_match
                )
                blind_solve = None
                difficulty_audit = None
                audit_error = None
                ok = False
                # A graph-wrapper defect must not suppress feedback about an easy or invalid
                # underlying solution. Audit every structurally complete blueprint.
                if blueprint_valid and difficulty_exact:
                    try:
                        blind_solve = audit_json_with_fallback(
                            verify_model,
                            SYSTEM_BLIND_SOLVER,
                            build_blind_solve_prompt(js.get("solution_blueprint") or {}),
                            args.provider,
                            BLIND_SOLVE_SCHEMA,
                        )
                        if repairable_choice_mismatch(blind_solve):
                            blueprint_to_repair = js.get("solution_blueprint") or {}
                            repaired_choices = llm_json(
                                args.model,
                                SYSTEM_CHOICE_REPAIR,
                                build_choice_repair_prompt(blueprint_to_repair, blind_solve),
                                args.provider,
                                temperature=0.15,
                                reasoning_effort="low",
                                json_schema=choice_repair_schema(args.num_choices),
                                token_budget_override=3000,
                                ollama_think=False if args.provider == "ollama" else None,
                            )
                            result = str(blind_solve.get("independent_result") or "").strip()
                            repaired_correct = str(repaired_choices.get("correct_choice") or "").strip()
                            repaired_all = repaired_choices.get("canonical_choices") or []
                            repaired_wrong = repaired_choices.get("distractor_choices") or []
                            repair_valid = (
                                isinstance(repaired_all, list)
                                and len(repaired_all) == args.num_choices
                                and isinstance(repaired_wrong, list)
                                and len(repaired_wrong) == args.num_choices - 1
                                and _answer_matches_choice(result, repaired_correct)
                                and sum(_answer_matches_choice(result, item) for item in repaired_all) == 1
                                and all(str(item).strip() for item in repaired_all)
                            )
                            if repair_valid:
                                blueprint_to_repair["canonical_choices"] = repaired_all
                                blueprint_to_repair["correct_choice"] = repaired_correct
                                blueprint_to_repair["expected_answer"] = repaired_correct
                                blueprint_to_repair["distractor_choices"] = repaired_wrong
                                independent_steps = [
                                    str(step).strip() for step in blind_solve.get("independent_derivation") or []
                                    if str(step).strip()
                                ]
                                if args.target_difficulty <= len(independent_steps) <= 12:
                                    blueprint_to_repair["derivation_steps"] = independent_steps
                                blueprint_to_repair["verification_method"] = (
                                    "Independent blind solve followed by a second blind solve after choice repair"
                                )
                                blueprint_to_repair["verification_steps"] = independent_steps[-6:] or [
                                    f"Independent solver obtained {result}.",
                                    "The repaired choices contain that result exactly once.",
                                ]
                                construction_match = True
                                static_ok = (
                                    graph_valid
                                    and solution_blueprint_ok(
                                        blueprint_to_repair,
                                        args.target_difficulty,
                                        args.num_choices - 1,
                                    )
                                    and difficulty_exact and originality_valid
                                )
                                blind_solve = audit_json_with_fallback(
                                    verify_model,
                                    SYSTEM_BLIND_SOLVER,
                                    build_blind_solve_prompt(blueprint_to_repair),
                                    args.provider,
                                    BLIND_SOLVE_SCHEMA,
                                )
                        if independent_solve_passes(blind_solve):
                            calibration_examples = [seed_question] + [
                                str(ex.get("question_text") or "")
                                for ex in (b.get("question_exemplars") or [])
                            ]
                            difficulty_audit = audit_json_with_fallback(
                                verify_model,
                                SYSTEM_DIFFICULTY_AUDIT,
                                build_difficulty_audit_prompt(
                                    js.get("solution_blueprint") or {}, blind_solve,
                                    [text for text in calibration_examples if text],
                                    args.required_topic,
                                    architecture,
                                ),
                                args.provider,
                                DIFFICULTY_AUDIT_SCHEMA,
                            )
                            ok = static_ok and difficulty_audit_passes(
                                difficulty_audit, args.target_difficulty
                            )
                    except (ValueError, RuntimeError) as exc:
                        last_exc = exc
                        audit_error = str(exc)
                best=(js, graph, gt, ok, blind_solve, difficulty_audit, audit_error)
                if ok:
                    break
                feedback_parts = []
                if not graph_valid:
                    feedback_parts.append(
                        "The typed graph fails the required D4/D5 state/law/constraint/edge structure; "
                        "rebuild it around the actual multi-stage solution rather than padding labels."
                    )
                if not blueprint_valid:
                    feedback_parts.append(
                        "The solution blueprint is incomplete or its answer/choices/verification are inconsistent."
                    )
                feedback_parts.extend(presentation_issues[:6])
                if not difficulty_exact:
                    feedback_parts.append(
                        f"The response difficulty must equal {args.target_difficulty}."
                    )
                if not originality_valid:
                    feedback_parts.append(
                        f"The proposed stem is too close to a retrieved source (similarity={overlap_score:.3f}): "
                        f"{overlap_source[:240]!r}. Discard the scenario and mechanism completely."
                    )
                if not construction_match:
                    feedback_parts.append(
                        "The rendered blueprint changed the frozen construction's derived answer."
                    )
                if isinstance(blind_solve, dict):
                    feedback_parts.extend(str(x) for x in (blind_solve.get("issues") or []))
                    if not independent_solve_passes(blind_solve) and not (blind_solve.get("issues") or []):
                        feedback_parts.append(
                            "The blind solve did not confirm a well-posed, physically valid problem with one exact choice."
                        )
                if isinstance(difficulty_audit, dict):
                    feedback_parts.extend(str(x) for x in (difficulty_audit.get("issues") or []))
                    if not difficulty_audit_passes(difficulty_audit, args.target_difficulty):
                        feedback_parts.append(
                            f"The verified shortest solution was rated {difficulty_audit.get('actual_difficulty')}/5, "
                            f"below the required {args.target_difficulty}/5; familiar_template="
                            f"{difficulty_audit.get('familiar_template')}, non_obvious_decisions="
                            f"{len(difficulty_audit.get('non_obvious_decisions') or [])}."
                        )
                if audit_error:
                    feedback_parts.append(f"Independent audit could not complete: {audit_error}")
                if feedback_parts:
                    rejected = js.get("solution_blueprint") or {}
                    rejected_core = {
                        key: rejected.get(key) for key in (
                            "canonical_problem", "canonical_choices", "fixed_givens",
                            "derivation_steps", "expected_answer", "correct_choice",
                            "verification_steps",
                        )
                    }
                    if use_architecture_stage(args.target_difficulty, seed_faithful_hard):
                        rejection_feedback = (
                            "\n\nTHE PREVIOUS MECHANISM FAILED; DO NOT REPAIR OR REUSE ITS SCENARIO.\n"
                            "CHECKER FINDINGS:\n- " + "\n- ".join(feedback_parts[:8])
                            + "\nChoose a genuinely different governing mechanism. Adding steps, variables, "
                              "or algebra to the rejected mechanism does not raise difficulty."
                        )
                    elif seed_faithful_hard:
                        rejection_feedback = (
                            "\n\nTHE PREVIOUS PARALLEL FORM FAILED. Preserve the primary seed's full "
                            "dependency chain, but choose a clearer surface realization. Do not expose a "
                            "derived intermediate as a given and do not add extra requested outputs.\n"
                            "CHECKER FINDINGS:\n- " + "\n- ".join(feedback_parts[:8])
                        )
                    else:
                        rejection_feedback = (
                            "\n\nTHE PREVIOUS SOLUTION BLUEPRINT FAILED AN INDEPENDENT CHECK:\n"
                            + json.dumps(rejected_core, ensure_ascii=False)
                            + "\nCHECKER FINDINGS:\n- " + "\n- ".join(feedback_parts[:8])
                            + "\nRecompute from first principles. Do not merely change the answer or choices "
                              "to match the checker."
                        )
                elif not static_ok:
                    rejection_feedback = (
                        "\n\nTHE PREVIOUS RESPONSE FAILED mandatory graph/blueprint structure or internal-choice "
                        "consistency. Return a compact graph and include canonical_problem, exactly the required "
                        "canonical_choices, a uniquely matching correct_choice, and 2-6 verification_steps."
                    )
                if args.debug and not ok:
                    static_summary = {
                        "graph_ok": graph_ok(graph, args.target_difficulty),
                        "blueprint_ok": solution_blueprint_ok(
                            js.get("solution_blueprint"), args.target_difficulty, args.num_choices - 1
                        ),
                        "difficulty_exact": js.get("difficulty") == args.target_difficulty,
                        "blind_issues": (blind_solve or {}).get("issues", []),
                        "difficulty_issues": (difficulty_audit or {}).get("issues", []),
                        "blind_verdict": (blind_solve or {}).get("verdict"),
                        "blind_choice_match": (blind_solve or {}).get("choice_match"),
                        "rated_difficulty": (difficulty_audit or {}).get("actual_difficulty"),
                        "audit_error": audit_error,
                    }
                    print(
                        f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} "
                        f"rejection={json.dumps(static_summary, ensure_ascii=False)}",
                        file=sys.stderr,
                        flush=True,
                    )
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} retry {k+1} constraints_satisfied={bool(ok)}", file=sys.stderr, flush=True)

            if best is None:
                failures += 1
                with open(rejected_path, "a", encoding="utf-8") as rejected_f:
                    write_jsonl_line(rejected_f, {
                        "bundle_id": b.get("bundle_id"),
                        "target_difficulty": args.target_difficulty,
                        "difficulty_architecture": architecture,
                        "architecture_adherence": architecture_adherence,
                        "solution_construction": construction,
                        "rejected_architecture_history": architecture_history,
                        "pipeline_error": str(last_exc or "No renderable solution blueprint was produced"),
                    })
                print(
                    f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} REJECTED before "
                    f"blueprint rendering: {last_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if not best[3]:
                failures += 1
                rejected_js, rejected_graph, rejected_gt, _, rejected_blind, rejected_difficulty, rejected_error = best
                with open(rejected_path, "a", encoding="utf-8") as rejected_f:
                    write_jsonl_line(rejected_f, {
                        "bundle_id": b.get("bundle_id"),
                        "problem_graph": rejected_graph,
                        "graph_text": rejected_gt,
                        "solution_blueprint": rejected_js.get("solution_blueprint") or {},
                        "difficulty_architecture": architecture,
                        "architecture_adherence": architecture_adherence,
                        "solution_construction": construction,
                        "rejected_architecture_history": architecture_history,
                        "source_overlap": {
                            "score": overlap_score, "closest_source": overlap_source,
                        },
                        "target_difficulty": args.target_difficulty,
                        "blind_independent_solve": rejected_blind,
                        "difficulty_audit": rejected_difficulty,
                        "audit_error": rejected_error,
                    })
                print(
                    f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} REJECTED: no "
                    f"difficulty-{args.target_difficulty} blueprint passed",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            js, graph, gt, ok, blind_solve, difficulty_audit, audit_error = best

            if args.debug:
                dt=time.time()-t0
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row={
                "bundle_id": b.get("bundle_id"),
                "generated_graph": graph,
                "problem_graph": graph,
                "graph_text": gt,
                "skeleton_text": gt,
                "topic": js.get("topic"),
                "difficulty": args.target_difficulty,
                "solution_blueprint": js.get("solution_blueprint") or {},
                "required_topic": args.required_topic,
                "_meta": {"seed_id": b.get("seed_id"), "mode": b.get("mode"), "anchor_id": b.get("anchor_id"), "anchor_distance": b.get("anchor_distance")},
                "_diagnostics": {
                    "constraints_satisfied": bool(ok),
                    "blind_independent_solve": blind_solve,
                    "difficulty_audit": difficulty_audit,
                    "difficulty_architecture": architecture,
                    "architecture_adherence": architecture_adherence,
                    "solution_construction": construction,
                    "rejected_architecture_history": architecture_history[:-1],
                    "source_overlap": {
                        "score": overlap_score, "closest_source": overlap_source,
                    },
                    "audit_error": audit_error,
                }
            }
            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote generated graphs -> {args.out} (success={total - failures}, rejected={failures})")
    if failures:
        print(f"Wrote rejected blueprint diagnostics -> {rejected_path}")

if __name__=="__main__":
    main()
