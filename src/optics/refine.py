from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.circuit_lab.model_client import embed_texts

from .common import graph_text, read_jsonl, write_jsonl
from .pipeline import create_item, normalized, retrieve_exemplars


SPECS = [
    {
        "topic": "optical_instruments", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "multiply objective and eyepiece magnifications for a compound microscope",
        "novel_context": "student changes microscope eyepieces",
        "fixed_givens": ["objective magnification is 10x", "eyepiece magnification is 15x",
                          "total magnification equals objective magnification times eyepiece magnification"],
        "verification": {"expected_answer": "150x", "expected_value": 150, "expression": "10*15",
                         "calculation": "Total magnification = 10 x 15 = 150x."},
    },
    {
        "topic": "refraction_tir", "difficulty": 2, "response_type": "numeric",
        "target_skill": "calculate the critical angle for light leaving a higher-index medium",
        "novel_context": "glass light guide surrounded by air",
        "fixed_givens": ["light travels inside glass with refractive index 1.50 toward air of index 1.00",
                          "the critical-angle condition is sin(theta_c)=n_air/n_glass",
                          "sin inverse of 0.667 is 41.8 degrees"],
        "verification": {"expected_answer": "41.8 degrees", "expected_value": 41.8, "expression": "41.8",
                         "calculation": "sin(theta_c)=1.00/1.50=0.667, so theta_c=41.8 degrees."},
    },
    {
        "topic": "color_spectra", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "find transmitted color by intersecting subtractive-filter passbands",
        "novel_context": "overlapping cyan and yellow theater-light filters",
        "fixed_givens": ["white light contains red, green, and blue", "an ideal cyan filter transmits green and blue",
                          "an ideal yellow filter transmits red and green", "the filters are placed in series"],
        "verification": {"expected_answer": "green", "calculation": "Only green is transmitted by both filters."},
    },
    {
        "topic": "light_photons", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "compare photon energies using the inverse wavelength relationship",
        "novel_context": "red and blue calibration lasers",
        "fixed_givens": ["red laser wavelength is 600 nm", "blue laser wavelength is 450 nm",
                          "photon energy is inversely proportional to wavelength"],
        "verification": {"expected_answer": "a blue photon has 1.33 times the energy of a red photon",
                         "expected_value": 1.3333333333333333, "expression": "600/450",
                         "calculation": "E_blue/E_red=lambda_red/lambda_blue=600/450=1.33."},
    },
    {
        "topic": "refraction_tir", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "apply Snell's law to calculate a refracted angle",
        "novel_context": "laser entering an acrylic window",
        "fixed_givens": ["light travels from air of index 1.00 into acrylic of index 1.50",
                          "incidence angle is 30.0 degrees from the normal", "sin(30.0 degrees)=0.500",
                          "sin inverse of 0.333 is 19.5 degrees", "n1 sin(theta1)=n2 sin(theta2)"],
        "verification": {"expected_answer": "19.5 degrees", "expected_value": 19.5, "expression": "19.5",
                         "calculation": "sin(theta2)=1.00(0.500)/1.50=0.333, so theta2=19.5 degrees."},
    },
    {
        "topic": "physical_optics", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "calculate Brewster's angle and identify the polarization consequence",
        "novel_context": "reducing glare from a glass display",
        "fixed_givens": ["light travels from air of index 1.00 toward glass of index 1.50",
                          "tan(theta_B)=n2/n1", "arctan(1.50)=56.3 degrees",
                          "at Brewster's angle the reflected beam is linearly polarized"],
        "verification": {"expected_answer": "56.3 degrees; reflected light is linearly polarized",
                         "expected_value": 56.3, "expression": "56.3",
                         "calculation": "theta_B=arctan(1.50/1.00)=56.3 degrees; the reflection is polarized."},
    },
    {
        "topic": "lenses_images", "difficulty": 3, "response_type": "multiple_choice",
        "target_skill": "use the thin-lens equation and magnification in sequence",
        "novel_context": "projecting an illuminated arrow",
        "fixed_givens": ["converging lens focal length is +8.0 cm", "object distance is +12.0 cm",
                          "1/f=1/do+1/di", "m=-di/do", "positive image distance means a real image"],
        "verification": {"expected_answer": "image distance +24 cm and magnification -2.0",
                         "expected_value": 24, "expression": "1/(1/8-1/12)",
                         "calculation": "di=24 cm, then m=-24/12=-2.0, so the image is real, inverted, and twice as tall."},
    },
    {
        "topic": "human_eye_vision", "difficulty": 3, "response_type": "numeric",
        "target_skill": "calculate corrective-lens power from an eye's near point",
        "novel_context": "reading correction for a farsighted student",
        "fixed_givens": ["student near point is 50 cm", "book is held 25 cm from a thin corrective lens close to the eye",
                          "lens must form a virtual image at the 50 cm near point", "do=+0.25 m and di=-0.50 m",
                          "1/f=1/do+1/di and power P=1/f in diopters"],
        "verification": {"expected_answer": "+2.0 D converging lens", "expected_value": 2,
                         "expression": "1/0.25+1/(-0.50)",
                         "calculation": "P=1/f=1/0.25-1/0.50=+2.0 D, so the corrective lens converges."},
    },
    {
        "topic": "light_photons", "difficulty": 3, "response_type": "multiple_choice",
        "target_skill": "calculate photon energy from an emission-line wavelength",
        "novel_context": "blue-green hydrogen emission line",
        "fixed_givens": ["wavelength is 486 nm", "h=6.63e-34 J s", "c=3.00e8 m/s",
                          "1 nm=1e-9 m", "photon energy E=hc/lambda"],
        "verification": {"expected_answer": "4.09e-19 J", "expected_value": 4.0925925925925924e-19,
                         "expression": "6.63e-34*3.00e8/(486e-9)",
                         "calculation": "486 nm=4.86e-7 m; E=(6.63e-34)(3.00e8)/(4.86e-7)=4.09e-19 J."},
    },
    {
        "topic": "lenses_images", "difficulty": 2, "response_type": "multiple_choice",
        "target_skill": "combine thin-lens powers and convert total power to focal length",
        "novel_context": "two thin lenses mounted together in a camera adapter",
        "fixed_givens": ["lens powers are +5.0 D and -2.0 D", "thin lenses are in contact",
                          "powers in contact add", "P=1/f with f in meters"],
        "verification": {"expected_answer": "+3.0 D, focal length +0.333 m, converging",
                         "expected_value": 0.3333333333333333, "expression": "1/(5-2)",
                         "calculation": "P_total=+5.0-2.0=+3.0 D; f=1/3=+0.333 m, so the pair converges."},
    },
    {
        "topic": "light_photons", "difficulty": 3, "response_type": "multiple_choice",
        "target_skill": "combine the Rydberg relation with photon energy",
        "novel_context": "hydrogen transition from n=3 to n=2",
        "fixed_givens": ["hydrogen electron falls from n=3 to n=2", "R=1.097e7 per meter",
                          "1/lambda=R(1/n_low^2-1/n_high^2)", "E=hc/lambda",
                          "h=6.63e-34 J s", "c=3.00e8 m/s"],
        "verification": {"expected_answer": "3.03e-19 J", "expected_value": 3.0307125e-19,
                         "expression": "6.63e-34*3.00e8*1.097e7*(1/4-1/9)",
                         "calculation": "1/lambda=R(1/4-1/9)=1.524e6 m^-1. Then E=hc/lambda=hc(1/lambda)=3.03e-19 J."},
    },
]


def plan_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    nodes = [{"id": f"g{i}", "type": "Given", "label": value}
             for i, value in enumerate(spec["fixed_givens"], 1)]
    nodes += [{"id": "law", "type": "Law", "label": spec["target_skill"]},
              {"id": "target", "type": "Target", "label": "determine requested result"}]
    return {**json.loads(json.dumps(spec)),
            "reasoning_graph": {"nodes": nodes,
                "edges": ([{"src": node["id"], "dst": "law", "type": "supports"} for node in nodes[:-2]] +
                          [{"src": "law", "dst": "target", "type": "derived_from"}])},
            "distractor_mechanisms": ["invert a ratio", "use only one stage or filter", "fail a unit conversion"],
            "audit": {"valid": True, "method": "human-vetted specification"}}


def choose_anchor(corpus: list[dict[str, Any]], topic: str, response_type: str, offset: int) -> dict[str, Any]:
    candidates = [row for row in corpus if row.get("response_type") == response_type and topic in row.get("topics", [])]
    if not candidates:
        candidates = [row for row in corpus if row.get("response_type") == response_type]
    if not candidates:
        raise RuntimeError(f"No source anchor for {topic}/{response_type}")
    return candidates[offset % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate targeted Optics coverage replacements")
    parser.add_argument("--root", default="science_olympiad/optics_b")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()
    root = Path(args.root)
    corpus = read_jsonl(root / "production" / "corpus" / "items.jsonl")
    q_rows = read_jsonl(root / "production" / "enriched" / "question_embeddings.jsonl")
    s_rows = read_jsonl(root / "production" / "enriched" / "structure_embeddings.jsonl")
    source_qmat = normalized([row["embedding"] for row in q_rows])
    existing: list[dict[str, Any]] = []
    for directory in ("production", "shard_b", "shard_c"):
        path = root / directory / "generated" / "items.jsonl"
        if path.exists():
            existing.extend(read_jsonl(path))
    items_path = root / "targeted" / "generated" / "items.jsonl"
    reports_path = root / "targeted" / "validation" / "item_reports.jsonl"
    plans_path = root / "targeted" / "generated" / "reasoning_plans.jsonl"
    items = read_jsonl(items_path) if items_path.exists() else []
    reports = read_jsonl(reports_path) if reports_path.exists() else []
    plans = read_jsonl(plans_path) if plans_path.exists() else []
    prior = existing + items
    prior_vectors = embed_texts(args.embedding_model, [row["prompt"] for row in prior], provider=args.provider)
    for index, spec in enumerate(SPECS):
        if index < len(items):
            continue
        plan = plan_from_spec(spec)
        anchor = choose_anchor(corpus, spec["topic"], spec["response_type"], index)
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        item, report, vector = create_item(anchor, exemplars, plan, prior, source_qmat, corpus, prior_vectors,
                                           args.embedding_model, args.model, args.provider,
                                           48000 + index * 1000, 101 + index)
        items.append(item)
        reports.append(report)
        plans.append(plan)
        prior.append(item)
        prior_vectors.append(vector)
        write_jsonl(items_path, items)
        write_jsonl(reports_path, reports)
        write_jsonl(plans_path, plans)
        print(f"Accepted targeted {index + 1}/{len(SPECS)}: {spec['topic']}", flush=True)


if __name__ == "__main__":
    main()
