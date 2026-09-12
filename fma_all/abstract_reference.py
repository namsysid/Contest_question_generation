#!/usr/bin/env python3
"""Convert verified solution references into surface-free transferable mechanisms."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


p = argparse.ArgumentParser()
p.add_argument("--bank", type=Path, required=True)
p.add_argument("--ids", nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--model", default="gpt-5-mini")
a = p.parse_args()
wanted = set(a.ids)
selected = [r for r in rows(a.bank) if r["id"] in wanted]
if {r["id"] for r in selected} != wanted:
    raise RuntimeError("requested verified reference is missing")
payload = [{
    "id": r["id"], "source_difficulty": r["difficulty"],
    "solution": r["solution_text"], "steps": r["shortest_solution_steps"],
    "decisions": r["non_obvious_decisions"], "mechanisms": r["mechanism_tags"],
    "pitfalls": r["pitfalls"], "blind_check": (r.get("independent_verdict") or {}).get("independent_solution"),
    "source_surface": r["question_text"],
} for r in selected]
item = {"type":"object","additionalProperties":False,"properties":{
    "id":{"type":"string"}, "source_difficulty":{"type":"integer","minimum":1,"maximum":4},
    "core_invariant":{"type":"string"},
    "dependency_chain":{"type":"array","minItems":2,"items":{"type":"string"}},
    "essential_decisions":{"type":"array","items":{"type":"string"}},
    "equations":{"type":"array","items":{"type":"string"}},
    "transferable_roles":{"type":"array","minItems":2,"items":{"type":"string"}},
    "forbidden_surface_features":{"type":"array","minItems":2,"items":{"type":"string"}},
},"required":["id","source_difficulty","core_invariant","dependency_chain","essential_decisions",
              "equations","transferable_roles","forbidden_surface_features"]}
schema={"type":"object","additionalProperties":False,"properties":{
    "mechanisms":{"type":"array","items":item}},"required":["mechanisms"]}
system="""Extract reusable F=ma reasoning structures from verified solutions. Preserve the actual
dependency chain and indispensable model choices. Remove all source-specific objects, geometry,
numbers, wording, event order, and requested observable from the transferable description, while
listing those recognizable surface features as forbidden. Do not invent extra physics. Return JSON."""
result=generate_json(a.model,"Extract:\n"+json.dumps(payload,ensure_ascii=False),provider="openai",
                     system=system,reasoning_effort="medium",max_output_tokens=10000,json_schema=schema)
output=result.get("mechanisms") or []
if {r["id"] for r in output} != wanted:
    raise RuntimeError("abstraction output ids differ from requested ids")
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in output))
print(json.dumps({"abstracted":len(output),"out":str(a.out)},indent=2))

