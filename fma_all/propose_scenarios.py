#!/usr/bin/env python3
"""Propose cheap scenario plans before paying for complete problem construction."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from circuit_lab.model_client import generate_json  # noqa: E402

p=argparse.ArgumentParser(); p.add_argument("--mechanisms",type=Path,required=True)
p.add_argument("--out",type=Path,required=True); p.add_argument("--model",default="gpt-5-mini"); a=p.parse_args()
mechanisms=[json.loads(x) for x in a.mechanisms.read_text().splitlines() if x.strip()]
item={"type":"object","additionalProperties":False,"properties":{
 "source_id":{"type":"string"},"proposal_id":{"type":"string"},"apparatus":{"type":"string"},
 "setup":{"type":"string"},"target":{"type":"string"},
 "mechanism_mapping":{"type":"array","minItems":2,"items":{"type":"string"}},
 "feasibility_constraints":{"type":"array","minItems":1,"items":{"type":"string"}},
 "expected_solution_steps":{"type":"array","minItems":2,"items":{"type":"string"}},
 "diagram_free":{"type":"boolean"},"anti_copy_explanation":{"type":"string"}},
 "required":["source_id","proposal_id","apparatus","setup","target","mechanism_mapping",
             "feasibility_constraints","expected_solution_steps","diagram_free","anti_copy_explanation"]}
schema={"type":"object","additionalProperties":False,"properties":{
 "proposals":{"type":"array","items":item}},"required":["proposals"]}
system="""Design one concise scenario plan for each abstract F=ma mechanism. Do not write questions,
choices, or solutions yet. Each scenario must be diagram-free, physically realizable, solvable from a
short self-contained stem, and recognizably different from every forbidden surface feature. Match the
mechanism's natural difficulty rather than adding complexity. Return JSON."""
result=generate_json(a.model,"MECHANISMS:\n"+json.dumps(mechanisms,ensure_ascii=False),provider="openai",
 system=system,reasoning_effort="low",max_output_tokens=8000,json_schema=schema)
out=result.get("proposals") or []
if {x["source_id"] for x in out}!={x["id"] for x in mechanisms}: raise RuntimeError("proposal ids mismatch")
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
print(json.dumps({"proposals":len(out),"out":str(a.out)},indent=2))

