#!/usr/bin/env python3
"""Construct one solved candidate from an approved scenario and its mechanism."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from circuit_lab.model_client import generate_json  # noqa: E402

def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
p=argparse.ArgumentParser(); p.add_argument("--mechanisms",type=Path,required=True)
p.add_argument("--screened",type=Path,required=True); p.add_argument("--proposal-id",required=True)
p.add_argument("--out",type=Path,required=True); p.add_argument("--model",default="gpt-5-mini"); a=p.parse_args()
proposal=next(x for x in rows(a.screened) if x["proposal_id"]==a.proposal_id)
if proposal.get("screen",{}).get("pass") is not True: raise RuntimeError("proposal did not pass screening")
mechanism=next(x for x in rows(a.mechanisms) if x["id"]==proposal["source_id"])
schema={"type":"object","additionalProperties":False,"properties":{
 "id":{"type":"string"},"question":{"type":"string"},"choices":{"type":"object","additionalProperties":False,
 "properties":{k:{"type":"string"} for k in "ABCDE"},"required":list("ABCDE")},
 "answer":{"type":"string","enum":list("ABCDE")},"solution":{"type":"string"},
 "estimated_difficulty":{"type":"integer","minimum":1,"maximum":4},
 "conceptual_steps":{"type":"array","minItems":2,"items":{"type":"string"}},
 "dimensional_check":{"type":"string"},"limiting_check":{"type":"string"}},
 "required":["id","question","choices","answer","solution","estimated_difficulty","conceptual_steps","dimensional_check","limiting_check"]}
system="""Write one self-contained, diagram-free F=ma multiple-choice problem from an approved scenario.
Use only the supplied scenario and abstract mechanism; never reconstruct the source surface. Verify the
physics and solve exactly before creating five compact plausible choices. Match the source's natural
difficulty. Do not add impossible repeated events or unsupported assumptions. Return strict JSON."""
payload={"mechanism":mechanism,"approved_scenario":{k:v for k,v in proposal.items() if k!="screen"}}
result=generate_json(a.model,json.dumps(payload,ensure_ascii=False),provider="openai",system=system,
 reasoning_effort="medium",max_output_tokens=8000,json_schema=schema)
result["_meta"]={"source_id":proposal["source_id"],"proposal_id":proposal["proposal_id"],"constructor_model":a.model,"local_rejections":[]}
if len(result["question"].split())>180: result["_meta"]["local_rejections"].append("stem exceeds 180 words")
if any(len(v.split())>14 for v in result["choices"].values()): result["_meta"]["local_rejections"].append("choice exceeds 14 words")
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,ensure_ascii=False)+"\n")
print(json.dumps({"proposal_id":a.proposal_id,"out":str(a.out),"local_rejections":result["_meta"]["local_rejections"]},indent=2))
