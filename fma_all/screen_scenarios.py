#!/usr/bin/env python3
"""Reject weak scenario plans before full construction."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from circuit_lab.model_client import generate_json  # noqa: E402

def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
p=argparse.ArgumentParser(); p.add_argument("--proposals",type=Path,required=True)
p.add_argument("--source-bank",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
p.add_argument("--model",default="gpt-5-mini"); a=p.parse_args()
proposals=rows(a.proposals); sources=[{"id":r["id"],"question":r["question_text"]} for r in rows(a.source_bank)]
system="""Screen F=ma scenario plans before construction. Pass only plans that are classical-mechanics
appropriate, diagram-free, physically coherent, self-contained in principle, preserve a substantive
reasoning chain, and are not surface-parallel to any source. Shared laws are allowed. Return JSON."""
prompt="PROPOSALS:\n"+json.dumps(proposals,ensure_ascii=False)+"\nSOURCES:\n"+json.dumps(sources,ensure_ascii=False)+'''\nReturn {"assessments":[{"proposal_id":"...","pass":true,"closest_source_id":"...|NONE","physics_risk":"...","copy_risk":"...","required_fix":"..."}]}'''
result=generate_json(a.model,prompt,provider="openai",system=system,reasoning_effort="low",max_output_tokens=5000)
assessment={x["proposal_id"]:x for x in result.get("assessments",[])}
out=[{**x,"screen":assessment.get(x["proposal_id"],{"pass":False,"required_fix":"missing assessment"})} for x in proposals]
a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
print(json.dumps({"screened":len(out),"passed":sum(x["screen"].get("pass") is True for x in out),"out":str(a.out)},indent=2))

