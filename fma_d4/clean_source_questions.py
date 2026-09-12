#!/usr/bin/env python3
"""Repair known PDF extraction artifacts before source-solution generation."""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def strip_page_junk(text: str) -> str:
    text = re.sub(r"\s+20(?:24|25|26)$", "", text.strip())
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--infile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    output = []
    for original in rows(args.infile):
        row = copy.deepcopy(original)
        problem = row["problem"]
        problem["stem"] = strip_page_junk(problem["stem"])
        problem["choices"] = [strip_page_junk(c) for c in problem["choices"]]
        qid = row["id"]

        if qid == "2024_fnet_ma_exam_Q11":
            problem["stem"] = problem["stem"].replace("where the air pressure is 105 Pa",
                                                       "where the air pressure is 10^5 Pa")
        elif qid == "2024_fnet_ma_exam_Q21":
            problem["choices"] = [
                "(A) rho v^2 A1",
                "(B) rho v^2 A1^2/(2 A2)",
                "(C) rho v^2 A1^2/A2",
                "(D) rho v^2 A1^3/(2 A2^2)",
                "(E) rho v^2 A1^3/A2^2",
            ]
        elif qid == "2025_fnet_ma_exam_Q12":
            problem["choices"][4] = "(E) arctan(4/5)"
        elif qid == "2025_fnet_ma_exam_Q14":
            context = ("A ball launcher fires balls along the floor at the same initial speed with no "
                       "initial rotation. Each ball initially slips, then rolls without slipping. Ignore "
                       "deformation and air resistance. ")
            problem["stem"] = context + problem["stem"]
        elif qid == "2025_fnet_ma_exam_Q17":
            before = problem["stem"].split("The closest point", 1)
            problem["stem"] = (before[0].split("with potential energy", 1)[0]
                               + "with potential energy U(x,y) = (-k x^2 + y^2)/2.\n\nThe closest point"
                               + before[1])
        elif qid == "2025_fnet_ma_exam_Q18":
            problem["stem"] = re.sub(r"U\(x, y\).*?2\s*\.", "U(x,y) = kxy/2.",
                                     problem["stem"], count=1, flags=re.DOTALL)
            problem["choices"] = [
                "(A) 2 pi sqrt(m/(4k))",
                "(B) 2 pi sqrt(m/(2k))",
                "(C) 2 pi sqrt(m/k)",
                "(D) 2 pi sqrt(2m/k)",
                "(E) 2 pi sqrt(4m/k)",
            ]
        elif qid == "2025_fnet_ma_exam_Q20":
            problem["choices"][4] = "(E) After 1 minute."
        elif qid == "2026_fnet_ma_exam_Q02":
            problem["choices"] = [
                "(A) v' scales as 1/sqrt(lambda)",
                "(B) v' scales as lambda",
                "(C) v' scales as 1/lambda",
                "(D) v' scales as sqrt(lambda)",
                "(E) v' is independent of lambda",
            ]
        elif qid == "2026_fnet_ma_exam_Q17":
            problem["stem"] = re.sub(
                r"a horizontal distance.*?away from the wall\?",
                "a horizontal distance sqrt(3)L/2 away from the wall?",
                problem["stem"], flags=re.DOTALL,
            )
            problem["choices"] = [
                "(A) 0.46 sqrt(gL)", "(B) 0.51 sqrt(gL)", "(C) 0.56 sqrt(gL)",
                "(D) 0.61 sqrt(gL)", "(E) 0.66 sqrt(gL)",
            ]
        output.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    print(json.dumps({"cleaned": len(output), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
