from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODULES = {
    "pipeline": "src.circuit_lab.pipeline",
    "ingest": "src.circuit_lab.ingest", "enrich": "src.circuit_lab.enrich",
    "embed": "src.circuit_lab.embed", "retrieve": "src.circuit_lab.retrieve",
    "plan": "src.circuit_lab.plan", "generate": "src.circuit_lab.generate",
    "validate": "src.circuit_lab.validate",
}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in MODULES:
        command = [sys.executable, "-m", MODULES[sys.argv[1]], *sys.argv[2:]]
        return subprocess.run(command, cwd=Path(__file__).parents[2]).returncode
    parser = argparse.ArgumentParser(description="Division B Circuit Lab generation pipeline")
    parser.add_argument("stage", choices=MODULES)
    parser.parse_args()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
