# Science Olympiad Machines B pipeline

This event-specific pipeline uses current 2026 Machines B test/key pairs, dual question and reasoning-structure
embeddings, graph-first planning, semantic novelty checks, independent blind solving, finalist copyediting, and both
advisory and reproducible bank audits.

The checked run is in `final/`. Its deliverable is `final/generated/items_final.jsonl`; the larger alternate-item pool
is `final/generated/items_curated.jsonl`.

From the repository root, a clean OpenAI-backed run is:

```bash
python3 -m src.machines.pipeline \
  --source-dir science_olympiad/machines_b/sources \
  --work-dir science_olympiad/machines_b/new_run \
  --count 30 \
  --provider openai \
  --generation-model gpt-5.1 \
  --embedding-model text-embedding-3-small
```

For the curated production workflow used here:

```bash
python3 -m src.machines.refine --run-dir science_olympiad/machines_b/final --provider openai
python3 -m src.machines.finalize --run-dir science_olympiad/machines_b/final --provider openai
python3 -m src.machines.polish --run-dir science_olympiad/machines_b/final --provider openai
python3 -m src.machines.finalize --run-dir science_olympiad/machines_b/final --provider openai
python3 -m src.machines.audit_bank --run-dir science_olympiad/machines_b/final --provider openai
```

Set `OPENAI_API_KEY` in `.env`. The shared model client also supports `--provider ollama`; no MongoDB upload is part of
this pipeline.
