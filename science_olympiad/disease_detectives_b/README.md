# Science Olympiad Disease Detectives B graph-RAG pilot

This directory contains a 10-item pilot calibrated to public previous-season material linked from the official
Disease Detectives event page. The official rules manual was deliberately neither downloaded nor used. The official
page listed 2027 current-season resources as “Coming soon” when accessed, so this is a 2027-targeted pilot, not a claim
of compliance with unpublished 2027 rules.

The canonical release artifact is `generated/gpt-5-mini/items_final.jsonl`.
Each item has a graph-first reasoning plan and retrieval anchor in its `generation` metadata. The raw model checkpoint
is retained as `generated/gpt-5-mini/items.jsonl`. The prior GPT-5.1 bank has not been removed; its complete generated
and validation artifacts are preserved under `historical/gpt-5.1/`.

The corpus has 68 extracted keyed questions; 63 pass the current deterministic text-safe source filter used for retrieval.
Question and reasoning-graph embeddings are stored separately. Generation checkpoints after every accepted item and
resumes from existing JSONL artifacts. The final bank passes deterministic validation, embedding novelty thresholds,
independent blind solving, and a holistic bank audit. See `source_manifest.json` for provenance and rights notes.

Run from the repository root:

```bash
python3 -m src.disease_detectives.pipeline --count 10 --provider openai --generation-model gpt-5-mini
python3 -m src.disease_detectives.finalize --provider openai --model gpt-5-mini
python3 -m pytest -q tests/test_disease_detectives_pipeline.py
```

The production run uses exact model `gpt-5-mini` for graph-first planning, item generation, blind solving, and holistic
bank audit, plus `text-embedding-3-small` for dual retrieval and novelty measurement. Raw API usage is checkpointed in
`validation/gpt-5-mini/model_usage.jsonl`; it records token usage but not dollar billing, so no USD cost is inferred. The first
attempt to ingest the 2026 workbook was stopped after an output-cap failure rather than repeating an oversized call.
