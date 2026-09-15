# Dynamic Planet B — Earth's Fresh Waters

This directory contains a graph-first RAG pilot for the 2027 Division B event topic. It uses three public
Fresh Waters practice test/key pairs linked from the Science Olympiad event page. It does **not** ingest or quote
the official rules manual. Consequently, this is a practice-question pipeline, not an authoritative rules summary.

Pipeline stages:

1. Paired PDF ingest and question/key provenance
2. Model enrichment into minimal reasoning graphs
3. Separate question-text and graph embedding indices
4. Topic-balanced dual retrieval
5. Graph-first reasoning-plan generation
6. Question generation with per-item checkpoints
7. Deterministic novelty/schema checks and model-based blind judging

Run:

```bash
python -m src.dynamic_planet.model_pipeline --provider openai --generation-model gpt-5-mini \
  --embedding-model text-embedding-3-small --enrich-limit 30 --count 10
```

See `source_manifest.json` for exact URLs and SHA-256 hashes. The USGS Water Science School references listed
there are authoritative open references for factual cross-checking; the generation corpus itself consists of the
paired public practice materials.

## Canonical model-generated bank

The canonical artifacts are a completed production model run, not the earlier agent-authored pilot. `gpt-5-mini`
generated 30 source reasoning graphs, all 10 graph-first plans, all 10 questions, 10 independent blind-solve reports,
and the whole-bank audit. `text-embedding-3-small` produced separate 30-row question-text and reasoning-structure
indices and performed per-draft novelty checks. All 10 questions passed deterministic checks, embedding novelty,
blind solving, and the whole-bank audit. Every configured topic family occurs at least once.

- `generated/items.jsonl`: canonical 10-item model-generated bank
- `generated/reasoning_plans.jsonl`: checkpointed `gpt-5-mini` plans
- `enriched/question_embeddings.jsonl`: real question-text embeddings
- `enriched/structure_embeddings.jsonl`: real model-reasoning-structure embeddings
- `validation/item_reports.jsonl`: independent blind-solving and novelty evidence
- `validation/run_summary.json`: final status and recorded usage totals
- `validation/model_usage.jsonl`: per-response model and token provenance

The original heuristic/feature-hash pilot is preserved unchanged under `provisional/` and is not a canonical output.
The superseded GPT-5.1 production run is preserved under `historical/gpt-5.1/`, while the complete promoted
`gpt-5-mini` run also remains under `runs/gpt-5-mini/`. Rejected plans and audit-driven replacements are retained
in the promoted run's validation directory.
The runner checkpoints each enrichment, plan, accepted item, and report, and resumes without regenerating accepted
work. Bounded retries record concrete failures in `validation/generation_failures.jsonl`.
