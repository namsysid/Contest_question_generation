# Science Olympiad Optics B pipeline

This is a separate graph-first RAG pipeline for the **2025** Division B Optics event. Optics is not a 2026 Division B
event, so the season label is intentional. The calibration PDFs came from three public 2025 test banks linked by the
[official Science Olympiad Optics page](https://www.soinc.org/optics-b): Bulls SO, Purdue, and USC invitationals.

The source extractor excludes build/laser-shoot scoring, ray drawings, eye-labeling, and questions that depend on an
unavailable figure, spectrum, graph, or preceding question. The generation pipeline uses question and reasoning-graph
embeddings, source retrieval, graph-first planning, arithmetic verification, semantic novelty thresholds, a judge
blind to the stored key, checkpoint/resume, and a source-calibrated whole-bank audit.

## Reproduce

```bash
python3 -m src.optics.ingest \
  --source-dir science_olympiad/optics_b/sources \
  --out science_olympiad/optics_b/new_run/corpus/items.jsonl \
  --provider openai --model gpt-5.1

python3 -m src.optics.pipeline \
  --source-dir science_olympiad/optics_b/sources \
  --work-dir science_olympiad/optics_b/new_run \
  --count 24 --provider openai \
  --generation-model gpt-5.1 \
  --embedding-model text-embedding-3-small
```

The checked-in final bank is in `final/generated/items_final.jsonl`. Its independent reports and set audits are in
`final/validation/`. Nothing in this pipeline uploads to MongoDB or another external database.
