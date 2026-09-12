# Division B Circuit Lab pipeline

This pipeline generates written-test questions in the style and scope of Division B Circuit Lab. Source PDF text is
treated as untrusted reference material. Hands-on task generation is intentionally excluded for now.

## Stages

1. `ingest`: extract four-choice questions and multipart short-answer/numeric problems, retaining points and page images.
2. `enrich`: label topics, skills, difficulty, timing, misconceptions, and a reasoning graph.
3. `embed`: independently embed question text and graph structure and select density anchors by response format.
4. `retrieve`: create format/topic-constrained dual-retrieval bundles with MMR diversity.
5. `plan`: generate a novel reasoning graph from each retrieved structural neighborhood.
6. `generate`: realize each graph as an original MCQ or constructed-response item using question-style exemplars.
7. `validate`: enforce schema/scope rules and optionally have a blind model solve each item independently.

Run the complete pipeline from the repository root:

```bash
python3 -m src.circuit_lab.cli pipeline \
  --input "/path/to/Circuit Lab B-TEST.pdf" --level regional --count 25
```

Or run individual stages:

```bash
python3 -m src.circuit_lab.cli ingest \
  --input "/path/to/Circuit Lab B-TEST.pdf" \
  --out science_olympiad/circuit_lab_b/corpus/items.jsonl

python3 -m src.circuit_lab.cli enrich \
  --input science_olympiad/circuit_lab_b/corpus/items.jsonl \
  --out science_olympiad/circuit_lab_b/enriched/items.jsonl

python3 -m src.circuit_lab.cli embed \
  --input science_olympiad/circuit_lab_b/enriched/items.jsonl \
  --question-out science_olympiad/circuit_lab_b/enriched/question_embeddings.jsonl \
  --structure-out science_olympiad/circuit_lab_b/enriched/structure_embeddings.jsonl \
  --anchors-out science_olympiad/circuit_lab_b/enriched/anchors.jsonl

python3 -m src.circuit_lab.cli retrieve \
  --enriched science_olympiad/circuit_lab_b/enriched/items.jsonl \
  --question-index science_olympiad/circuit_lab_b/enriched/question_embeddings.jsonl \
  --structure-index science_olympiad/circuit_lab_b/enriched/structure_embeddings.jsonl \
  --anchors science_olympiad/circuit_lab_b/enriched/anchors.jsonl \
  --out science_olympiad/circuit_lab_b/enriched/retrieval_bundles.jsonl

python3 -m src.circuit_lab.cli plan \
  --bundles science_olympiad/circuit_lab_b/enriched/retrieval_bundles.jsonl \
  --out science_olympiad/circuit_lab_b/generated/reasoning_plans.jsonl --level regional

python3 -m src.circuit_lab.cli generate \
  --bundles science_olympiad/circuit_lab_b/enriched/retrieval_bundles.jsonl \
  --plans science_olympiad/circuit_lab_b/generated/reasoning_plans.jsonl \
  --out science_olympiad/circuit_lab_b/generated/items.jsonl

python3 -m src.circuit_lab.cli validate \
  --input science_olympiad/circuit_lab_b/generated/items.jsonl \
  --out science_olympiad/circuit_lab_b/validation/item_reports.jsonl \
  --model qwen2.5:7b-instruct
```

Supported topic labels are `history`, `electrostatics`, `dc`, `ac_household`, `quantities_ohms_law`, `magnetism`,
`controls_safety`, `circuit_analysis`, and `led`. LED generation is rejected below State level unless explicitly enabled.

Source tests without answer keys are usable as style exemplars. Generated items always require answers and worked
solutions. Add source keys later during corpus preparation if answer-aware retrieval is desired.
