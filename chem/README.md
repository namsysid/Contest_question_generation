# USNCO Chemistry Question Generator

This folder contains the chemistry-only USNCO generation system.

Source data is kept in `data/usnco_questions.jsonl`. The first block of rows is still placeholder metadata, but transcribed rows begin later in the file. `generate_usnco.py --prepare-only` filters those transcribed rows and converts the nested USNCO format into the raw format expected by the graph-first `src` pipeline.

The chem flow uses:

```text
src/01_enrich_problem_schema_paper-2.py
src/02_embed_and_index.py
src/generate.py
```

It keeps the six-question range interface for USNCO. Builds are range-scoped by default: `--range 1-6` prepares, enriches, embeds, and generates only from original question numbers 1 through 6.

Each range has its own reusable cache under:

```text
chem/range_runs/<range>/
```

For example, rebuilding `7-12` will not overwrite the cached enrichment/embeddings for `1-6`.

## Ranges

Generation is always done in six-question buckets:

```text
1-6, 7-12, 13-18, 19-24, 25-30,
31-36, 37-42, 43-48, 49-54, 55-60
```

List the mapped topics:

```bash
python3 chem/generate_usnco.py --list-ranges
```

Generate a bucket:

```bash
python3 chem/run_usnco_all_in_one.py --range 1-6
```

Rebuild enrichment and embeddings explicitly:

```bash
python3 chem/run_usnco_all_in_one.py --rebuild --range 1-6
```

Inspect the converted raw rows without running models:

```bash
python3 chem/generate_usnco.py --prepare-only
```

Output defaults to:

```text
chem/outputs/usnco_1-6_generated.jsonl
```

Generation appends to the output file by default, so rerunning a range adds new rows without replacing earlier generations. Use `--overwrite-output` only when you intentionally want to replace that file.

Add four more generated questions to an existing `1-6` output:

```bash
python3 chem/run_usnco_all_in_one.py --range 1-6 --num 4
```

Upload generated questions to MongoDB:

```bash
python3 chem/upload_chem_questions_to_mongodb.py chem/outputs/usnco_1-6_generated.jsonl --dry-run
python3 chem/upload_chem_questions_to_mongodb.py chem/outputs/usnco_1-6_generated.jsonl
```

The uploader checks for overlap before inserting. It skips documents already present by `DedupeKey` (USNCO + question range + normalized question text) or `source_id`. Use `--allow-duplicates` only if you want to bypass that check.

The uploader uses chem-specific `.env` variables:

```text
MONGODB_URI
CHEM_DB_NAME
CHEM_COLLECTION_NAME
```

Uploaded documents include:

```text
Question
Choices
Answer
AnswerText
QuestionRange
QuestionNumber
RangeTitle
Year
ExamType
ExamCode
SeedQuestionNumber
Solution
SkeletonText
DedupeKey
```

By default this uses OpenAI API models through `src/ollama_client.py`:

```text
enrichment: gpt-4o
embeddings: text-embedding-3-large
generation: gpt-4o
```

Set `OPENAI_API_KEY` in `.env` or your shell. Use `--enrich-model`, `--embed-model`, and `--gen-model` to override defaults.
