# F=ma D4 pipeline

This is separate from `fma/run_fma_pipeline.py`. The existing pipeline remains available for fast
D1-D3 production. This directory is intentionally fail-closed for hardest-quartile F=ma generation.

## Why a separate pipeline is necessary

The 2024-2026 experimental corpus contains questions and LLM-created problem graphs, but no answer
keys or worked solutions. The current retriever calls graph neighbors `solution_exemplars`; they are
not solutions. At least one graph (2024 Q23) contained a sign error that generation inherited.

Embeddings do not teach physics. They only choose which records enter a prompt. Supplying thousands
of floating-point coordinates to a language model is useless. The useful retrieval object is a verified
solution signature containing the shortest solution, indispensable decisions, traps, and mechanism tags.

The D4 workflow is therefore:

1. Curate and verify official hard-source solutions using `verified_solutions.jsonl`.
2. Join them to text-only source questions with `build_seed_bank.py`.
3. Embed each readable `solution_signature` using `embed_signatures.py`.
4. Retrieve one primary mechanism plus diverse contrast mechanisms using `retrieve.py`.
5. Generate a synthesis, not a synonym-swapped parallel form, using `generate.py`.
6. Batch-solve candidates independently before upload. Never upload the generator's unchecked key.

## Required verified-solution fields

See `verified_solutions.example.jsonl`. A record is rejected unless it has an answer-key source,
derived answer agreeing with the official answer, at least four indispensable conceptual steps, at
least one non-obvious decision, two verification methods, and D4/D5 classification.

## Commands

```bash
python3 fma_d4/audit_sources.py \
  --questions experiments/fma_gpt5mini_2024_2026/work/source_2024_2026_text_only.jsonl \
  --out fma_d4/source_inventory.json

python3 fma_d4/build_seed_bank.py \
  --questions experiments/fma_gpt5mini_2024_2026/work/source_2024_2026_text_only.jsonl \
  --solutions fma_d4/verified_solutions.jsonl \
  --out fma_d4/verified_seed_bank.jsonl

python3 fma_d4/embed_signatures.py \
  --seed-bank fma_d4/verified_seed_bank.jsonl \
  --provider openai --model text-embedding-3-small \
  --out fma_d4/signature_embeddings.jsonl

python3 fma_d4/retrieve.py \
  --seed-bank fma_d4/verified_seed_bank.jsonl \
  --embedding-index fma_d4/signature_embeddings.jsonl \
  --anchor-id SOURCE_ID --count 3 --out fma_d4/bundles.jsonl

python3 fma_d4/generate.py \
  --bundles fma_d4/bundles.jsonl --out fma_d4/candidates.jsonl \
  --model gpt-5-mini --provider openai

python3 fma_d4/verify.py \
  --candidates fma_d4/candidates.jsonl \
  --out fma_d4/verified_candidates.jsonl \
  --model gpt-5-mini --provider openai
```

The verifier solves batches of five, keeping the added model cost bounded. Only
its accepted output is eligible for upload.

See `SOLUTION_BANK_RESULT.md` for the measured GPT-5/full-solution experiment. The accepted bank
uses blind answer verification and embeds both independently produced derivations. Graph difficulty
labels are not carried forward as truth.

The corrected generation path is:

1. `abstract_solutions.py` hides source wording and produces a mechanism-only causal invariant plus
   explicit forbidden objects, geometry, target forms, and event sequence.
2. `generate_from_abstract.py` constructs from that invariant, applies compact-choice and forbidden-term
   checks, and records dimensional, limiting-case, and causal-chain audits.
3. `verify.py` receives only the new stem and choices and solves them blindly.
4. `check_novelty.py` separately compares source surfaces; correctness verification never sees the source.
5. `promote_candidate.py` requires local, blind D4, and concrete surface-novelty gates.

`D4_ACCEPTED.md` contains the first candidate to pass the corrected end-to-end process.

`RETRIEVAL_DECISION.md` records the current policy of keeping regenerated solutions as joined
reference context instead of immediately re-embedding them.

`verified_solutions.jsonl` is deliberately not fabricated by this repository. Populate it only after
answer-key provenance and independent derivations are available.

For the explicit graph-as-solution ablation, `import_graph_seeds.py` converts enriched graphs into
provisional seeds. It reuses existing structural embeddings and still requires independent verification;
this tests whether verified algebra adds value without pretending the graphs are official solutions.
