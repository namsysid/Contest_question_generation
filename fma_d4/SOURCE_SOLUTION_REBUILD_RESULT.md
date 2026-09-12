# 2024-2026 source-solution rebuild result

- Input: 26 extracted diagram-free F=ma questions.
- Extraction repairs: corrected page-boundary leakage, shared Q13-14 context, powers, fractions,
  radicals, and `10^5 Pa` before solving.
- Reference bank: 25 accepted records.
- Verification provenance: 19 blind GPT-5/GPT-5-mini answer agreements and 6 explicit manual
  equation reviews after verifier service failures.
- Quarantined: 2025 Q23 because the solvers disagreed about uncertainty propagation and answer choice.
- Difficulty distribution: 6 D1, 7 D2, 8 D3, 4 D4.
- Logged API usage for this rebuild: 102,995 tokens. Interrupted requests with no returned usage record
  make the true billed total potentially higher.

The accepted solutions were **not re-embedded**. Existing question embeddings may retrieve source IDs;
the pipeline should then join those IDs to the full readable structured solution records. Exhaustive
generation should simply iterate all 25 records without retrieval.

Machine-readable bank:
`experiments/fma_gpt5mini_2024_2026/work/source_solution_rebuild/reference_solution_bank.jsonl`

Human-readable bank: `SOURCE_SOLUTION_BANK.md`.
