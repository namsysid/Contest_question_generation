# D4 generation repair result

The generation path was rebuilt and iterated until one candidate passed all gates.

## Repaired architecture

1. `abstract_solutions.py` separates a verified derivation into a mechanism-only invariant and explicit
   forbidden source objects, geometry, targets, and event sequence.
2. `generate_from_abstract.py` receives no source wording. It enforces compact choices, a short stem,
   causal-chain, dimensional, limiting-case, and forbidden-feature audits.
3. `verify.py` is now genuinely blind: it receives only the generated stem and choices, not the claimed
   answer or solution.
4. `check_novelty.py` separately sees the sources and evaluates concrete student-facing similarity.
5. `promote_candidate.py` requires local checks, blind correctness/D4, and surface novelty.

Generic labels no longer override their concrete evidence. A candidate fails novelty when apparatus,
geometry, event sequence, or target form is actually shared.

## Iteration findings

- Axial-gravity variants were correct but remained source-parallel or algebraically ugly.
- A moving-contact bead problem reached D4 but was parallel to another repeated-collision source.
- An explicitly stated velocity relation destroyed difficulty and was rated D1.
- A first moving-wedge transfer used singular density and omitted wedge orientation; both were rejected.
- The final nonsingular moving-wedge problem passed blind D4 and source-surface novelty.

## Cost

The abstract-generation repair used 154,308 recorded tokens: 126,235 GPT-5 and 28,073 GPT-5-mini.
The preceding regenerated-solution-bank work used 66,552 recorded tokens, for 220,860 recorded tokens
across the complete repair sequence. One disconnected verifier request produced no usage record.

The accepted question is in `D4_ACCEPTED.md`; the machine-readable record is under
`experiments/fma_gpt5mini_2024_2026/work/d4_abstract_generation/accepted_d4_final.jsonl`.
