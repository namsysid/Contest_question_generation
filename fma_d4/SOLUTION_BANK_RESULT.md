# Regenerated solution bank result

## Configuration

- Source derivation: GPT-5, medium reasoning, batches of four.
- Independent verification: GPT-5 mini, medium reasoning, blind to the first answer and solution.
- Embedding: `text-embedding-3-small`, one batch.
- Source scope: eight clean, text-only 2024-2026 candidates selected for likely difficulty.

The verifier was changed to be genuinely blind after an earlier verifier copied a wrong claimed
answer. Usage logging now creates its parent directory, preventing completed responses from being lost.

## Accepted and embedded solutions

Six of eight source solutions had matching independently derived answers:

- 2024 Q09: D1
- 2024 Q23: D2
- 2024 Q24: D2
- 2025 Q16: D1
- 2026 Q09: D3
- 2026 Q10: D4

2025 Q20 and 2026 Q22 were excluded because the two solvers disagreed. In particular, GPT-5's
2025 Q20 orbital-coordinate solution was wrong; blind verification rejected its answer.

The accepted records contain the full first derivation, blind derivation, shortest steps, decisions,
pitfalls, and normalized blind difficulty. Their combined solution signatures were re-embedded.

## End-to-end candidate

The D4 seed (2026 Q10) produced a hemisphere-versus-cylinder gravitational-field matching problem.
Its answer `H/a = 8/15` was correct and unique. Independent grading found:

- physics valid: yes
- difficulty: D3
- conceptual deductions: 4
- non-obvious decisions: 0
- familiar template: yes
- D4 gate: rejected

This is better than the prior invalid and D1 candidates, but is too visibly parallel to its source and
still not hardest-quartile practice quality.

## Cost and conclusion

The recorded source-bank experiment used 66,552 tokens across nine API calls, including two verifier
responses that hit their output ceiling. The embedding call used 2,026 of those tokens.

Full solutions and re-embedding improve correctness and expose inflated graph difficulty. They do not
by themselves create D4 questions. This sample contained only one independently rated D4 source, so
retrieval lacked a sufficiently broad hard-mechanism bank. The next useful work is to obtain more truly
D4 verified seeds and add a semantic novelty gate; regenerating routine sources would not help.
