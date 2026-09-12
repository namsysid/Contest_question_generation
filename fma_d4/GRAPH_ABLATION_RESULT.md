# Graph-as-solution D4 ablation

One bounded pilot was run with GPT-5 mini. It reused the existing
`text-embedding-3-large` structural embeddings; no new embedding call was made.

Retrieved source graphs:

- 2026 Q22: asymmetric loading and frictional torque
- 2024 Q09: piecewise energy and a threshold check
- 2026 Q15: translation-rotation coupling and impulse

## Generated question

A uniform solid sphere, with moment of inertia `I = (2/5)mR^2`, slides without
rotation at `14.0 m/s` on a dry surface with kinetic friction coefficient `0.30`.
After a dry run of length `L`, it reaches frictionless ice. What minimum dry
length is required for it to attain pure rolling before reaching the ice?

- A. 8.16 m
- B. 16.3 m
- C. 28.7 m
- D. 46.7 m
- E. 58.0 m

Generated answer: **B**. The answer is correct: while slipping,
`a=-mu g`, `alpha=mu m g R/I`, and `v=omega R` occurs at
`t=2v0/(7 mu g)`. Integrating translation gives
`L=(12/49)v0^2/(mu g)=16.3 m`.

## Independent verdict

- Physics valid and uniquely answered: yes
- Correct choice: B
- Difficulty: **D1**
- Conceptual deductions: 3
- Non-obvious modeling decisions: 0
- Familiar template: yes
- Source copy: no
- D4 gate: **rejected**

The graph representation was therefore sufficient for correctness here, but
not for preserving hard-source depth. The generator selected one familiar
translation-rotation mechanism and attached a superficial ice-boundary
threshold; it did not preserve the indispensable dependency chains of the
other retrieved graphs.

Token usage was 11,248 for generation and 4,257 for independent verification,
15,505 total. No further calls were made.
