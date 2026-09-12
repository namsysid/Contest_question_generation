# General F=ma generation pipeline

This is the production-oriented counterpart to `fma_d4/`. It generates one
diagram-free practice question from every verified source mechanism, regardless
of difficulty. Difficulty is measured and retained as metadata; it is not an
acceptance gate.

A question is retained only when:

- a blind solver independently selects the constructor's answer;
- the blind solver finds the physics valid and the answer unique;
- no source has the same apparatus, geometry, event sequence, or target form;
- deterministic stem, answer-choice, and forbidden-feature checks pass.

The constructor and graders default to `gpt-5-mini`. Run artifacts are written
under `experiments/fma_gpt5mini_2024_2026/work/fma_all_generation/`.

