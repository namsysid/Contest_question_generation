# Verified-reference generation pilot

## Scope

Eight verified structured solutions were sampled across projectile motion, circular motion, fluids,
quadratic drag, rolling, normal modes, orbital motion, and rigid-body constraints. GPT-5-mini produced
surface-free mechanisms, then cheap scenario plans, before any full question construction.

## Result

- Scenario proposals: 8
- Rejected before construction as source-parallel: 7
- Passed scenario screen: 1
- Actually usable after construction: 0

The sole passed scenario renamed the source ladder as a plank whose endpoints slide in perpendicular
horizontal and vertical guides. This is the same constraint topology and solution as 2026 Q17, so the
screen was a false positive. The completed question also had three algebraically identical answer
choices (A, D, and E), making it non-unique. It was rejected without spending a verifier call.

## Interpretation

The rebuilt source bank improves source correctness, difficulty labels, and reusable reference context.
It does **not** by itself make direct mechanism-transfer generation reliable. GPT-5-mini tends to map
the abstract equations back to the most familiar source topology, even when source nouns are forbidden.

The scenario-first stage was still useful: it prevented seven additional full construction calls. The
next generation architecture should use source solutions for difficulty/style calibration and critique,
not as a mechanism that the constructor is instructed to preserve. Scenario invention should occur
without source solutions in context; verified solutions can be attached afterward for calibration.

## Cost

Recorded pilot usage: 38,281 tokens. No blind-verifier or embedding call was made.

