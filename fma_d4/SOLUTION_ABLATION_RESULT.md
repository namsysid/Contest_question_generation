# Full-solution ablation result

Two source solutions were regenerated from question text without showing the model the old graphs:

- 2025 Q20: Hill/Clohessy-Wiltshire relative motion, answer C (93 minutes).
- 2026 Q09: falling-ball energy, moving-wall collision recurrence, and escape threshold, answer B (208).

Both derivations are algebraically correct on direct review. They also reveal that graph node count was
inflating source difficulty: each has only three indispensable steps. Q20 has a genuinely non-routine
modeling insight; Q09 is substantially more routine than its old graph rating of D5 suggested.

## Generated candidate

The generator combined the two into a ball falling radially toward Earth and repeatedly receiving
tangential impulses from a moving plate. It claimed the speed after `k` impacts was
`v + 2kV` and selected 104 collisions.

## Independent verdict

- Physics valid: **no**
- Unique listed answer: **no**
- Difficulty: D2
- Failure: radial and tangential velocity components were added as scalar speeds.
- Under the stated impulse model the magnitude is `sqrt(v^2 + (2kV)^2)`, giving 250 collisions,
  which was absent from the choices.

Therefore full solutions alone do not fix the pipeline. They improve the input and make true source
difficulty visible, but the generator's forced cross-source synthesis is another root cause. The generator
must preserve one hard causal chain and combine another mechanism only through an explicit shared state.

Recorded API usage for this ablation was 27,714 tokens. One earlier completed source-solution response
could not be logged because the usage-log directory did not exist, so the true total is higher. That logger
bug is fixed in `src/circuit_lab/model_client.py`; no duplicate calls were launched while requests ran.

## Revised-generator preflight

The generator was changed to preserve the primary hard chain, avoid forced cross-source synthesis,
and explicitly audit vectors and assumptions. One preflight was then run before bulk regeneration.

It generalized 2025 Q20 to an arbitrary in-plane initial impulse and claimed the tool always reunites
with the astronaut after one orbital period. This is false. Its own Hill-equation solution contains the
along-track secular term `-3 V_y t`; at `t=T`, `x=0` but `y=-3 V_y T`, which is nonzero whenever the
initial impulse has an along-track component. The correct listed response is therefore “it never
coincides again for some impulse directions,” not the generated key.

This preflight used 10,566 tokens. No verifier call was needed because the contradiction is explicit in
the generated derivation. Bulk solution regeneration and re-embedding were deferred: doing them before
the construction/solve consistency problem is fixed would waste tokens.
