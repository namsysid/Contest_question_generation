# F=ma D1/D2/D3 ceiling audit — 2026-09-15

## Bottom line

- The current D2 set is usable only after calibration: 4/8 are clearly appropriate for F=ma, 2/8 are defensible only as late-exam/upper-tail questions, and 2/8 are borderline overshoots for a normal F=ma competition pool.
- Two of the F=ma-safe D2 items (Momentum and Rotation) are structurally D1 under the corrected shortest-solve rubric because their shortest routes are routine template chains with no indispensable hidden model choice.
- The current D3 parents are substantially too hard for an F=ma-oriented `challenging` pool. Of 18 unique parents, 3 are plausible only at the hardest tail and 15 should be held out as Olympiad-style/multi-regime problems.
- The 78% D1 number is **not an official-exam statistic**. It is the result of relabeling the 159-document live generated bank: 124 D1 and 35 D2. It shows that the old generator produced many routine/template questions.

## Calibration used

Difficulty was judged twice:

1. **Structural tier:** the corrected collapsed-shortest-solve rubric. D1 can include substantial algebra or advanced-looking apparatus if the route is routine. D2 needs linked deductions and a genuine hidden model choice.
2. **F=ma ceiling:** whether the item belongs in a normal 25-question F=ma exam or its immediate practice/challenge extension.

Official late-question anchors are generally shorter and stay within one governing regime. For example, 2026 Q22 requires a load split and torque balance; 2025 Q24 turns on one invariant/geometric insight. Several generated D3 items instead require three to five regime handoffs, tuned recontacts, or a new collision after every model change.

## D2 audit

| Topic / ID | F=ma placement | Shortest-solve tier | Finding |
|---|---|---|---|
| Kinematics — `fma_d2_bulk_20260914_kinematics_01` | Borderline upper tail | D2 | Moving-guide constraint plus two differentiations is valid, but unusually abstract without a diagram and harder than a typical competition item. |
| Forces — `..._forces_01` | Safe, late exam | D2 | Accelerating-frame force balance, friction direction, and limiting condition are linked but still standard F=ma mechanics. |
| Energy — `..._energy_01` | Safe, late exam | D2 | Effective gravity, work/energy, then radial force is a legitimate linked chain with one meaningful frame choice. |
| Momentum — `..._momentum_01` | Safe | D1, not D2 | Restitution/momentum followed by spring energy is a routine fully signposted chain. |
| Circular & gravity — `..._circular_gravity_01` | Defensible upper tail | D2 | Recognizing linear interior gravity and projected SHM is a real insight, but it belongs near the end. |
| Rotation — `..._rotation_01` | Safe | D1, not D2 | Angular momentum about the hinge followed by energy is a standard pellet-and-rod template. |
| Oscillations — `..._oscillations_01` | Defensible upper tail | D2 | Effective-gravity equilibrium and the first turning point are concise but non-obvious. |
| Fluids — `..._fluids_01` | Borderline overshoot | D2 | Vector buoyancy in an accelerating liquid is compact but too niche for a normal F=ma competition pool unless explicitly taught/scaffolded. |

Recommended immediate split:

- Normal F=ma competition: Forces, Energy, Momentum, Rotation.
- Late-exam/challenge extension: Circular & gravity, Oscillations.
- Hold or simplify: Kinematics, Fluids.
- Relabel Momentum and Rotation to D1 if the corrected structural rubric remains authoritative.

## D3 audit

### Plausible only as hardest-tail F=ma challenges (3/18)

- `fma_d3_more_20260915_kinematics_01`: collision plus acceleration reversal and event ordering; borderline but solvable with elementary constant-acceleration models.
- `fma_d3_more_20260915_forces_01`: vector static-friction threshold followed by sliding; borderline late-exam extension.
- `fma_d3_more_20260915_momentum_01_v2`: launch, wall return, friction stop, and sticking; contrived and multi-event, but still built from elementary F=ma laws.

These should not define the center of the `challenging` distribution. At most they are ceiling anchors.

### Hold out from F=ma challenging (15/18)

- Old curated set: `fma_d3_curated_0001`, `0002`, `0003`, `0004`, `0005`, `0006`, `0007`, `0008`, `0010`, and `fma_d3_curated_forces_rod_0001`.
- New all-topic set: `fma_d3_more_20260915_energy_01`, `circular_gravity_01_v2`, `rotation_01`, `oscillations_02`, and `fluids_03`.

The recurring issue is not equation count. It is too many successive governing-model changes: slack/taut impacts, wall rebounds followed by rolling transitions, latch/release system-boundary changes, contact loss and recontact, orbital transfer followed by collision and a second orbit, or fluid–cart effective-inertia modeling. These are Olympiad-style mechanisms and several approach IPhO-style reasoning, even if each individual law is elementary.

## Why “D1 = first five questions” conflicts with 78%

It conflicts only if D1 is interpreted as an ordinal label meaning “the easiest first five.” That is not this rubric.

- Here, D1 means **routine shortest route**: recall, a familiar template, or obvious substitutions with no genuine hidden model selection.
- A later-position official problem can therefore still be D1 structurally if it is algebraically long but routine.
- Conversely, an early concise problem can be D2 if a hidden physical choice is indispensable.
- Most importantly, `124/159 = 78.0%` came from the generated live bank audit, not from classifying the 2024–2026 official exams. It should not be used as the expected D1 share of an F=ma exam.

The clean product interpretation is therefore:

- `Basic` / D1: routine F=ma practice, broader than “questions 1–5.”
- `Competition` / D2: ordinary middle-to-late F=ma, with at least one real modeling choice.
- `Challenging` / D3: the hardest F=ma tail, capped at roughly one major regime change—not general Olympiad mechanics.

## Generation guardrail

For future F=ma D3 generation, reject candidates with more than one major regime switch, tuned repeated impacts/recontacts, or a continuum/effective-inertia derivation not standard in F=ma. Target two indispensable modeling choices and about three linked deductions inside one coherent physical regime. That preserves genuine difficulty without drifting into Olympiad problems.
