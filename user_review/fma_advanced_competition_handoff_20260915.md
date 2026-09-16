# Handoff: add an Advanced Competition tier for F=ma

## Objective

Add a student-facing **Advanced Competition** tier between Competition and Challenge without changing the meaning of the corrected shortest-solve D1/D2/D3 rubric.

The key design decision is to keep two independent fields:

- `Difficulty`: structural shortest-solve tier (`1`, `2`, or `3`).
- `PracticeBand`: student-facing placement (`foundation`, `competition`, `advanced_competition`, `challenge`, or `olympiad`).

Do not equate D1/D2/D3 directly with website bands. A routine but lengthy question can be structural D1, while a concise question with a hidden modeling decision can be D2.

## Why this is needed

The latest manual ceiling audit found:

- D2 parents: 4/8 clearly appropriate for normal F=ma, 2/8 defensible only in the late-exam tail, and 2/8 borderline overshoots.
- Two F=ma-safe D2 questions—Momentum and Rotation—are actually structural D1 under the corrected rubric.
- D3 parents: only 3/18 are plausible as hardest-tail F=ma challenges; 15/18 are multi-regime/Olympiad-style and should not populate a normal F=ma Challenge tier.

Audit report:

`/Users/namansiddheshwar/RAG/Contest_question_generation/user_review/fma_d2_d3_ceiling_audit_20260915/REPORT.md`

Important correction: the previously cited `124/159 = 78.0% D1` figure describes the relabeled generated live MongoDB bank. It is **not** the D1 distribution of official F=ma exams and does not mean D1 corresponds to only questions 1–5.

## Recommended product bands

- `foundation`: routine introductory F=ma practice.
- `competition`: harder D1 and easier/ordinary D2; representative of normal F=ma.
- `advanced_competition`: harder D2 and carefully selected low-D3; representative of the late-exam tail.
- `challenge`: only D3 questions that remain inside the F=ma ceiling.
- `olympiad`: multi-regime D3 questions that exceed the F=ma ceiling.

The website may initially expose the first four bands and keep `olympiad` hidden until there is enough content.

## Existing implementation locations

Generation repository:

- D2 system: `.worktrees/fma_original_qc/d2_generation/`
- D3 system: `.worktrees/fma_original_qc/d3_generation/`
- D2 upload verification currently requires `validation.publication_class == "competition_standard"`.
- D3 upload verification currently requires `validation.publication_class == "challenging"`.

Website repository:

`/Users/namansiddheshwar/steamcoach/steamcoach_master`

Relevant files:

- `server.js`: `physicsDifficultyLevels` hardcodes `foundation: 1`, `competition: 2`, and `challenge: 3`; `/api/physics/questions` queries `Difficulty` directly.
- `server.js`: family endpoints assume `Difficulty: 3` for Challenge families.
- `physics/index.html`: currently contains three difficulty cards.
- `physics/app.js`: displays the selected difficulty and hardcodes similar-family sessions as `challenge`.
- `physics/styles.css`: difficulty grid currently uses three columns.

## Implementation plan

1. Add a top-level `PracticeBand` field to F=ma MongoDB documents. Preserve `Difficulty` unchanged as the structural label.
2. Backfill `PracticeBand` conservatively from the manual audit. Do not bulk-map every D3 to Challenge.
3. Change `/api/physics/questions` to validate and query `PracticeBand` rather than translating the requested band into `Difficulty`.
4. Add the Advanced Competition card and update the responsive four-card layout.
5. Make similar-family queries and sessions inherit the parent document's `PracticeBand` rather than assuming D3/Challenge.
6. Update D2/D3 packaging, upload verification, API tests, and website tests to check both structural `Difficulty` and `PracticeBand`.
7. Read back the migrated MongoDB records and run a website/API smoke test before deploying or pushing.

## Initial content placement

Use the full audit report for item-level decisions. Conservatively:

- Normal Competition candidates: current D2 Forces and Energy; Momentum and Rotation are also F=ma-safe but should be structurally relabeled D1.
- Advanced Competition candidates: current D2 Circular & gravity and Oscillations; Kinematics only after review/simplification.
- Borderline/hold: current D2 Fluids.
- Possible Challenge ceiling anchors: new D3 Kinematics, Forces, and Momentum only. These remain borderline and should not define the center of the tier.
- Olympiad/hidden: the other 15 unique D3 parents.

There is not yet enough balanced content for every topic in Advanced Competition or Challenge. Generate and independently solve additional items only after the band field and ceiling gate exist.

## New acceptance gate

After the existing independent solve assigns structural D1/D2/D3, run a separate F=ma-ceiling audit. Reject from Advanced Competition/Challenge when a problem has:

- more than one major regime switch;
- tuned repeated impacts, detachments, or recontacts;
- several successive system-boundary changes;
- advanced continuum/effective-inertia modeling not standard for F=ma;
- difficulty created mainly by obscure terminology, long algebra, or calculation volume.

Target for Advanced Competition or low Challenge: approximately two indispensable modeling choices and three linked deductions within one coherent physical regime.

## Safety constraints

- Do not upload or deploy until MongoDB readback and website/API smoke tests pass.
- Preserve the existing similar-question family grouping and keep it below the challenge problem, outside the worked solution.
- Do not silently relabel the current D3 bank as F=ma Challenge.
- Avoid recognizable copying and diagram-dependent questions, consistent with the existing D2/D3 pipelines.
