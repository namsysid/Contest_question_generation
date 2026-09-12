# Retrieval decision after solution-representation ablations

Decision: do **not** re-embed the regenerated source solutions yet.

Reasons:

- The historical evaluation found no statistically supported advantage of typed solution graphs over
  plain-text solutions after Holm-Bonferroni correction.
- Recent pilots showed that embeddings route to related records but do not transmit causal structure
  to the constructor.
- The current 2024-2026 corpus has only 26 diagram-free records. Exhaustive generation can iterate the
  verified records directly, without retrieval.
- Re-embedding cannot repair an incorrect derivation or a weak constructor.

Current policy:

1. Retrieve IDs using the existing question embeddings when retrieval is needed.
2. Join those IDs to independently verified structured solutions.
3. Show the constructor the readable steps, decisions, mechanisms, pitfalls, and blind derivation.
4. For exhaustive generation, skip retrieval and process every verified record once.
5. Revisit mechanism-signature embeddings only after an evaluation demonstrates better selection or
   generation quality, not merely higher semantic similarity.

