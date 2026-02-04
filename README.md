python3 src/01_enrich_problem_schema_paper.py --input data/text/exams.txt/all_questions.jsonl --out data/enriched_schemae/enriched.jsonl \
  --model gpt-4.1-mini \
  --max_steps 8 \
  --allowed_ops IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,INTRODUCE_AUX,CONSTRAINT_COUPLING,CASEWORK/REGIME,INVARIANT/SYMMETRY,CHECK/SANITY

python src/02_embed_and_index.py --input data/enriched_schemae/enriched.jsonl \
  --out_skel data/skeleton_embedded.jsonl \
  --out_q data/question_embedded.jsonl \
  --out_anchors data/anchors.jsonl

python3 src/03_retrieve.py \
  --skeleton_embedded data/skeleton_embedded.jsonl \
  --question_embedded data/question_embedded.jsonl \
  --anchors data/anchors.jsonl \
  --enriched data/enriched_schemae/enriched.jsonl \
  --require_skeleton_text \
  --out retrieval_bundles.jsonl

python src/04_generate_skeletons.py --bundles retrieval_bundles.jsonl --out generated_skeletons.jsonl

python src/05_generate_questions.py \
  --bundles retrieval_bundles.jsonl \
  --skeletons generated_skeletons.jsonl \
  --out generated_problems.jsonl

python src/06_verify_and_score.py --input data/text/exam1-2015-1-8.jsonl --out scored.jsonl
