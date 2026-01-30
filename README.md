python 01_enrich_problem_schema_paper.py --input raw.jsonl --out enriched.jsonl \
  --model gpt-4.1-mini \
  --max_steps 8 \
  --allowed_ops IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,SAVE_RESULT,CHECK

python 02_embed_and_index.py --input enriched.jsonl \
  --out_skel skeleton_embedded.jsonl \
  --out_q question_embedded.jsonl \
  --out_anchors anchors.jsonl

python 03_retrieve_paper.py \
  --skeleton_embedded skeleton_embedded.jsonl \
  --question_embedded question_embedded.jsonl \
  --anchors anchors.jsonl \
  --out retrieval_bundles.jsonl

python 04_generate_skeletons.py --bundles retrieval_bundles.jsonl --out generated_skeletons.jsonl

python 05_generate_questions.py \
  --bundles retrieval_bundles.jsonl \
  --skeletons generated_skeletons.jsonl \
  --out generated_problems.jsonl

python 06_verify_and_score.py --input generated_problems.jsonl --out scored.jsonl
