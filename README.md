UNIFIED ENTRYPOINT:

python3 src/all_in_one.py pipeline --domain chem
python3 src/all_in_one.py pipeline --domain phys

# Run only part of the pipeline
python3 src/all_in_one.py pipeline --domain chem --from-stage 03 --to-stage 05

# Direct generation (stage 07, OpenAI API variant)
python3 src/07_direct_generate_openai.py --input data/enriched_schemae/enriched.jsonl --out direct_generated.jsonl --model gpt-4.1-mini

# Sample 25 original source questions and score them through stage 06
python3 src/07_sample_original_and_score.py --input data/txts/all_questions.jsonl --count 25 --scored-out scored/original_sampled_scored.jsonl

# Stage 00: download F=net=ma PDFs from AAPT, then parse to txt/jsonl
python3 src/00_pdf-to-txt.py --out data

# See all wrapped scripts
python3 src/all_in_one.py list


PHYSICS:

python3 src/01_enrich_problem_schema_paper-2.py --input data/txts/all_questions.jsonl --out data/enriched_schemae/enriched.jsonl \
  --model gpt-4.1-mini \
  --grammar-mode strict

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


CHEM:

python3 src/01_enrich_problem_schema_paper-2.py --input chem_data/txts/all_questions.jsonl --out chem_data/enriched_schemae/enriched.jsonl \
  --model gpt-4.1-mini \
  --grammar-mode strict

python src/02_embed_and_index.py --input chem_data/enriched_schemae/enriched.jsonl \
  --out_skel chem_data/skeleton_embedded.jsonl \
  --out_q chem_data/question_embedded.jsonl \
  --out_anchors chem_data/anchors.jsonl

python3 src/03_retrieve.py \
  --skeleton_embedded chem_data/skeleton_embedded.jsonl \
  --question_embedded chem_data/question_embedded.jsonl \
  --anchors chem_data/anchors.jsonl \
  --enriched chem_data/enriched_schemae/enriched.jsonl \
  --require_skeleton_text \
  --out chem_data/retrieval_bundles.jsonl

python src/04_generate_skeletons.py --bundles chem_data/retrieval_bundles.jsonl --out chem_data/generated_skeletons.jsonl

python src/05_generate_questions.py \
  --bundles chem_data/retrieval_bundles.jsonl \
  --skeletons chem_data/generated_skeletons.jsonl \
  --out chem_data/generated_problems.jsonl

python src/06_verify_and_score.py --input chem_data/generated_problems.jsonl --out chem_data/scored.jsonl
