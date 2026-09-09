from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.circuit_lab.model_client import embed_texts

from .common import classify_topics, graph_text, read_jsonl, write_jsonl
from .expand_2022 import deterministic_audit, holistic_audit, item_text
from .finalize_2022 import plan_from_spec
from .pipeline import create_item, normalized, retrieve_exemplars


MCQ_SPECS: list[dict[str, Any]] = [
    {"response_type": "multiple_choice", "topic": "carbohydrates_sweeteners", "difficulty": 1,
     "target_skill": "Identify the monosaccharide components of lactose.", "novel_context": "Direct disaccharide-composition item.",
     "fixed_givens": [], "verification": {"expected_answer": "glucose and galactose",
     "calculation": "Lactose is composed of glucose and galactose."},
     "distractor_mechanisms": ["confuse lactose with sucrose", "confuse lactose with maltose", "mix components of two disaccharides"]},
    {"response_type": "multiple_choice", "topic": "proteins_enzymes", "difficulty": 1,
     "target_skill": "Distinguish an enzyme from its substrate, cofactor, and a structural protein.", "novel_context": "Direct enzyme-role classification.",
     "fixed_givens": [], "verification": {"expected_answer": "A catalyst that speeds a reaction without being consumed.",
     "calculation": "Enzymes are biological catalysts; substrates are transformed, while cofactors assist and structural proteins provide support."},
     "distractor_mechanisms": ["call the substrate the enzyme", "confuse an enzyme with a required cofactor", "confuse catalytic and structural protein roles"]},
    {"response_type": "multiple_choice", "topic": "lipids_emulsions", "difficulty": 1,
     "target_skill": "Identify the structural feature that makes a fatty acid unsaturated.", "novel_context": "Direct lipid-structure classification.",
     "fixed_givens": [], "verification": {"expected_answer": "One or more carbon-carbon double bonds.",
     "calculation": "Unsaturated fatty acids contain at least one C=C bond; saturated fatty acids contain none."},
     "distractor_mechanisms": ["confuse ester linkage with unsaturation", "confuse chain length with saturation", "choose only single carbon-carbon bonds"]},
    {"response_type": "multiple_choice", "topic": "nutrition_processing", "difficulty": 2,
     "target_skill": "Apply water solubility and heat sensitivity to vitamin loss during cooking.", "novel_context": "Boiled vegetables with cooking water discarded.",
     "fixed_givens": ["Equal portions of a vegetable are steamed or boiled; the boiling water is discarded."],
     "verification": {"expected_answer": "The boiled portion is more likely to lose vitamin C because it can leach into water and is heat-sensitive.",
     "calculation": "Vitamin C is water-soluble and heat-sensitive, so boiling plus discarding the water promotes greater loss than steaming."},
     "distractor_mechanisms": ["treat vitamin C as fat-soluble", "claim steaming removes more into water", "confuse vitamin loss with mineral creation"]},
    {"response_type": "multiple_choice", "topic": "analytical_tests", "difficulty": 2,
     "target_skill": "Combine iodine, Biuret, and Benedict results to identify nutrients.", "novel_context": "One unknown food extract receives three qualitative tests.",
     "fixed_givens": ["An extract gives amber iodine, violet Biuret, and orange heated Benedict results."],
     "verification": {"expected_answer": "Protein and reducing sugar are present, but starch is not detected.",
     "calculation": "Amber iodine is negative for starch, violet Biuret is positive for protein, and orange Benedict is positive for reducing sugar."},
     "distractor_mechanisms": ["reverse iodine result", "reverse Biuret result", "treat blue-free Benedict as negative"]},
    {"response_type": "multiple_choice", "topic": "experimental_analysis", "difficulty": 2,
     "target_skill": "Calculate density and use it to predict layering in a food mixture.", "novel_context": "Aqueous syrup and vegetable oil form two layers.",
     "fixed_givens": ["A 10.0 mL syrup sample has mass 12.5 g; vegetable oil has density 0.92 g/mL; the liquids do not mix."],
     "verification": {"expected_answer": "The syrup density is 1.25 g/mL, so the oil forms the upper layer.",
     "expected_value": 1.25, "expression": "12.5 / 10.0",
     "calculation": "12.5 g / 10.0 mL = 1.25 g/mL. The less-dense oil (0.92 g/mL) floats above it."},
     "distractor_mechanisms": ["invert mass and volume", "put denser liquid on top", "compare masses without calculating density"]},
    {"response_type": "multiple_choice", "topic": "carbohydrates_sweeteners", "difficulty": 2,
     "target_skill": "Distinguish reducing from nonreducing common sugars.", "novel_context": "Select the pair expected to give positive Benedict tests.",
     "fixed_givens": [], "verification": {"expected_answer": "glucose and lactose",
     "calculation": "Glucose and lactose are reducing sugars; sucrose and sucralose are nonreducing in the standard classroom test."},
     "distractor_mechanisms": ["assume every sweet sugar is reducing", "assume every disaccharide is nonreducing", "confuse sucrose with glucose"]},
    {"response_type": "multiple_choice", "topic": "proteins_enzymes", "difficulty": 3,
     "target_skill": "Use retained activity after a pH treatment to distinguish reversible inhibition from enzyme denaturation.",
     "novel_context": "An enzyme is exposed to low pH, returned to its optimum pH, and retested.",
     "fixed_givens": ["At pH 7 an untreated enzyme has activity 10 units; after exposure to pH 2 and return to pH 7, activity is 1 unit."],
     "verification": {"expected_answer": "The pH 2 exposure caused largely irreversible denaturation or active-site damage.",
     "calculation": "Activity remains low after optimum pH is restored, supporting persistent structural damage rather than only reversible pH inhibition."},
     "distractor_mechanisms": ["claim full reversible inhibition despite failed recovery", "claim substrate specificity changed without evidence", "treat lower activity as faster catalysis"]},
    {"response_type": "multiple_choice", "topic": "lipids_emulsions", "difficulty": 3,
     "target_skill": "Relate cis/trans fatty-acid shape to packing and melting behavior.",
     "novel_context": "Compare two C18:1 fats differing only in cis versus trans geometry.",
     "fixed_givens": ["Fat X contains mostly cis C18:1; Fat Y contains mostly trans C18:1."],
     "verification": {"expected_answer": "Fat Y packs more tightly and is likely firmer at room temperature.",
     "calculation": "A cis double bond creates a kink that disrupts packing; the straighter trans chain packs more tightly and tends to have a higher melting point."},
     "distractor_mechanisms": ["reverse cis and trans packing", "claim identical formula guarantees identical melting", "confuse tighter packing with lower melting point"]},
    {"response_type": "multiple_choice", "topic": "food_safety_preservation",
     "secondary_topics": ["food_chemistry_reactions"], "difficulty": 4,
     "target_skill": "Correctly classify sugar-, dryness-, and salt-tolerant microorganisms.",
     "novel_context": "Match three microbial growth preferences to osmophile, xerophile, and halophile.",
     "fixed_givens": ["Microbe P grows best in concentrated syrup; Q grows on very dry food; R grows in concentrated brine."],
     "verification": {"expected_answer": "P is an osmophile, Q a xerophile, and R a halophile.",
     "calculation": "Osmophiles favor high osmotic sugar, xerophiles low-moisture environments, and halophiles high salt."},
     "distractor_mechanisms": ["swap osmophile and halophile", "confuse xerophile with salt tolerance", "reverse all three classifications"]},
    {"response_type": "multiple_choice", "topic": "experimental_analysis",
     "secondary_topics": ["carbohydrates_sweeteners"], "difficulty": 2,
     "target_skill": "Choose a controlled design for testing sucrose concentration's effect on jam viscosity.",
     "novel_context": "Four proposed jam-flow experiments vary different factors.",
     "fixed_givens": ["The goal is to test only the effect of sucrose concentration on jam viscosity."],
     "verification": {"expected_answer": "Vary sucrose concentration while holding temperature, sample volume, heating time, and flow apparatus constant.",
     "calculation": "Only the independent variable should change; other viscosity-affecting variables must be controlled."},
     "distractor_mechanisms": ["change sugar and temperature together", "change sample volume and apparatus", "measure color instead of viscosity"]},
]

FRQ_SPECS: list[dict[str, Any]] = [
    {"response_type": "short_answer", "topic": "proteins_enzymes", "difficulty": 2, "points": 2,
     "target_skill": "Explain the different effects of fresh and canned pineapple on gelatin.",
     "novel_context": "Compare gelatin made with fresh versus canned pineapple.",
     "fixed_givens": ["Gelatin with fresh pineapple remains liquid; gelatin with canned pineapple sets."],
     "verification": {"expected_answer": "Fresh pineapple contains active bromelain that hydrolyzes gelatin protein; canning heat denatures bromelain, so canned pineapple allows the gel to set.",
     "calculation": "Active protease disrupts the gelatin network, whereas heat-denatured protease does not."},
     "distractor_mechanisms": ["attribute the difference only to sugar", "claim canning creates gelatin", "reverse the effect of enzyme denaturation"]},
    {"response_type": "short_answer", "topic": "food_chemistry_reactions", "difficulty": 2,
     "target_skill": "Distinguish caramelization from Maillard browning by reactants.",
     "novel_context": "Compare browning of pure sugar syrup and bread crust.",
     "fixed_givens": ["Pure sugar syrup browns when heated; bread crust browns while containing sugars and proteins."],
     "verification": {"expected_answer": "The sugar syrup mainly caramelizes; bread crust undergoes Maillard browning between reducing sugars and amino groups.",
     "calculation": "Caramelization involves heated sugars alone, while Maillard browning requires a reducing sugar and amino compound."},
     "distractor_mechanisms": ["call both reactions Maillard", "call both reactions caramelization", "reverse their required reactants"]},
    {"response_type": "numeric", "topic": "analytical_tests", "secondary_topics": ["carbohydrates_sweeteners"],
     "difficulty": 3, "target_skill": "Undo a laboratory dilution to find reducing-sugar concentration.",
     "novel_context": "A diluted juice aliquot is measured with a calibrated reducing-sugar assay.",
     "fixed_givens": ["5.0 mL juice is diluted to 25.0 mL; the diluted sample measures 0.80 mg/mL reducing sugar."],
     "verification": {"expected_answer": "4.0 mg/mL", "expected_value": 4.0,
     "expression": "0.80 * 25.0 / 5.0", "calculation": "The dilution factor is 25.0/5.0 = 5, so the original concentration is 0.80 × 5 = 4.0 mg/mL."},
     "distractor_mechanisms": ["invert dilution factor", "report diluted concentration", "multiply by aliquot volume without dividing"]},
    {"response_type": "short_answer", "topic": "nutrition_processing", "secondary_topics": ["proteins_enzymes"],
     "difficulty": 2, "points": 2, "target_skill": "Distinguish lactose intolerance from milk-protein allergy.",
     "novel_context": "Explain why lactose-free milk addresses intolerance but not allergy.",
     "fixed_givens": ["Lactose-free milk has had lactose hydrolyzed but still contains milk proteins."],
     "verification": {"expected_answer": "It can help lactose intolerance because lactose is removed or hydrolyzed, but it is unsafe for milk allergy because allergenic milk proteins remain.",
     "calculation": "Intolerance concerns digestion of lactose; allergy is an immune response to milk proteins."},
     "distractor_mechanisms": ["claim both conditions are caused by lactose", "claim lactase destroys milk proteins", "confuse immune and digestive responses"]},
    {"response_type": "short_answer", "topic": "carbohydrates_sweeteners",
     "secondary_topics": ["food_chemistry_reactions"], "difficulty": 2, "points": 2,
     "target_skill": "Explain graininess in fudge using sugar crystallization.",
     "novel_context": "A fudge mixture is stirred while cooling and becomes grainy.",
     "fixed_givens": ["One fudge batch is stirred vigorously while cooling and becomes grainy; an otherwise identical batch is left undisturbed and stays smooth."],
     "verification": {"expected_answer": "Stirring promotes nucleation and growth of many sucrose crystals; those crystals create the grainy texture.",
     "calculation": "Agitation of a supersaturated cooling sugar solution encourages crystal formation, producing graininess."},
     "distractor_mechanisms": ["attribute graininess only to water evaporation", "claim stirring dissolves all crystals", "confuse crystallization with caramelization"]},
    {"response_type": "short_answer", "topic": "nutrition_processing",
     "secondary_topics": ["food_chemistry_reactions"], "difficulty": 3, "points": 2,
     "target_skill": "Compare accumulation of excess fat- and water-soluble vitamins.",
     "novel_context": "Compare repeated excess intake of vitamins A and C.",
     "fixed_givens": ["A person repeatedly consumes much more vitamin A and vitamin C than needed."],
     "verification": {"expected_answer": "Vitamin A is more likely to accumulate because it is fat-soluble and stored; excess vitamin C is water-soluble and is more readily excreted in urine.",
     "calculation": "Fat-soluble vitamins can be stored in body tissues, whereas excess water-soluble vitamins are generally excreted more readily."},
     "distractor_mechanisms": ["reverse vitamin solubilities", "claim all excess vitamins are stored equally", "confuse excretion with absorption"]},
    {"response_type": "short_answer", "topic": "experimental_analysis",
     "secondary_topics": ["carbohydrates_sweeteners"], "difficulty": 3, "points": 4,
     "target_skill": "Specify variables and controls in a food-viscosity investigation.",
     "novel_context": "Design a fair test of sucrose concentration and jam flow time.",
     "fixed_givens": ["Several jam samples will differ only in sucrose concentration and drain through the same funnel."],
     "verification": {"expected_answer": "Independent variable: sucrose concentration; dependent variable: drain time or viscosity; valid controls include temperature and equal sample volume (also heating time or funnel geometry).",
     "calculation": "A fair test changes sugar concentration, measures flow behavior, and holds other viscosity-affecting conditions constant."},
     "distractor_mechanisms": ["identify drain time as independent", "vary temperature with sugar", "measure an unrelated outcome"]},
    {"response_type": "short_answer", "topic": "nutrition_processing", "difficulty": 2, "points": 2,
     "target_skill": "Map common vitamin chemical names to the four fat-soluble vitamin letters.",
     "novel_context": "Classify retinol, cholecalciferol, tocopherol, and phylloquinone.",
     "fixed_givens": ["A supplement contains retinol, cholecalciferol, tocopherol, and phylloquinone."],
     "verification": {"expected_answer": "Vitamins A, D, E, and K, respectively.",
     "calculation": "Retinol is A, cholecalciferol D, tocopherol E, and phylloquinone K; all are fat-soluble."},
     "distractor_mechanisms": ["confuse chemical names with B vitamins", "omit vitamin K", "reverse D and E"]},
    {"response_type": "short_answer", "topic": "carbohydrates_sweeteners",
     "secondary_topics": ["nutrition_processing"], "difficulty": 2, "points": 2,
     "target_skill": "Distinguish added sugars from naturally occurring sugars on a food label.",
     "novel_context": "Classify honey added during processing versus lactose naturally present in milk.",
     "fixed_givens": ["A yogurt contains lactose naturally from milk and honey mixed in during manufacturing."],
     "verification": {"expected_answer": "The honey sugars count as added sugars because honey was added during processing; the milk's naturally occurring lactose does not.",
     "calculation": "Added sugars are introduced during processing or preparation, unlike sugars naturally present in intact ingredients such as milk lactose."},
     "distractor_mechanisms": ["count every sugar as added", "exclude honey because it is natural", "classify lactose as added despite no added lactose ingredient"]},
    {"response_type": "short_answer", "topic": "food_safety_preservation",
     "secondary_topics": ["food_chemistry_reactions"], "difficulty": 3, "points": 2,
     "target_skill": "Distinguish moisture content from water activity and identify the better microbial-growth predictor.",
     "novel_context": "Explain why foods with equal total water can differ in microbial stability.",
     "fixed_givens": ["Two foods have the same percent water by mass but different water activities."],
     "verification": {"expected_answer": "Moisture content measures total water, while water activity measures water available for reactions and microbial use; water activity better predicts microbial growth.",
     "calculation": "Water can be bound differently even at equal total content, so the available-water measure is more relevant to growth."},
     "distractor_mechanisms": ["treat the quantities as identical", "claim total water always predicts growth", "reverse bound and available water"]},
    {"response_type": "numeric", "topic": "experimental_analysis", "difficulty": 3, "points": 2,
     "target_skill": "Calculate food density while converting customary mass and volume units.",
     "novel_context": "Find the density of syrup in a shipping pail using pounds and US gallons.",
     "fixed_givens": ["A pail contains 3.00 US gal of syrup with mass 34.0 lb; use 1 lb = 453.6 g and 1 US gal = 3.785 L."],
     "verification": {"expected_answer": "approximately 1.36 × 10^3 g/L", "expected_value": 1358.2034346,
     "expression": "34.0 * 453.6 / (3.00 * 3.785)",
     "calculation": "Mass = 15422.4 g; volume = 11.355 L; density = 1358 g/L ≈ 1.36 × 10^3 g/L."},
     "distractor_mechanisms": ["convert only one unit", "invert mass and volume", "treat gallons as liters"]},
]

KEEP_MCQ = [
    "food-science-b-generated-204", "food-science-b-generated-205",
]
KEEP_FRQ = [
    "food-science-b-frq-301", "food-science-b-frq-303", "food-science-b-frq-304",
]
SELECT_MCQ_INDICES = [index for index in range(len(MCQ_SPECS)) if index != 5]
SELECT_FRQ_INDICES = [0, 7, 8, 9, 10]

KEPT_FRQ_OVERRIDES: dict[str, dict[str, Any]] = {
    "food-science-b-frq-301": {
        "prompt": "Name the general class of indigestible plant carbohydrates that adds stool bulk.",
        "points": 1, "rubric": ["1 point: names dietary fiber; accept fiber or cellulose as a representative plant fiber."],
    },
    "food-science-b-frq-303": {
        "prompt": "A fatty acid is written as 18:2 Δ9,12. Report its omega number only.",
        "points": 1, "rubric": ["1 point: reports 6 (omega-6)."],
    },
    "food-science-b-frq-304": {
        "prompt": "Batch A pickled carrots ferment to pH 3.8; unfermented Batch B remains at pH 6.2. Which batch is better preserved, and which acid produced during fermentation provides the main protection?",
        "points": 2, "rubric": [
        "1 point: identifies Batch A as better preserved.",
        "1 point: identifies lactic acid as the protective compound.",
    ]},
}


def anchor_for(corpus: list[dict[str, Any]], spec: dict[str, Any], offset: int) -> dict[str, Any]:
    desired = "multiple_choice" if spec["response_type"] == "multiple_choice" else spec["response_type"]
    matches = [row for row in corpus if spec["topic"] in (row.get("topics") or []) and
               row.get("response_type") == desired]
    if not matches:
        matches = [row for row in corpus if spec["topic"] in (row.get("topics") or [])]
    return matches[offset % len(matches)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate the expanded Food Science 2022 bank")
    parser.add_argument("--pilot-dir", default="science_olympiad/food_science_b/season_2022_pilot")
    parser.add_argument("--candidate-dir", default="science_olympiad/food_science_b/season_2022_expanded_v4")
    parser.add_argument("--work-dir", default="science_olympiad/food_science_b/season_2022_expanded_final")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--generation-model", default="gpt-5.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--seed", type=int, default=88000)
    args = parser.parse_args()
    pilot, candidate_root, root = Path(args.pilot_dir), Path(args.candidate_dir), Path(args.work_dir)
    corpus = read_jsonl(pilot / "corpus" / "items.jsonl")
    q_rows = read_jsonl(pilot / "enriched" / "question_embeddings.jsonl")
    s_rows = read_jsonl(pilot / "enriched" / "structure_embeddings.jsonl")
    source_qmat = normalized([row["embedding"] for row in q_rows])
    source_smat = normalized([row["embedding"] for row in s_rows])
    old = read_jsonl(candidate_root / "generated" / "items_final.jsonl")
    old_by_id = {item["id"]: item for item in old}
    for item_id, changes in KEPT_FRQ_OVERRIDES.items():
        old_by_id[item_id].update(changes)
    kept = [old_by_id[item_id] for item_id in KEEP_MCQ + KEEP_FRQ]
    pilot_items = read_jsonl(pilot / "generated" / "items_final.jsonl")
    targeted_path = root / "generated" / "targeted_items.jsonl"
    reports_path = root / "validation" / "targeted_reports.jsonl"
    plans_path = root / "generated" / "targeted_plans.jsonl"
    targeted = read_jsonl(targeted_path) if targeted_path.exists() else []
    reports = read_jsonl(reports_path) if reports_path.exists() else []
    plans = read_jsonl(plans_path) if plans_path.exists() else []
    prior = pilot_items + old + targeted
    prior_vectors = embed_texts(args.embedding_model, [item_text(item) for item in prior], provider=args.provider)
    selected_mcq_specs = [MCQ_SPECS[index] for index in SELECT_MCQ_INDICES]
    selected_frq_specs = [FRQ_SPECS[index] for index in SELECT_FRQ_INDICES]
    specs = selected_mcq_specs + selected_frq_specs
    while len(targeted) < len(specs):
        index = len(targeted)
        spec = specs[index]
        plan = plan_from_spec(spec)
        anchor = anchor_for(corpus, spec, index)
        exemplars = retrieve_exemplars(anchor, corpus, q_rows, s_rows)
        item, report, vector = create_item(
            anchor, exemplars, plan, prior, source_qmat, source_smat, corpus, prior_vectors,
            args.embedding_model, args.generation_model, args.provider,
            args.seed + index * 1000, 401 + index)
        item["id"] = (f"food-science-b-curated-mcq-{index + 1:03d}" if index < len(selected_mcq_specs)
                      else f"food-science-b-curated-frq-{index - len(selected_mcq_specs) + 1:03d}")
        report["id"] = item["id"]
        targeted.append(item)
        reports.append(report)
        plans.append(plan)
        prior.append(item)
        prior_vectors.append(vector)
        write_jsonl(targeted_path, targeted)
        write_jsonl(reports_path, reports)
        write_jsonl(plans_path, plans)
        print(f"Accepted targeted {index + 1}/{len(specs)}: {item['id']}", flush=True)

    mcqs = [old_by_id[item_id] for item_id in KEEP_MCQ] + targeted[:len(selected_mcq_specs)]
    frqs = [old_by_id[item_id] for item_id in KEEP_FRQ] + targeted[len(selected_mcq_specs):]
    items = mcqs + frqs
    for item in items:
        inferred = classify_topics(str(item.get("prompt") or "") + " " + str(item.get("solution") or ""))
        item["topics"] = list(dict.fromkeys([*(item.get("topics") or []), *inferred]))
    item_vectors = embed_texts(args.embedding_model, [item_text(item) for item in items], provider=args.provider)
    structure_texts = [graph_text({"response_type": item["response_type"], "topics": item["topics"],
                                   "analysis": {"reasoning_graph": item["generation"]["plan"]["reasoning_graph"]}})
                       for item in items]
    structure_vectors = embed_texts(args.embedding_model, structure_texts, provider=args.provider)
    deterministic = deterministic_audit(items, corpus, source_qmat, source_smat, item_vectors,
                                        structure_vectors, 12, 8, "source-faithful")
    audit_dir = root / "validation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "deterministic_audit.json").write_text(
        json.dumps(deterministic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    holistic = holistic_audit(items, corpus, args.generation_model, args.provider, args.seed + 999999)
    (audit_dir / "holistic_audit.json").write_text(
        json.dumps(holistic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(root / "generated" / "mcq_items_final.jsonl", mcqs)
    write_jsonl(root / "generated" / "frq_items_final.jsonl", frqs)
    write_jsonl(root / "generated" / "items_final.jsonl", items)
    print(f"Final: {len(mcqs)} MCQ, {len(frqs)} FRQ; deterministic={deterministic['overall_good']} "
          f"holistic={holistic.get('overall_good')}")


if __name__ == "__main__":
    main()
