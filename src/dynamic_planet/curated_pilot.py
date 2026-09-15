from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .pipeline import (deterministic_embedding, deterministic_errors, deterministic_indices, ingest_pairs, lexical_similarity,
                       read_jsonl, topics_for, write_jsonl)


SPECS: list[dict[str, Any]] = [
    {
        "query": "Strahler stream order tributaries drainage network",
        "plan": {"topics": ["surface_water"], "target_skill": "apply Strahler ordering through two confluences",
                 "difficulty": 3, "response_type": "multiple_choice", "fixed_givens": ["A and B are order 2", "C is order 1", "D is order 3"],
                 "expected_answer": "B", "verification": "A+B makes order 3; adding C leaves 3; joining D makes order 4.",
                 "reasoning_graph": {"nodes": [{"id":"g1","type":"Given","label":"orders and confluence sequence"},{"id":"c1","type":"Concept","label":"equal orders increase; unequal retain larger"},{"id":"s1","type":"Process","label":"evaluate confluences in sequence"},{"id":"t1","type":"Target","label":"final Strahler order"}],"edges":[{"src":"g1","dst":"s1","type":"supports"},{"src":"c1","dst":"s1","type":"supports"},{"src":"s1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"multiple_choice","prompt":"Headwater streams A and B are each second order and join. First-order stream C then joins their channel. Farther downstream, that channel joins third-order stream D. What is the order immediately below the final confluence?","choices":{"A":"3","B":"4","C":"5","D":"7"},"answer":"B","solution":"A and B have equal order, so their junction forms a third-order channel. Adding first-order C does not change that order. The resulting third-order channel then joins third-order D, producing order 4.","points":3,"difficulty":3,"topics":["surface_water"]}
    },
    {
        "query": "runoff hydrograph discharge watershed storm lag time",
        "plan": {"topics":["surface_water","stream_dynamics"],"target_skill":"infer land-cover change from paired storm hydrographs","difficulty":3,"response_type":"multiple_choice","fixed_givens":["same storm and basin","peak rises from 22 to 38 m3/s","lag falls from 6 to 3 h"],"expected_answer":"C","verification":"More impervious cover reduces infiltration, accelerating and increasing runoff.","reasoning_graph":{"nodes":[{"id":"g1","type":"Observation","label":"higher peak discharge"},{"id":"g2","type":"Observation","label":"shorter lag"},{"id":"c1","type":"Concept","label":"impervious cover reduces infiltration"},{"id":"t1","type":"Target","label":"best land-cover change"}],"edges":[{"src":"g1","dst":"t1","type":"supports"},{"src":"g2","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"multiple_choice","prompt":"For equal storms in the same basin, the later hydrograph peaks at 38 m³/s after 3 hours; the earlier one peaked at 22 m³/s after 6 hours. Which change best explains both differences?","choices":{"A":"Wetlands were restored across the basin","B":"More fields were converted to forest","C":"More roads and roofs were constructed","D":"A reservoir began storing storm runoff"},"answer":"C","solution":"Roads and roofs are relatively impermeable. They reduce infiltration and route water quickly to channels, causing both a higher discharge peak and a shorter lag time.","points":3,"difficulty":3,"topics":["surface_water","stream_dynamics"]}
    },
    {
        "query": "groundwater hydraulic gradient water table wells flow direction",
        "plan": {"topics":["groundwater"],"target_skill":"calculate hydraulic gradient and infer groundwater-flow direction","difficulty":3,"response_type":"numeric","fixed_givens":["heads 146 m west and 134 m east","600 m separation"],"expected_answer":{"value":0.020,"unit":"m/m","direction":"east"},"verification":"(146-134)/600=0.020; flow is from high head west to low head east.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"two hydraulic heads and distance"},{"id":"c1","type":"Concept","label":"groundwater flows down hydraulic head"},{"id":"p1","type":"Process","label":"head difference divided by distance"},{"id":"t1","type":"Target","label":"gradient and direction"}],"edges":[{"src":"g1","dst":"p1","type":"supports"},{"src":"p1","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"numeric","prompt":"Two wells 600 m apart tap the same unconfined aquifer. The western well's water-table elevation is 146 m and the eastern well's is 134 m. Calculate the hydraulic gradient and state the groundwater-flow direction.","answer":{"value":0.020,"unit":"m/m","direction":"east","tolerance":0.001},"solution":"The head difference is 146 − 134 = 12 m. Hydraulic gradient = 12 m / 600 m = 0.020 m/m. Groundwater moves from higher hydraulic head to lower hydraulic head, so it flows eastward.","points":3,"difficulty":3,"topics":["groundwater"]}
    },
    {
        "query": "confined aquifer artesian well potentiometric surface",
        "plan": {"topics":["groundwater"],"target_skill":"distinguish an artesian well from a flowing artesian well","difficulty":2,"response_type":"multiple_choice","fixed_givens":["screen at 90 m","land surface 125 m","potentiometric surface 132 m","confined aquifer"],"expected_answer":"C","verification":"Pressure raises water above land surface, so it flows without pumping.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"well elevation relationships"},{"id":"c1","type":"Concept","label":"potentiometric surface controls rise in confined well"},{"id":"t1","type":"Target","label":"well behavior"}],"edges":[{"src":"g1","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"multiple_choice","prompt":"A well enters a confined aquifer at elevation 90 m. Land surface at the well is 125 m, and the aquifer's potentiometric surface is 132 m. What should occur when the well is opened?","choices":{"A":"Water remains at 90 m","B":"Water rises but stays below land surface","C":"Water flows at the surface without pumping","D":"Water drains from the well into the aquifer"},"answer":"C","solution":"Water in a confined well rises toward the potentiometric surface. Because that surface is 7 m above the land surface, water can emerge without pumping; this is a flowing artesian well.","points":2,"difficulty":2,"topics":["groundwater"]}
    },
    {
        "query": "dimictic lake summer stratification turnover dissolved oxygen",
        "plan": {"topics":["lakes","water_quality"],"target_skill":"predict seasonal oxygen redistribution during lake turnover","difficulty":3,"response_type":"short_answer","fixed_givens":["dimictic lake","late-summer hypolimnion low oxygen","autumn surface cooling and wind"],"expected_answer":"Density differences weaken; mixing carries oxygen downward and nutrients upward.","verification":"Cooling removes stable density layering, allowing wind-driven whole-lake circulation.","reasoning_graph":{"nodes":[{"id":"g1","type":"Observation","label":"cooling surface and wind"},{"id":"c1","type":"Concept","label":"weakened density stratification permits turnover"},{"id":"p1","type":"Process","label":"vertical mixing"},{"id":"t1","type":"Target","label":"oxygen and nutrient changes"}],"edges":[{"src":"g1","dst":"c1","type":"causes"},{"src":"c1","dst":"p1","type":"causes"},{"src":"p1","dst":"t1","type":"causes"}]}},
        "item": {"response_type":"short_answer","prompt":"A dimictic lake's hypolimnion has little dissolved oxygen by late summer. In autumn, surface water cools to nearly the same density as deeper water while winds continue. Explain the resulting circulation and one change in oxygen or nutrient distribution.","answer":"Autumn turnover mixes the water column, carrying oxygen-rich surface water downward and nutrient-rich deep water upward.","solution":"Surface cooling weakens the density barrier that maintained summer stratification. Wind can then circulate water through much of the lake. This autumn turnover replenishes deep dissolved oxygen and can transport accumulated nutrients toward surface waters.","points":4,"difficulty":3,"topics":["lakes","water_quality"]}
    },
    {
        "query": "eutrophication nutrient algae dissolved oxygen decomposition lake",
        "plan": {"topics":["lakes","water_quality"],"target_skill":"order the causal chain from nutrient input to fish stress","difficulty":3,"response_type":"multiple_choice","fixed_givens":["phosphorus runoff","algal bloom","night and decay"],"expected_answer":"D","verification":"Nutrients increase algae; respiration and decomposer demand reduce dissolved oxygen.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"phosphorus runoff increases"},{"id":"p1","type":"Process","label":"algal biomass grows"},{"id":"p2","type":"Process","label":"respiration and decomposition consume oxygen"},{"id":"t1","type":"Target","label":"fish experience low oxygen"}],"edges":[{"src":"g1","dst":"p1","type":"causes"},{"src":"p1","dst":"p2","type":"causes"},{"src":"p2","dst":"t1","type":"causes"}]}},
        "item": {"response_type":"multiple_choice","prompt":"Phosphorus runoff into a lake causes a dense algal bloom. Several days later, dissolved oxygen is lowest before sunrise and falls further as dead algae accumulate. Which process best connects the bloom to fish stress?","choices":{"A":"Algae permanently stop all cellular respiration","B":"Phosphorus directly removes oxygen atoms from water","C":"Photosynthesis becomes fastest at night","D":"Algal respiration and microbial decomposition consume dissolved oxygen"},"answer":"D","solution":"Algae respire during both day and night, while photosynthesis stops in darkness, explaining the pre-sunrise minimum. As algae die, decomposers use additional oxygen. The resulting low dissolved oxygen can stress or kill fish.","points":3,"difficulty":3,"topics":["lakes","water_quality"]}
    },
    {
        "query": "stream gradient elevation distance sediment competence velocity",
        "plan": {"topics":["stream_dynamics"],"target_skill":"compute channel gradient and connect slope reduction to competence","difficulty":3,"response_type":"numeric","fixed_givens":["elevation loss 18 m","channel length 1.5 km"],"expected_answer":{"value":1.2,"unit":"percent"},"verification":"18/1500*100=1.2%; lower slope generally lowers velocity and competence.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"elevation loss and channel length"},{"id":"p1","type":"Process","label":"convert length and calculate percent gradient"},{"id":"c1","type":"Concept","label":"reduced gradient tends to reduce velocity and competence"},{"id":"t1","type":"Target","label":"gradient and downstream particle-size inference"}],"edges":[{"src":"g1","dst":"p1","type":"supports"},{"src":"p1","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"numeric","prompt":"A stream descends 18 m along 1.5 km of channel. Calculate its percent gradient. If gradient decreases farther downstream while other factors are similar, state the expected change in the largest particle size the stream can transport.","answer":{"value":1.2,"unit":"percent","particle_change":"decreases","tolerance":0.05},"solution":"Convert 1.5 km to 1500 m. Percent gradient = (18/1500) × 100 = 1.2%. A lower slope generally lowers flow velocity and competence, so the maximum particle size transportable decreases.","points":3,"difficulty":3,"topics":["stream_dynamics"]}
    },
    {
        "query": "lake water budget residence time inflow outflow volume",
        "plan": {"topics":["lakes","water_cycle"],"target_skill":"combine a lake water budget with residence-time calculation","difficulty":3,"response_type":"numeric","fixed_givens":["volume 9.0e6 m3","stream inflow 2.4e6 m3/year","precipitation 0.6e6","evaporation 0.4e6","storage constant"],"expected_answer":{"value":3.00,"unit":"years"},"verification":"Stream outlet=2.4+0.6-0.4=2.6 million m3/y; total loss including evaporation=3.0; 9.0/3.0=3.00 y.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"volume, inflows, evaporation, steady storage"},{"id":"p1","type":"Process","label":"solve water balance for stream outlet"},{"id":"p2","type":"Process","label":"divide volume by total water loss"},{"id":"t1","type":"Target","label":"residence time"}],"edges":[{"src":"g1","dst":"p1","type":"supports"},{"src":"p1","dst":"p2","type":"supports"},{"src":"p2","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"numeric","prompt":"A lake holds 9.0×10⁶ m³. Annual stream inflow is 2.4×10⁶ m³, direct precipitation adds 0.6×10⁶ m³, and evaporation removes 0.4×10⁶ m³. Storage is constant and groundwater exchange is negligible. Find its mean water residence time.","answer":{"value":3.00,"unit":"years","tolerance":0.05},"solution":"Constant storage gives stream outlet = 2.4 + 0.6 − 0.4 = 2.6 million m³/year. Total water loss is outlet plus evaporation = 2.6 + 0.4 = 3.0 million m³/year, equal to total input. Residence time = 9.0/3.0 = 3.00 years.","points":4,"difficulty":3,"topics":["lakes","water_cycle"]}
    },
    {
        "query": "porosity permeability sediment grain sorting groundwater aquifer",
        "plan": {"topics":["groundwater"],"target_skill":"distinguish porosity from permeability using sample data","difficulty":3,"response_type":"multiple_choice","fixed_givens":["equal-length columns","equal porosity","same hydraulic gradient","sample X faster"],"expected_answer":"A","verification":"Equal pore fraction means equal porosity; faster transmission under controlled conditions indicates greater permeability.","reasoning_graph":{"nodes":[{"id":"g1","type":"Observation","label":"equal pore fraction"},{"id":"g2","type":"Observation","label":"different travel times under matched conditions"},{"id":"c1","type":"Concept","label":"porosity is storage; permeability is transmission"},{"id":"t1","type":"Target","label":"compare properties"}],"edges":[{"src":"g1","dst":"t1","type":"supports"},{"src":"g2","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"multiple_choice","prompt":"Two equal-length saturated sediment columns each contain 35% pore space. A tracer crosses X in 4 minutes and Y in 40 minutes under the same hydraulic gradient. Which comparison is best supported?","choices":{"A":"They have equal porosity, but X has greater permeability","B":"X has greater porosity, but equal permeability","C":"Y has greater porosity and permeability","D":"They have equal porosity and permeability"},"answer":"A","solution":"The stated 35% pore space gives both columns the same porosity. With equal lengths and the same gradient, much faster tracer travel through X indicates better-connected flow paths and therefore greater permeability.","points":3,"difficulty":3,"topics":["groundwater"]}
    },
    {
        "query": "topographic contour stream direction drainage divide elevation map",
        "plan": {"topics":["maps_data","surface_water"],"target_skill":"infer stream direction and drainage-divide placement from elevations","difficulty":3,"response_type":"short_answer","fixed_givens":["channel intersections 520, 500, 480 m west to east","north ridge 610 m","south ridge 590 m"],"expected_answer":"east; divide follows ridge crests, not the channel","verification":"Water follows decreasing channel elevations; divides occupy adjacent high ground.","reasoning_graph":{"nodes":[{"id":"g1","type":"Given","label":"successive channel elevations"},{"id":"g2","type":"Given","label":"ridge elevations on both sides"},{"id":"c1","type":"Concept","label":"water flows downhill; divides follow high ground"},{"id":"t1","type":"Target","label":"flow direction and divide location"}],"edges":[{"src":"g1","dst":"t1","type":"supports"},{"src":"g2","dst":"t1","type":"supports"},{"src":"c1","dst":"t1","type":"supports"}]}},
        "item": {"response_type":"short_answer","prompt":"Along a mapped channel, contour intersections are 520 m, 500 m, and 480 m from west to east. Ridge crests lie north at 610 m and south at 590 m. State streamflow direction and explain where nearby drainage divides should lie.","answer":"The stream flows east; drainage divides follow the high ridge crests north and south of the channel.","solution":"Channel elevation decreases eastward from 520 m to 480 m, so water flows east. A drainage divide separates runoff moving toward different basins and therefore follows surrounding high ground—here, the ridge crests—not the low channel.","points":3,"difficulty":3,"topics":["maps_data","surface_water"]}
    },
]


def heuristic_enrich(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    primary = (row.get("topics") or ["freshwater_systems"])[0]
    result["analysis"] = {"topics": row.get("topics"), "skills": [f"interpret {primary.replace('_', ' ')} evidence"],
                          "difficulty": 2 if row.get("response_type") == "multiple_choice" else 3,
                          "estimated_seconds": 60,
                          "reasoning_graph": {"nodes":[{"id":"n1","type":"Given","label":"source observations or data"},{"id":"n2","type":"Concept","label":f"relevant {primary.replace('_',' ')} principle"},{"id":"n3","type":"Target","label":"requested conclusion"}],"edges":[{"src":"n1","dst":"n3","type":"supports"},{"src":"n2","dst":"n3","type":"supports"}]},
                          "method":"deterministic_topic_template_v1"}
    result["graph_text"] = f"EVENT Dynamic Planet B; TOPICS {','.join(row.get('topics',[]))}; FORMAT {row.get('response_type')}; NODES Given:observations | Concept:{primary} | Target:conclusion; EDGES n1-supports->n3 | n2-supports->n3"
    return result


def closest(query: str, corpus: list[dict[str, Any]], count: int, response_type: str | None = None) -> list[dict[str, Any]]:
    pool = [row for row in corpus if response_type is None or row.get("response_type") == response_type] or corpus
    return sorted(pool, key=lambda row: lexical_similarity(query, row["prompt"]), reverse=True)[:count]


def materialize(root: Path) -> dict[str, Any]:
    corpus = ingest_pairs(root / "sources")
    write_jsonl(root / "corpus/items.jsonl", corpus)
    enriched = [heuristic_enrich(row) for row in corpus]
    write_jsonl(root / "enriched/items.jsonl", enriched)
    qrows, grows = deterministic_indices(enriched)
    write_jsonl(root / "enriched/question_embeddings.jsonl", qrows)
    write_jsonl(root / "enriched/graph_embeddings.jsonl", grows)
    bundles, plans, items, reports = [], [], [], []
    for index, spec in enumerate(SPECS, 1):
        qnear = closest(spec["query"], enriched, 3)
        primary = spec["plan"]["topics"][0]
        graph_pool = [row for row in enriched if primary in row.get("topics", [])]
        gnear = closest(primary.replace("_", " "), graph_pool or enriched, 3, spec["item"]["response_type"] if spec["item"]["response_type"] == "multiple_choice" else None)
        bundle_id = f"dp-bundle-{index:02d}"
        bundle = {"bundle_id":bundle_id,"retrieval_query":spec["query"],"anchor_id":qnear[0]["id"],
                  "question_exemplar_ids":[r["id"] for r in qnear],"structure_exemplar_ids":[r["id"] for r in gnear],
                  "retrieval_method":{"question":"signed feature hashing + lexical rerank","graph":"topic-filtered graph index + lexical rerank","external_model":False}}
        plan = {"bundle_id":bundle_id, **spec["plan"], "provenance":bundle}
        item = {"id":f"dynamic-planet-b-pilot-{index:02d}","event":"dynamic_planet","division":"B","season":2027,"season_topic":"Earth's Fresh Waters", **spec["item"],
                "generation":{"method":"agent_authored_no_external_model","bundle_id":bundle_id,"reasoning_plan":plan}}
        errors, lex_score, nearest = deterministic_errors(item, corpus)
        candidate_vector = deterministic_embedding(item["prompt"])
        semantic_scores = [(sum(a*b for a, b in zip(candidate_vector, row["embedding"])), row["id"]) for row in qrows]
        semantic_score, semantic_nearest = max(semantic_scores)
        if semantic_score >= 0.82:
            errors.append(f"hashed semantic similarity too high versus {semantic_nearest}")
        plan_errors = []
        if item["response_type"] != plan["response_type"]: plan_errors.append("response type differs from plan")
        if item["difficulty"] != plan["difficulty"]: plan_errors.append("difficulty differs from plan")
        if not set(item["topics"]) & set(plan["topics"]): plan_errors.append("topics do not overlap plan")
        reports.append({"id":item["id"],"validation_method":"deterministic_only_no_external_judge",
                        "schema_errors":errors,"plan_errors":plan_errors,"max_lexical_similarity":lex_score,
                        "nearest_source_id":nearest,"max_hashed_semantic_similarity":semantic_score,
                        "nearest_semantic_source_id":semantic_nearest,
                        "semantic_novelty_status":"deterministic_hashed_proxy_only_not_model_judged",
                        "answer_verification":plan["verification"],"final_pass":not errors and not plan_errors})
        bundles.append(bundle); plans.append(plan); items.append(item)
    write_jsonl(root / "enriched/retrieval_bundles.jsonl", bundles)
    write_jsonl(root / "generated/reasoning_plans.jsonl", plans)
    write_jsonl(root / "generated/items.jsonl", items)
    write_jsonl(root / "validation/item_reports.jsonl", reports)
    summary = {"source_pairs":3,"corpus_items":len(corpus),"paired_key_excerpts":sum(r.get("answer_key_excerpt") is not None for r in corpus),
               "heuristically_enriched_items":len(enriched),"question_index_items":len(qrows),"graph_index_items":len(grows),
               "retrieval_bundles":len(bundles),"generated_items":len(items),"deterministic_passed":sum(r["final_pass"] for r in reports),
               "external_model_calls":0,"external_judge_calls":0,"validation_caveat":"Answer derivations are agent-authored and deterministically audited; no independent model judge was available."}
    (root / "validation/audit_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(materialize(Path("science_olympiad/dynamic_planet_b")), indent=2))
