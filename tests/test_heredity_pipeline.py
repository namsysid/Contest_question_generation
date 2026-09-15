from src.heredity.audit import answers_agree, independent_solve
from src.heredity.common import classify_topics, hash_embedding, validate_item
from src.heredity.pipeline import PILOT


def materialize(index):
    spec=PILOT[index-1]; item=dict(spec["item"])
    item.update({"id":f"heredity-b-generated-{index:03d}","response_type":spec["response_type"],
                 "difficulty":spec["difficulty"],"topics":[spec["topic"]]})
    return item


def test_topic_classification_and_embedding_are_stable():
    assert "sex_linked_pedigrees" in classify_topics("An X-linked pedigree contains a carrier")
    assert hash_embedding("DNA replication") == hash_embedding("DNA replication")
    assert abs(sum(x*x for x in hash_embedding("DNA replication"))-1) < 1e-9


def test_all_pilot_items_validate_and_independent_answers_agree():
    for index in range(1,11):
        item=materialize(index)
        assert validate_item(item)==[]
        assert answers_agree(item,independent_solve(item))


def test_visual_dependency_is_rejected():
    item=materialize(1); item["prompt"]="Using the diagram above, choose the inheritance mode."
    assert "unavailable visual dependency" in validate_item(item)


def test_reasoning_graphs_have_connected_targets():
    for spec in PILOT:
        graph=spec["reasoning_graph"]
        ids={x["id"] for x in graph["nodes"]}
        assert any(x["type"]=="Target" for x in graph["nodes"])
        assert all(x["src"] in ids and x["dst"] in ids for x in graph["edges"])
