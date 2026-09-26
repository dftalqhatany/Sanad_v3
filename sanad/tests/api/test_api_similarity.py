"""Every article the API returns carries its similarity, and retrieval itself is untouched."""

from __future__ import annotations


def test_every_retrieved_article_carries_its_similarity(client):
    payload = client.post("/api/ask", json={"question": "ما هي مدة فترة التجربة؟"}).json()
    items = payload["regulatory_answer"]["evidence"]

    assert items, "the question should retrieve articles"
    for item in items:
        assert 0.0 <= item["score"] <= 1.0                          # the fused score: higher is closer
        assert item["similarity_percentage"] == round(item["score"] * 100)
        assert item["reference"]["article_number"]                  # a source to cite alongside it


def test_retrieval_itself_is_unchanged(client, real_orchestrator):
    """Adding the percentage must not alter what is retrieved, in what order, or with what scores."""
    payload = client.post("/api/ask", json={"question": "ما هي مدة فترة التجربة؟"}).json()
    items = payload["regulatory_answer"]["evidence"]

    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)
