import pytest
from pydantic import ValidationError

from models import ErrorInfo, RegulatoryEvidenceResult, RegulatoryQuery, ResultStatus


def test_query_is_normalised_and_validated():
    query = RegulatoryQuery(question="  What is the probation period?  ", contract_context="   ", clause_name=" Probation ")
    assert query.question == "What is the probation period?"
    assert query.contract_context is None
    assert query.clause_name == "Probation"
    with pytest.raises(ValidationError):
        RegulatoryQuery(question="   ")
    with pytest.raises(ValidationError):
        RegulatoryQuery(question="q", unexpected="field")


def test_result_status_invariants_prevent_silent_failures():
    query = RegulatoryQuery(question="q")
    with pytest.raises(ValidationError, match="requires at least one ErrorInfo"):
        RegulatoryEvidenceResult(status=ResultStatus.ERROR, query=query, legacy_entry_point="x")
    with pytest.raises(ValidationError, match="requires evidence"):
        RegulatoryEvidenceResult(status=ResultStatus.SUCCESS, query=query, legacy_entry_point="x")
    ok = RegulatoryEvidenceResult(
        status=ResultStatus.ERROR, query=query, legacy_entry_point="x",
        errors=[ErrorInfo(code="vector_db_unreachable", stage="retrieval", message="down")],
    )
    assert ok.model_dump(mode="json")["status"] == "error"
