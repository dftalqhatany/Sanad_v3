"""CV Analysis Agent: explicit CV content only, job compatibility only against an explicit target job."""

from __future__ import annotations

import logging

import pytest

from agents import CvAnalysisAgent
from agents.cv_analysis import tokens
from extraction import extract_cv
from models.analysis import AnalysisStatus, CompatibilityStatus, CvAnalysisResult, RequirementAssessment, TargetJob
from models.common import ResultStatus
from models.extraction import FieldStatus

EXPLICIT_MATCH, PARTIAL = CompatibilityStatus.EXPLICIT_MATCH, CompatibilityStatus.PARTIAL_MATCH
MISSING, INSUFFICIENT = CompatibilityStatus.MISSING_REQUIREMENT, CompatibilityStatus.INSUFFICIENT_INFORMATION


def requirement(result: CvAnalysisResult, text: str) -> RequirementAssessment:
    return next(r for r in result.job_compatibility.requirements if r.requirement == text)


# --------------------------------------------------------------------------- profile
def test_profile_is_exactly_the_extracted_cv_content_with_sources(cv_en, parsed):
    result = CvAnalysisAgent().analyze(cv_en)
    document = parsed["sample_cv_en.docx"]

    assert result.status is AnalysisStatus.SUCCESS and result.document_type == "cv"
    profile = result.profile
    assert profile.name.value == "Jordan Sample" and profile.email.value == "jordan.sample@example.com"
    assert profile.phone.value == "+966 55 000 1234" and profile.location.value == "Riyadh, Saudi Arabia"
    assert [s.text for s in profile.skills] == ["Python", "SQL", "Power BI", "Data Visualization"]
    assert profile.skills == cv_en.skills.value
    assert profile.education == cv_en.education.value and profile.certifications == cv_en.certifications.value
    assert profile.work_experience == cv_en.work_experience.value and profile.languages == cv_en.languages.value
    for item in [*profile.skills, *profile.education, *profile.certifications, *profile.work_experience,
                 *profile.languages]:
        assert item.source.text in document.full_text
    assert profile.email.source_text in document.full_text
    assert result.job_compatibility is None  # no target job was given: nothing is inferred
    assert [r.field for r in result.field_reviews] == ["name", "email", "phone", "location", "summary", "skills",
                                                      "education", "certifications", "work_experience", "languages"]
    assert result.status_counts == {"found": 10}
    assert CvAnalysisResult.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


def test_arabic_cv_profile(cv_ar):
    result = CvAnalysisAgent().analyze(cv_ar)
    assert result.status is AnalysisStatus.SUCCESS
    assert result.profile.skills == cv_ar.skills.value and result.profile.languages == cv_ar.languages.value


def test_missing_sections_are_not_found_and_nothing_is_invented(make_cv):
    _, cv = make_cv(lambda d: (d.add_heading("Casey Placeholder", 0), d.add_paragraph("Email: casey@example.com")))
    result = CvAnalysisAgent().analyze(cv)

    profile = result.profile
    assert profile.skills == [] and profile.education == [] and profile.work_experience == []
    assert profile.certifications == [] and profile.languages == []
    reviews = {r.field: r.extraction_status for r in result.field_reviews}
    assert reviews["skills"] is FieldStatus.NOT_FOUND and reviews["email"] is FieldStatus.FOUND
    codes = {(o.code, o.field) for o in result.observations}
    assert {("section_not_found", "skills"), ("section_not_found", "education"),
            ("section_not_found", "work_experience")} <= codes
    assert all("not a statement that the person has none" in o.message for o in result.observations
               if o.code == "section_not_found")
    assert "Not found in the extracted CV" in result.overall_summary


def test_ambiguous_fields_and_undated_experience_are_observations(make_cv):
    def build(d):
        d.add_heading("Casey Placeholder", 0)
        d.add_paragraph("Email: first@example.com")
        d.add_paragraph("Alternative: second@example.com")
        d.add_heading("Experience", 1)
        d.add_paragraph("Analyst, Example Co.")
        d.add_paragraph("Used Python and SQL daily for five years.")

    _, cv = make_cv(build)
    result = CvAnalysisAgent().analyze(cv)
    codes = {o.code for o in result.observations}
    assert {"ambiguous_field", "experience_without_dates", "experience_title_not_explicit"} <= codes
    ambiguous = next(o for o in result.observations if o.code == "ambiguous_field")
    assert ambiguous.field == "email" and len(ambiguous.sources) == 2
    assert result.profile.email.extraction_status is FieldStatus.AMBIGUOUS and result.profile.email.value is None
    assert result.profile.skills == []  # "Used Python" in a job description is not a skills section


# --------------------------------------------------------------------------- job compatibility
def test_compatibility_statuses_against_an_explicit_job(cv_en):
    job = TargetJob(title="Data Analyst", required_skills=["Python", "SQL", "Kubernetes", "Data Engineering"],
                    preferred_skills=["dashboards"], required_education=["Computer Science"],
                    required_certifications=["Certified Data Professional"], required_languages=["English", "French"],
                    minimum_years_experience=3)
    result = CvAnalysisAgent().analyze(cv_en, job)
    compatibility = result.job_compatibility

    assert requirement(result, "Data Analyst").status is EXPLICIT_MATCH
    assert requirement(result, "Python").status is EXPLICIT_MATCH
    assert requirement(result, "Kubernetes").status is MISSING
    assert "only means the CV does not mention it" in requirement(result, "Kubernetes").explanation
    assert requirement(result, "Data Engineering").status is PARTIAL  # "Data" only
    assert requirement(result, "dashboards").status is PARTIAL  # in experience details, not in the skills section
    assert requirement(result, "dashboards").cv_evidence[0].cv_field == "work_experience"
    assert requirement(result, "Computer Science").status is EXPLICIT_MATCH
    assert requirement(result, "Certified Data Professional").status is EXPLICIT_MATCH
    assert requirement(result, "English").status is EXPLICIT_MATCH
    assert requirement(result, "French").status is MISSING
    years = next(r for r in compatibility.requirements if r.requirement_type == "experience_years")
    assert years.status is INSUFFICIENT and "not calculated" in years.explanation and "Jan 2021" in years.explanation

    assert compatibility.overall == "gaps_identified"
    assert compatibility.target_job == job and "not a hiring decision" in compatibility.note
    assert sum(compatibility.status_counts.values()) == len(compatibility.requirements)


def test_every_match_cites_cv_text_from_the_document(cv_en, parsed):
    job = TargetJob(title="Data Analyst", required_skills=["Python", "Power BI"], required_languages=["Arabic"])
    result = CvAnalysisAgent().analyze(cv_en, job)
    document = parsed["sample_cv_en.docx"]
    for assessment in result.job_compatibility.requirements:
        if assessment.status in (EXPLICIT_MATCH, PARTIAL):
            assert assessment.cv_evidence
            assert all(e.source.text in document.full_text for e in assessment.cv_evidence)
    assert result.job_compatibility.overall == "required_requirements_explicitly_met"


def test_missing_cv_section_is_insufficient_information_not_missing(make_cv):
    _, cv = make_cv(lambda d: (d.add_heading("Casey Placeholder", 0), d.add_paragraph("Email: casey@example.com")))
    job = TargetJob(title="Accountant", required_skills=["IFRS"], required_languages=["Arabic"],
                    required_education=["Accounting"])
    result = CvAnalysisAgent().analyze(cv, job)
    assert {r.status for r in result.job_compatibility.requirements} == {INSUFFICIENT}
    assert result.job_compatibility.overall == "insufficient_information"


def test_arabic_cv_languages_match_english_requirements_but_other_scripts_are_not_guessed(cv_ar):
    job = TargetJob(title="Software Engineer", required_languages=["Arabic", "English"],
                    required_education=["Computer Science"])
    result = CvAnalysisAgent().analyze(cv_ar, job)
    assert requirement(result, "Arabic").status is EXPLICIT_MATCH
    assert requirement(result, "English").status is EXPLICIT_MATCH
    assert requirement(result, "Computer Science").status is INSUFFICIENT
    assert "different languages" in requirement(result, "Computer Science").explanation


def test_arabic_requirements_match_arabic_cv_text(cv_ar):
    title = cv_ar.work_experience.value[0].title
    assert title  # e.g. "محاسبة"
    result = CvAnalysisAgent().analyze(cv_ar, TargetJob(title=title))
    assert requirement(result, title).status is EXPLICIT_MATCH


def test_target_job_must_be_explicit():
    with pytest.raises(ValueError):
        TargetJob(title="")
    with pytest.raises(ValueError):
        TargetJob(title="Analyst", inferred_from_cv=True)
    with pytest.raises(ValueError):
        TargetJob(title="Analyst", minimum_years_experience=-1)
    with pytest.raises(ValueError, match="must cite"):
        RequirementAssessment(requirement_type="skill", requirement="Python", importance="required",
                              status=EXPLICIT_MATCH, explanation="x")


def test_tokens_normalise_arabic_articles_and_stopwords():
    assert tokens("Bachelor of Science in Computer Science") == ["bachelor", "science", "computer", "science"]
    assert tokens("اللغة الإنجليزية") == ["لغه", "انجليزيه"]


# --------------------------------------------------------------------------- statuses
def test_wrong_type_is_invalid_input(contract_en):
    result = CvAnalysisAgent().analyze(contract_en)
    assert result.status is AnalysisStatus.INVALID_INPUT and result.profile is None
    with pytest.raises(TypeError):
        CvAnalysisAgent().analyze("a cv")
    with pytest.raises(TypeError):
        CvAnalysisAgent().analyze(None)


def test_target_job_must_be_a_target_job(cv_en):
    with pytest.raises(TypeError):
        CvAnalysisAgent().analyze(cv_en, {"title": "Analyst"})


def test_unreadable_cv_is_invalid_input(parsed):
    cv = extract_cv(parsed["sample_scanned.pdf"])
    assert cv.status is ResultStatus.ERROR
    result = CvAnalysisAgent().analyze(cv)
    assert result.status is AnalysisStatus.INVALID_INPUT and result.errors[0].code == "extraction_not_usable"


def test_unexpected_failure_is_analysis_error(cv_en, monkeypatch):
    monkeypatch.setattr(CvAnalysisAgent, "_observations", staticmethod(lambda *a: 1 / 0))
    result = CvAnalysisAgent().analyze(cv_en)
    assert result.status is AnalysisStatus.ANALYSIS_ERROR and result.errors[0].exception_type == "ZeroDivisionError"


def test_logs_contain_no_personal_information(cv_en, caplog):
    with caplog.at_level(logging.DEBUG, logger="sanad"):
        CvAnalysisAgent().analyze(cv_en, TargetJob(title="Data Analyst"))
    assert "Jordan" not in caplog.text and "example.com" not in caplog.text and "+966" not in caplog.text
