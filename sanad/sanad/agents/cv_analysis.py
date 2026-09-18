"""CV Analysis Agent: structured interpretation of a Phase 3 CvExtraction, optionally against an explicit job.

Nothing is inferred: skills, education, certifications, experience and languages are exactly the
extracted items (with their source spans); years of experience are not computed; a missing section
is reported as not found, never as "the candidate has none". Job compatibility is deterministic,
word-based and cites the CV text behind every match. The target job must be supplied explicitly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sanad.agents.errors import AnalysisErrorCode, analysis_error
from sanad.agents.shared import fact_from_field, label, status_counts
from sanad.extraction.text import matching_key
from sanad.models.analysis import (
    AnalysisStatus,
    CompatibilityStatus,
    CvAnalysisResult,
    CvEvidence,
    CvFieldReview,
    CvObservation,
    CvProfile,
    JobCompatibilityResult,
    RequirementAssessment,
    TargetJob,
)
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.extraction import CvExtraction, FieldStatus, SourceSpan

logger = logging.getLogger(__name__)

CV_FIELDS = ("name", "email", "phone", "location", "summary", "skills", "education", "certifications",
             "work_experience", "languages")
_STOPWORDS = {"of", "in", "and", "the", "a", "an", "for", "with", "to", "at", "on", "or", "degree",
              "في", "و", "من", "على", "الى", "عن", "او"}
_LANGUAGE_ALIASES = {
    "arabic": {"arabic", "عربيه", "عربي"},
    "english": {"english", "انجليزيه", "انكليزيه", "انجليزي"},
    "french": {"french", "فرنسيه", "فرنسي"},
    "urdu": {"urdu", "اورديه", "اوردو"},
    "hindi": {"hindi", "هنديه", "هندي"},
    "spanish": {"spanish", "اسبانيه", "اسباني"},
    "german": {"german", "المانيه", "الماني"},
    "chinese": {"chinese", "صينيه", "صيني"},
    "turkish": {"turkish", "تركيه", "تركي"},
    "filipino": {"filipino", "tagalog", "فلبينيه"},
}
_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)


@dataclass
class _Text:
    cv_field: str
    text: str
    source: SourceSpan


def tokens(text: str) -> list[str]:
    result = []
    for token in matching_key(text).split():
        if token in _STOPWORDS:
            continue
        if token.startswith("وال") and len(token) > 4:
            token = token[3:]
        elif token.startswith("ال") and len(token) > 3:
            token = token[2:]
        result.append(token)
    return result


def _scripts(text: str) -> set[str]:
    return {name for name, pattern in (("arabic", _ARABIC), ("latin", _LATIN)) if pattern.search(text)}


class CvAnalysisAgent:
    def analyze(self, cv: CvExtraction, target_job: TargetJob | None = None) -> CvAnalysisResult:
        if not isinstance(cv, CvExtraction):
            if not all(hasattr(cv, a) for a in ("document_id", "filename", "document_status", "status")):
                raise TypeError(f"Expected a CvExtraction, got {type(cv).__name__}.")
            error = analysis_error(AnalysisErrorCode.INVALID_INPUT, "input", f"Expected a CvExtraction, got {type(cv).__name__}.")
            return self._empty(cv, AnalysisStatus.INVALID_INPUT, [error], "No analysis was performed: invalid input.")
        if target_job is not None and not isinstance(target_job, TargetJob):
            raise TypeError("target_job must be a TargetJob")
        try:
            result = self._analyze(cv, target_job)
        except Exception as exc:
            logger.warning("CV analysis failed for document %s: %s", cv.document_id, type(exc).__name__)
            error = analysis_error(AnalysisErrorCode.ANALYSIS_FAILED, "analysis",
                                   "The CV analysis failed unexpectedly; no findings are reported.", exc)
            return self._empty(cv, AnalysisStatus.ANALYSIS_ERROR, [error], "The analysis could not be completed.")
        logger.info("CV analysis for document %s: status=%s", cv.document_id, result.status.value)
        return result

    # ------------------------------------------------------------------ workflow
    def _analyze(self, cv: CvExtraction, target_job: TargetJob | None) -> CvAnalysisResult:
        if cv.status is ResultStatus.ERROR:
            errors = [analysis_error(AnalysisErrorCode.EXTRACTION_NOT_USABLE, "input",
                                     "The CV extraction has no usable text, so it cannot be analysed."), *cv.errors]
            return self._empty(cv, AnalysisStatus.INVALID_INPUT, errors,
                               "No analysis was performed because the CV text was not available.")
        fields = cv.fields()
        items = lambda name: list(fields[name].value or []) if fields[name].status is FieldStatus.FOUND else []  # noqa: E731
        profile = CvProfile(
            name=fact_from_field("name", cv.name), email=fact_from_field("email", cv.email),
            phone=fact_from_field("phone", cv.phone), location=fact_from_field("location", cv.location),
            summary=fact_from_field("summary", cv.summary),
            skills=items("skills"), education=items("education"), certifications=items("certifications"),
            work_experience=items("work_experience"), languages=items("languages"),
        )
        reviews = [
            CvFieldReview(field=name, extraction_status=fields[name].status,
                          item_count=len(fields[name].value) if isinstance(fields[name].value, list) else None,
                          notes=list(fields[name].notes))
            for name in CV_FIELDS
        ]
        observations = self._observations(cv, profile)
        compatibility = self._compatibility(cv, profile, target_job) if target_job is not None else None

        status = AnalysisStatus.PARTIAL if cv.status is ResultStatus.PARTIAL else AnalysisStatus.SUCCESS
        warnings = ["The CV was only partly readable; sections on unread pages appear as not found."] \
            if cv.status is ResultStatus.PARTIAL else []
        counts = status_counts(review.extraction_status for review in reviews)
        found = [r.field for r in reviews if r.extraction_status is FieldStatus.FOUND]
        missing = [label(r.field) for r in reviews if r.extraction_status is FieldStatus.NOT_FOUND]
        review = [label(r.field) for r in reviews if r.extraction_status is FieldStatus.AMBIGUOUS]
        summary = f"{len(found)} of {len(CV_FIELDS)} CV fields were found."
        if missing:
            summary += f" Not found in the extracted CV: {', '.join(missing)}."
        if review:
            summary += f" Ambiguous, needs review: {', '.join(review)}."
        if compatibility is not None:
            summary += " " + compatibility.summary
        return CvAnalysisResult(
            status=status, document_id=cv.document_id, filename=cv.filename, document_status=cv.document_status,
            extraction_status=cv.status, overall_summary=summary, status_counts=counts, profile=profile,
            field_reviews=reviews, observations=observations, job_compatibility=compatibility, warnings=warnings,
        )

    @staticmethod
    def _observations(cv: CvExtraction, profile: CvProfile) -> list[CvObservation]:
        observations = []
        for name in ("skills", "education", "work_experience"):
            if getattr(cv, name).status is FieldStatus.NOT_FOUND:
                observations.append(CvObservation(
                    code="section_not_found", field=name,
                    message=f"No {label(name)} section was found in the extracted CV (this is not a statement that "
                            "the person has none)."))
        for name, field in cv.fields().items():
            if field.status is FieldStatus.AMBIGUOUS:
                observations.append(CvObservation(
                    code="ambiguous_field", field=name, sources=[c.source for c in field.candidates],
                    message=f"The CV states several different values for {label(name)}: "
                            + "; ".join(f"'{c.raw_value}'" for c in field.candidates)))
        for entry in profile.work_experience:
            if entry.start is None:
                observations.append(CvObservation(code="experience_without_dates", field="work_experience",
                                                  sources=[entry.source],
                                                  message=f"No dates are written for the experience entry '{entry.header}'."))
            if entry.title is None:
                observations.append(CvObservation(code="experience_title_not_explicit", field="work_experience",
                                                  sources=[entry.source],
                                                  message=f"The job title is not explicitly separated in '{entry.header}'."))
        return observations

    # ------------------------------------------------------------------ job compatibility
    def _compatibility(self, cv: CvExtraction, profile: CvProfile, job: TargetJob) -> JobCompatibilityResult:
        texts = {
            "skills": [_Text("skills", s.text, s.source) for s in profile.skills],
            "summary": [_Text("summary", s.text, s) for s in profile.summary.sources] if profile.summary.value else [],
            "work_experience": [_Text("work_experience", t, e.source) for e in profile.work_experience
                                for t in [e.header, *e.details]],
            "education": [_Text("education", e.text, e.source) for e in profile.education],
            "certifications": [_Text("certifications", c.name, c.source) for c in profile.certifications],
            "languages": [_Text("languages", f"{l.language} {l.level or ''}".strip(), l.source) for l in profile.languages],
        }
        status_of = {name: getattr(cv, name).status for name in texts}
        plan = [("job_title", job.title, "required", "work_experience", ("summary",))]
        plan += [("skill", s, "required", "skills", ("work_experience", "summary", "certifications", "education"))
                 for s in job.required_skills]
        plan += [("skill", s, "preferred", "skills", ("work_experience", "summary", "certifications", "education"))
                 for s in job.preferred_skills]
        plan += [("education", s, "required", "education", ("certifications",)) for s in job.required_education]
        plan += [("certification", s, "required", "certifications", ("education", "skills", "summary"))
                 for s in job.required_certifications]
        plan += [("language", s, "required", "languages", ()) for s in job.required_languages]

        assessments = [self._assess(kind, requirement, importance, primary, secondary, texts, status_of)
                       for kind, requirement, importance, primary, secondary in plan]
        if job.minimum_years_experience is not None:
            assessments.append(self._years(job, profile))

        counts = status_counts(a.status for a in assessments)
        required = [a for a in assessments if a.importance == "required"]
        if any(a.status is CompatibilityStatus.MISSING_REQUIREMENT for a in required):
            overall = "gaps_identified"
        elif any(a.status is CompatibilityStatus.INSUFFICIENT_INFORMATION for a in required):
            overall = "insufficient_information"
        elif all(a.status is CompatibilityStatus.EXPLICIT_MATCH for a in required):
            overall = "required_requirements_explicitly_met"
        else:
            overall = "partially_evidenced"
        described = ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in counts.items())
        summary = (f"Compared with the {'contract job title' if job.source == 'contract_job_title' else 'provided job'} "
                   f"'{job.title}': {described}.")
        return JobCompatibilityResult(target_job=job, requirements=assessments, status_counts=counts,
                                      overall=overall, summary=summary)

    def _assess(self, kind: str, requirement: str, importance: str, primary: str, secondary: tuple[str, ...],
                texts: dict[str, list[_Text]], status_of: dict[str, FieldStatus]) -> RequirementAssessment:
        searched = [primary, *secondary]
        base = dict(requirement_type=kind, requirement=requirement, importance=importance, searched_fields=searched)
        wanted = tokens(requirement)
        if kind == "language":
            wanted = [self._language(requirement)] if self._language(requirement) else wanted
        if not wanted:
            return RequirementAssessment(**base, status=CompatibilityStatus.INSUFFICIENT_INFORMATION,
                                         explanation="The requirement has no comparable words.")

        def coverage(item: _Text) -> set[str]:
            item_tokens = set(tokens(item.text))
            if kind == "language":
                item_tokens = {self._language(t) or t for t in item_tokens} | item_tokens
            return {w for w in wanted if w in item_tokens}

        def evidence(matches: list[_Text]) -> list[CvEvidence]:
            return [CvEvidence(cv_field=m.cv_field, text=m.text, source=m.source) for m in matches]

        full_primary = [t for t in texts[primary] if coverage(t) == set(wanted)]
        if full_primary:
            return RequirementAssessment(**base, status=CompatibilityStatus.EXPLICIT_MATCH, cv_evidence=evidence(full_primary),
                                         explanation=f"'{requirement}' is stated in the CV's {label(primary)}: "
                                                     + "; ".join(f"'{t.text}'" for t in full_primary) + ".")
        full_secondary = [t for name in secondary for t in texts[name] if coverage(t) == set(wanted)]
        if full_secondary:
            return RequirementAssessment(
                **base, status=CompatibilityStatus.PARTIAL_MATCH, cv_evidence=evidence(full_secondary),
                explanation=f"'{requirement}' is mentioned in the CV ({', '.join(sorted({t.cv_field for t in full_secondary}))}) "
                            f"but not in its {label(primary)} section.")
        partial_primary = [t for t in texts[primary] if coverage(t)]
        if partial_primary:
            words = sorted({w for t in partial_primary for w in coverage(t)})
            return RequirementAssessment(
                **base, status=CompatibilityStatus.PARTIAL_MATCH, cv_evidence=evidence(partial_primary),
                explanation=f"The CV's {label(primary)} covers only part of '{requirement}' (matching words: "
                            f"{', '.join(words)}): " + "; ".join(f"'{t.text}'" for t in partial_primary) + ".")
        if status_of[primary] is not FieldStatus.FOUND:
            return RequirementAssessment(
                **base, status=CompatibilityStatus.INSUFFICIENT_INFORMATION,
                explanation=f"The CV's {label(primary)} section was {status_of[primary].value.replace('_', ' ')}, "
                            f"so '{requirement}' cannot be checked.")
        searched_texts = [t.text for name in searched for t in texts[name]]
        if kind != "language" and not (_scripts(requirement) & set().union(*(_scripts(t) for t in searched_texts))):
            return RequirementAssessment(
                **base, status=CompatibilityStatus.INSUFFICIENT_INFORMATION,
                explanation=f"'{requirement}' and the CV are written in different languages, so they were not compared.")
        return RequirementAssessment(
            **base, status=CompatibilityStatus.MISSING_REQUIREMENT,
            explanation=f"'{requirement}' is not stated in the CV ({', '.join(label(s) for s in searched)}). This only "
                        "means the CV does not mention it.")

    @staticmethod
    def _years(job: TargetJob, profile: CvProfile) -> RequirementAssessment:
        dated = [e for e in profile.work_experience if e.start is not None]
        written = "; ".join(f"'{e.header}': {e.start} - {e.end or '?'}" for e in dated) or "no dated entries"
        return RequirementAssessment(
            requirement_type="experience_years", requirement=f"{job.minimum_years_experience:g} years",
            importance="required", status=CompatibilityStatus.INSUFFICIENT_INFORMATION,
            searched_fields=["work_experience"],
            explanation=f"Years of experience are not calculated automatically. Dates written in the CV: {written}.")

    @staticmethod
    def _language(text: str) -> str | None:
        for token in tokens(text):
            for canonical, aliases in _LANGUAGE_ALIASES.items():
                if token in aliases:
                    return canonical
        return None

    @staticmethod
    def _empty(cv, status: AnalysisStatus, errors: list[ErrorInfo], summary: str) -> CvAnalysisResult:
        return CvAnalysisResult(status=status, document_id=cv.document_id, filename=cv.filename,
                                document_status=cv.document_status, extraction_status=cv.status,
                                overall_summary=summary, errors=errors)
