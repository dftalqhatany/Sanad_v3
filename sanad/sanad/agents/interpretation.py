"""Interpretation of one contract fact against retrieved regulatory evidence.

Only runs after the existing RAG returned relevant evidence. Two interpreters:
  * NoInterpreter (default): no automated legal conclusion; the finding stays 'requires_review'.
  * LLMEvidenceInterpreter: asks an LLM for a structured verdict that must quote the retrieved Arabic
    article text and the extracted contract value. Every quote, evidence id and article number is
    verified; an ungrounded 'compliant'/'non_compliant' answer is downgraded to 'requires_review'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from sanad.agents.errors import AnalysisErrorCode, InterpretationError
from sanad.agents.regulatory import RegulatoryTopic
from sanad.extraction.text import matching_key, to_ascii_digits
from sanad.models.analysis import (
    DocumentFact,
    EvidenceQuote,
    EvidenceReference,
    FindingStatus,
    Interpretation,
    InterpretationMethod,
)

MIN_QUOTE_CHARS = 12


@dataclass
class InterpretationRequest:
    field: str
    topic: RegulatoryTopic
    fact: DocumentFact
    evidence: list[EvidenceReference]
    related_facts: list[DocumentFact] = field(default_factory=list)


class EvidenceInterpreter(Protocol):
    name: str

    def interpret(self, request: InterpretationRequest) -> Interpretation: ...


class NoInterpreter:
    name = "none"

    def interpret(self, request: InterpretationRequest) -> Interpretation:
        articles = ", ".join(ref.citation for ref in request.evidence)
        return Interpretation(
            method=InterpretationMethod.NONE,
            assessment=FindingStatus.REQUIRES_REVIEW,
            explanation=("Relevant regulatory evidence was retrieved, but no automated interpretation is configured, "
                         f"so no compliance conclusion was made. Compare the contract value with: {articles}."),
            cited_evidence_ids=[ref.evidence_id for ref in request.evidence],
        )


# --------------------------------------------------------------------------- LLM
class LLMClient(Protocol):
    model: str

    def complete_json(self, *, system: str, user: str) -> str: ...


class OpenAIChatClient:
    """Minimal OpenAI chat client returning a JSON string. The key is never logged or repr'd."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout_s: float = 60.0) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self._api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"OpenAIChatClient(model={self.model!r})"

    def complete_json(self, *, system: str, user: str) -> str:
        from openai import OpenAI  # imported lazily: the package is optional for Sanad

        client = OpenAI(api_key=self._api_key, timeout=self.timeout_s)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return response.choices[0].message.content or ""


SYSTEM_PROMPT = """You assist a human reviewer of Saudi employment contracts. You are not a lawyer and you do not give legal advice.
Rules:
1. Use ONLY the contract facts and the regulatory evidence given in the user message. Do not use any other knowledge of Saudi law.
2. Never invent articles, article numbers, legal text, numbers, conditions or contract terms. Do not assume facts that are not given.
3. The Arabic article text is the legal reference. The English text is a machine translation that may be wrong; never rely on it alone.
4. Answer "compliant" or "non_compliant" only if an Arabic passage that you copy exactly from a cited article directly supports that conclusion for the given contract value.
5. Answer "requires_review" if the evidence is relevant but the outcome depends on conditions, exceptions or facts that are not given.
6. Answer "insufficient_evidence" if the supplied evidence does not address the question.
7. Reply with one JSON object only, with exactly these keys:
   {"assessment": "compliant" | "non_compliant" | "requires_review" | "insufficient_evidence",
    "explanation": "<short explanation that refers only to the supplied evidence>",
    "cited_evidence_ids": ["<evidence_id>", ...],
    "evidence_quotes": [{"evidence_id": "<evidence_id>", "quote": "<exact Arabic passage>"}],
    "contract_quote": "<text copied exactly from the contract fact>"}"""


class LLMInterpretationOutput(BaseModel):
    assessment: Literal["compliant", "non_compliant", "requires_review", "insufficient_evidence"]
    explanation: str = Field(min_length=1, max_length=4000)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    evidence_quotes: list[dict[str, str]] = Field(default_factory=list)
    contract_quote: str | None = None


_ARTICLE_NUMBER = re.compile(r"(?:article|المادة|مادة)\s*[(\[]?\s*(\d{1,3})", re.IGNORECASE)


class LLMEvidenceInterpreter:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def name(self) -> str:
        return f"llm:{self.client.model}"

    def interpret(self, request: InterpretationRequest) -> Interpretation:
        user = json.dumps(build_payload(request), ensure_ascii=False)
        try:
            raw = self.client.complete_json(system=SYSTEM_PROMPT, user=user)
        except Exception as exc:
            raise InterpretationError(AnalysisErrorCode.LLM_REQUEST_FAILED, "The LLM interpretation request failed.",
                                      type(exc).__name__) from None
        try:
            output = LLMInterpretationOutput.model_validate_json(_strip_code_fence(raw))
        except (ValidationError, ValueError, TypeError):
            raise InterpretationError(AnalysisErrorCode.LLM_OUTPUT_INVALID,
                                      "The LLM returned output that does not match the required JSON structure.") from None
        return verify_output(output, request, model=self.client.model)


def build_payload(request: InterpretationRequest) -> dict:
    def fact(item: DocumentFact) -> dict:
        return {"field": item.field, "value": item.value, "raw_value": item.raw_value, "source_text": item.source_text}

    return {
        "task": "Assess one extracted contract field against the supplied Saudi Labor Law evidence only.",
        "regulatory_questions": list(request.topic.questions),
        "contract_fact": fact(request.fact),
        "related_contract_facts": [fact(item) for item in request.related_facts],
        "evidence": [
            {
                "evidence_id": ref.evidence_id,
                "citation": ref.citation,
                "article_number": ref.article_number,
                "article_name": ref.article_name,
                "arabic_text": ref.arabic_text,
                "english_machine_translation": ref.english_text,
            }
            for ref in request.evidence
        ],
    }


def verify_output(output: LLMInterpretationOutput, request: InterpretationRequest, *, model: str | None) -> Interpretation:
    by_id = {ref.evidence_id: ref for ref in request.evidence}
    unknown = sorted({i for i in output.cited_evidence_ids if i not in by_id})
    quotes: list[EvidenceQuote] = []
    for item in output.evidence_quotes:
        evidence_id, text = str(item.get("evidence_id", "")), str(item.get("quote", ""))
        reference = by_id.get(evidence_id)
        if reference is None:
            unknown.append(evidence_id)
        quotes.append(_verified_quote(evidence_id, text, reference))
    unknown = sorted(set(unknown))

    assessment = FindingStatus(output.assessment)
    notes: list[str] = []
    grounded = False
    cited = [i for i in dict.fromkeys(output.cited_evidence_ids) if i in by_id]
    if assessment in (FindingStatus.COMPLIANT, FindingStatus.NON_COMPLIANT):
        problems = []
        if unknown:
            problems.append(f"it cites evidence that was not supplied ({', '.join(unknown)})")
        if not any(q.verified and q.language == "ar" and q.evidence_id in cited for q in quotes):
            problems.append("no exact quote from the Arabic text of a cited article supports it")
        if not _contract_quote_matches(output.contract_quote, request.fact):
            problems.append("its contract quote does not match the extracted contract fact")
        foreign_articles = _foreign_article_numbers(output.explanation, request.evidence)
        if foreign_articles:
            problems.append(f"its explanation mentions article numbers not in the retrieved evidence ({', '.join(foreign_articles)})")
        if problems:
            notes.append(f"The LLM proposed '{output.assessment}', which was not accepted because " + "; ".join(problems) + ".")
            assessment = FindingStatus.REQUIRES_REVIEW
        else:
            grounded = True
    elif unknown:
        notes.append(f"Ignored citations of evidence that was not supplied: {', '.join(unknown)}.")

    return Interpretation(
        method=InterpretationMethod.LLM,
        assessment=assessment,
        explanation=output.explanation,
        cited_evidence_ids=cited,
        evidence_quotes=quotes,
        contract_quote=output.contract_quote,
        grounded=grounded,
        model=model,
        notes=notes,
    )


def _verified_quote(evidence_id: str, text: str, reference: EvidenceReference | None) -> EvidenceQuote:
    key = matching_key(text)
    if reference is not None and len(key) >= MIN_QUOTE_CHARS:
        if key in matching_key(reference.arabic_text or ""):
            return EvidenceQuote(evidence_id=evidence_id, quote=text, verified=True, language="ar")
        if key in matching_key(reference.english_text or ""):
            return EvidenceQuote(evidence_id=evidence_id, quote=text, verified=True, language="en")
    return EvidenceQuote(evidence_id=evidence_id, quote=text, verified=False)


def _contract_quote_matches(quote: str | None, fact: DocumentFact) -> bool:
    key = matching_key(quote or "")
    if not key:
        return False
    texts = [fact.raw_value or "", fact.source_text or "", *(source.text for source in fact.sources)]
    return any(key in matching_key(text) for text in texts if text)


def _foreign_article_numbers(explanation: str, evidence: list[EvidenceReference]) -> list[str]:
    allowed = {str(ref.article_number) for ref in evidence if ref.article_number is not None}
    mentioned = {match.group(1) for match in _ARTICLE_NUMBER.finditer(to_ascii_digits(explanation))}
    return sorted(mentioned - allowed)


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text
