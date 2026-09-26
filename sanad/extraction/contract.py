"""Employment-contract field extraction (explicit values only, with provenance).

Sources used, in order of confidence:
  * labeled lines ("Basic Salary: 12,000 SAR", "الراتب الأساسي: ...") and two-column table rows
  * clause sentences containing the field's keyword (or under a heading containing it) - only for
    durations, working hours/days and amounts
  * sections under a termination heading (termination terms)
  * narrative sentences, via extraction.semantic - appointment letters and ordinary contract prose
    state the same facts in sentences rather than labels ("You shall be appointed to the position
    of X"). That path is additive: a value read from a label is never replaced by one read from
    prose, and it fills fields the label path left empty.
No legal interpretation happens here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from extraction import values as readers
from extraction.clauses import ClauseSegmenter
from extraction.fields import candidate, not_extracted, resolve, resolve_list
from extraction.semantic import SEMANTIC_FIELDS, SemanticExtractor, detect_negations, sentence_allows
from extraction.text import (
    LabeledValue,
    contains_keyword,
    iter_labeled_values,
    iter_sentences,
    label_keys,
    matching_key,
)
from models.common import ErrorInfo, ResultStatus
from models.documents import DocumentStatus, ParsedDocument, SectionType
from models.extraction import (
    AllowanceValue,
    ClauseValue,
    ContractExtraction,
    ContractTypeValue,
    DateValue,
    DurationValue,
    ExtractedField,
    ExtractionMethodName,
    FieldCandidate,
    FieldStatus,
    MoneyValue,
    SourceSpan,
    WorkingDaysValue,
    WorkingHoursValue,
)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    labels: tuple[str, ...]
    keywords: tuple[str, ...] = ()

    @property
    def label_keys(self) -> set[str]:
        return {key for label in self.labels for key in label_keys(label)}

    @property
    def keyword_keys(self) -> list[str]:
        return [matching_key(keyword) for keyword in self.keywords]


CONTRACT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("employee_name", "text", ("employee name", "name of employee", "employee's name", "employee",
                                        "worker name", "اسم الموظف", "اسم العامل", "الموظف", "العامل", "الاسم")),
    FieldSpec("employer_name", "text", ("employer name", "employer", "company name", "company", "name of employer",
                                        "صاحب العمل", "اسم صاحب العمل", "اسم المنشأة", "المنشأة", "اسم الشركة", "الشركة")),
    FieldSpec("job_title", "text", ("job title", "position", "designation", "occupation", "job", "title of position",
                                    "المسمى الوظيفي", "مسمى الوظيفة", "الوظيفة", "المهنة")),
    FieldSpec("contract_type", "contract_type", ("contract type", "type of contract", "نوع العقد")),
    FieldSpec("start_date", "date", ("start date", "commencement date", "contract start date", "date of joining",
                                     "joining date", "effective date", "تاريخ بداية العقد", "تاريخ بدء العقد",
                                     "تاريخ البدء", "تاريخ بدء العمل", "تاريخ المباشرة", "تاريخ مباشرة العمل", "بداية العقد")),
    FieldSpec("end_date", "date", ("end date", "contract end date", "expiry date", "expiration date",
                                   "تاريخ نهاية العقد", "تاريخ انتهاء العقد", "نهاية العقد", "انتهاء العقد")),
    FieldSpec("contract_duration", "duration", ("contract duration", "duration of contract", "contract term",
                                                "term of contract", "مدة العقد"),
              ("contract duration", "duration of this contract", "term of this contract", "مدة العقد", "مدة هذا العقد")),
    FieldSpec("probation_period", "duration", ("probation period", "probation", "probationary period", "trial period",
                                               "فترة التجربة", "مدة التجربة", "فترة الاختبار"),
              ("probation", "trial period", "فترة التجربة", "فترة تجربة", "مدة التجربة", "فترة الاختبار")),
    FieldSpec("salary", "money", ("basic salary", "base salary", "basic wage", "basic pay", "monthly basic salary",
                                  "salary", "monthly salary", "الراتب الأساسي", "الأجر الأساسي", "الراتب", "الأجر",
                                  "الراتب الشهري", "الأجر الشهري"),
              ("basic salary", "base salary", "basic wage", "الراتب الأساسي", "الأجر الأساسي")),
    FieldSpec("total_salary", "money", ("total salary", "gross salary", "total monthly salary", "total compensation",
                                        "total monthly wage", "total wage", "monthly wage", "gross wage",
                                        "إجمالي الراتب", "الراتب الإجمالي", "إجمالي الأجر", "الأجر الإجمالي",
                                        "إجمالي الأجر الشهري", "الأجر الشهري الإجمالي"),
              ("total salary", "gross salary", "total monthly wage", "total wage",
               "إجمالي الراتب", "الراتب الإجمالي", "إجمالي الأجر الشهري")),
    FieldSpec("net_salary", "money", ("net salary", "net wage", "net pay", "net monthly salary", "net monthly wage",
                                      "صافي الراتب", "الراتب الصافي", "صافي الأجر", "الأجر الصافي"),
              ("net salary", "net wage", "net pay", "صافي الراتب", "صافي الأجر")),
    FieldSpec("housing_allowance", "allowance", ("housing allowance", "housing", "بدل السكن", "بدل سكن", "السكن"),
              ("housing allowance", "بدل السكن", "بدل سكن")),
    FieldSpec("transportation_allowance", "allowance", ("transportation allowance", "transport allowance",
                                                       "transportation", "بدل النقل", "بدل نقل", "بدل المواصلات",
                                                       "بدل مواصلات"),
              ("transportation allowance", "transport allowance", "بدل النقل", "بدل نقل", "بدل المواصلات")),
    FieldSpec("working_hours", "hours", ("working hours", "work hours", "hours of work", "daily working hours",
                                         "ساعات العمل", "عدد ساعات العمل", "ساعات العمل اليومية"),
              ("working hours", "work hours", "hours of work", "ساعات العمل", "ساعات عمل")),
    FieldSpec("working_days", "days", ("working days", "work days", "workdays", "work week", "days of work",
                                       "أيام العمل", "عدد أيام العمل"),
              ("working days", "work days", "workdays", "work week", "working week", "أيام العمل", "أيام عمل")),
    FieldSpec("annual_leave", "duration", ("annual leave", "annual vacation", "paid annual leave", "vacation",
                                           "الإجازة السنوية", "إجازة سنوية"),
              ("annual leave", "annual vacation", "إجازة سنوية", "الإجازة السنوية")),
    FieldSpec("notice_period", "duration", ("notice period", "notice", "فترة الإشعار", "مدة الإشعار", "مهلة الإشعار",
                                            "فترة الإنذار", "مدة الإنذار"),
              ("notice", "إشعار", "إنذار")),
    FieldSpec("termination_terms", "clauses", ("termination", "termination terms", "termination of contract",
                                               "إنهاء العقد", "فسخ العقد")),
    FieldSpec("work_location", "text", ("work location", "place of work", "workplace", "location", "work place",
                                        "مكان العمل", "موقع العمل", "مقر العمل")),
    FieldSpec("nationality", "text", ("nationality", "الجنسية")),
    FieldSpec("employee_id", "text", ("employee id", "employee number", "employee no", "staff id", "staff number",
                                      "الرقم الوظيفي", "رقم الموظف")),
)
_TERMINATION_HEADINGS = ("terminat", "انهاء العقد", "فسخ العقد", "انتهاء العقد", "انهاء الخدمه", "انهاء علاقه العمل")
_OTHER_ALLOWANCE_PREFIXES = ("بدل ",)
_OTHER_ALLOWANCE_SUFFIXES = (" allowance", " allowances")
_MAX_TEXT_VALUE = 150


def _looks_like_allowance_label(keys: set[str]) -> bool:
    """A generic structural cue - not a FieldSpec alias - that lets a colon-less or bilingual-glued
    label such as "Total Other Cash Allowances" / "بدل مواصلات" be recognised as a label worth
    resolving (extraction.text.iter_labeled_values's `is_label`), and also gates the "other
    allowances" bucket for any labeled value that did not match a known field."""
    return any(key.startswith(_OTHER_ALLOWANCE_PREFIXES) or key.endswith(_OTHER_ALLOWANCE_SUFFIXES) for key in keys)


class ContractExtractor:
    def __init__(self, fields: tuple[FieldSpec, ...] = CONTRACT_FIELDS,
                 semantic: SemanticExtractor | None = None,
                 segmenter: ClauseSegmenter | None = None) -> None:
        self.fields = fields
        self.semantic = semantic or SemanticExtractor()
        self.segmenter = segmenter or ClauseSegmenter()
        self._by_label: dict[str, list[FieldSpec]] = {}
        for spec in fields:
            for key in spec.label_keys:
                self._by_label.setdefault(key, []).append(spec)

    def _is_label(self, text: str) -> bool:
        keys = label_keys(text)
        return bool(keys & self._by_label.keys()) or _looks_like_allowance_label(keys)

    def extract(self, document: ParsedDocument) -> ContractExtraction:
        if not document.status.has_text:
            reason = f"document text is not available (document status: {document.status.value})"
            return ContractExtraction(
                document_id=document.document_id, filename=document.filename, document_status=document.status,
                status=ResultStatus.ERROR,
                errors=[ErrorInfo(code=f"document_{document.status.value}", stage="extraction",
                                  message=f"Fields could not be extracted: {reason}.")],
                **{spec.name: not_extracted(spec.name, reason) for spec in self.fields},
                **{name: not_extracted(name, reason) for name in SEMANTIC_FIELDS
                   if name not in {spec.name for spec in self.fields}},
                other_allowances=not_extracted("other_allowances", reason),
                benefits=not_extracted("benefits", reason),
                probation_status=not_extracted("probation_status", reason),
            )

        candidates: dict[str, list[FieldCandidate | None]] = {spec.name: [] for spec in self.fields}
        other_allowances: list[tuple[AllowanceValue, SourceSpan]] = []
        clauses: list[tuple[ClauseValue, SourceSpan]] = []
        labeled = list(iter_labeled_values(document, is_label=self._is_label))
        for item in labeled:
            specs = self._match_label(item)
            for spec in specs:
                if spec.kind == "clauses":
                    source = item.unit.source(item.method, item.source_text)
                    clauses.append((ClauseValue(heading=item.label, text=item.value), source))
                else:
                    candidates[spec.name].extend(self._from_labeled_value(spec, item))
            if not specs:
                allowance = self._other_allowance(item)
                if allowance is not None:
                    other_allowances.append(allowance)

        previous_text = ""
        for sentence in iter_sentences(document):
            sentence_key, heading_key = matching_key(sentence.text), matching_key(sentence.heading or "")
            for spec in self.fields:
                if not spec.keywords or spec.kind not in ("duration", "hours", "days", "money", "allowance"):
                    continue
                in_sentence = any(contains_keyword(sentence_key, k) for k in spec.keyword_keys)
                in_heading = spec.kind != "money" and spec.kind != "allowance" and any(
                    contains_keyword(heading_key, k) for k in spec.keyword_keys)
                if not (in_sentence or in_heading):
                    continue
                # A keyword is not enough. "During the period of probation ... 90 days' notice"
                # contains "probation" but states a notice period; crediting its 90 days to
                # probation_period is what made that field ambiguous and hid the real 180 days. The
                # same check also looks at the immediately preceding unit's text: a PDF's own layout
                # frequently splits one clause ("Deducted at a rate of (9.75%) of the basic wage ..."
                # / "... plus housing allowance = SAR 1,218.75") across two blank-line-separated
                # paragraphs, so the disqualifying cue ("Deducted", "9.75%") is not always in the same
                # sentence as the number it explains.
                context = f"{previous_text} {sentence.text}" if previous_text else sentence.text
                if not sentence_allows(spec.name, context):
                    continue
                source = sentence.source(ExtractionMethodName.CLAUSE_SENTENCE)
                candidates[spec.name].extend(self._read(spec, sentence.text, source, from_label=False))
            previous_text = sentence.text

        for section in document.sections:
            if section.section_type in (SectionType.PARAGRAPH, SectionType.LIST_ITEM) and section.title and any(
                keyword in matching_key(section.title) for keyword in _TERMINATION_HEADINGS
            ):
                source = SourceSpan(text=section.text, page_number=section.page_number, section_id=section.section_id,
                                    method=ExtractionMethodName.SECTION_CONTENT)
                clauses.append((ClauseValue(heading=section.title, text=section.text), source))

        fields: dict[str, Any] = {}
        for spec in self.fields:
            if spec.kind == "clauses":
                fields[spec.name] = resolve_list(spec.name, [c for c, _ in clauses], [s for _, s in clauses])
            else:
                fields[spec.name] = self._resolve(spec, candidates[spec.name])
        fields["other_allowances"] = resolve_list(
            "other_allowances", [a for a, _ in other_allowances], [s for _, s in other_allowances])
        self._merge_semantic(fields, self.semantic.extract_fields(document))
        self._apply_negations(fields, document)
        fields["probation_status"] = _derive_probation_status(fields["probation_period"])

        warnings = []
        status = ResultStatus.SUCCESS
        if document.status is DocumentStatus.PARTIAL:
            status = ResultStatus.PARTIAL
            warnings.append("Only part of the document could be read; fields on unread pages may be reported as not found.")
        return ContractExtraction(document_id=document.document_id, filename=document.filename,
                                  document_status=document.status, status=status, warnings=warnings,
                                  clauses=self.segmenter.segment(document), **fields)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _apply_negations(fields: dict[str, Any], document: ParsedDocument) -> None:
        """The document explicitly stating a topic does not apply (section 10) outranks both the
        label and narrative paths for that field: NOT_FOUND/AMBIGUOUS/NOT_EXTRACTED there means those
        paths could not read a value, not that the topic is silent, and the negation sentence itself
        is direct positive evidence of the correct answer. A field the label path already resolved to
        FOUND is left alone - a genuine label-vs-negation conflict is a contradiction to flag for
        review (Stage 4), not something to silently overwrite here."""
        for name, negation in detect_negations(document).items():
            existing = fields.get(name)
            if existing is not None and existing.status is FieldStatus.FOUND:
                continue
            fields[name] = negation

    @staticmethod
    def _merge_semantic(fields: dict[str, Any], semantic: dict[str, Any]) -> None:
        """Add narrative readings without overruling the label path.

        A label ("Probation Period: 90 days") is the most explicit thing a document can say, so a
        FOUND label value always wins. Narrative readings fill fields the label path left empty, and
        resolve a field the label path could only call ambiguous - recording in the notes that they did.
        """
        for name, narrative in semantic.items():
            existing = fields.get(name)
            if existing is None:
                fields[name] = narrative
                continue
            if existing.status is FieldStatus.FOUND:
                continue
            if narrative.status is not FieldStatus.FOUND:
                continue
            note = ("read from a sentence rather than a label"
                    if existing.status is FieldStatus.NOT_FOUND else
                    "read from a sentence; the labeled readings disagreed and were not used")
            fields[name] = narrative.model_copy(update={
                "notes": list(dict.fromkeys([*narrative.notes, note])),
            })

    def _match_label(self, item: LabeledValue) -> list[FieldSpec]:
        matched: list[FieldSpec] = []
        for key in item.keys:
            for spec in self._by_label.get(key, []):
                if spec not in matched:
                    matched.append(spec)
        return matched

    def _from_labeled_value(self, spec: FieldSpec, item: LabeledValue) -> list[FieldCandidate | None]:
        source = item.unit.source(item.method, item.source_text)
        if spec.kind == "text":
            if len(item.value) > _MAX_TEXT_VALUE:
                return [candidate(None, item.value, source, ["labeled value is too long to be a single value"])]
            return [candidate(item.value, item.value, source)]
        if spec.kind == "contract_type":
            normalized = readers.contract_type(item.value)
            return [candidate(ContractTypeValue(raw=item.value, normalized=normalized), item.value, source)]
        if spec.kind == "date":
            dates = readers.find_dates(item.value)
            if len(dates) != 1:
                note = "no date found in the labeled value" if not dates else "several dates are written"
                return [candidate(None, item.value, source, [note])]
            reading = dates[0]
            return [candidate(DateValue(**reading.value), reading.raw, source, reading.notes)]
        return self._read(spec, item.value, source, from_label=True)

    def _read(self, spec: FieldSpec, text: str, source: SourceSpan, *, from_label: bool) -> list[FieldCandidate | None]:
        if spec.kind == "duration":
            readings = readers.find_durations(text, allow_bare_singular=from_label)
            build: Callable[[dict], Any] = lambda v: DurationValue(**v)
        elif spec.kind == "hours":
            readings = readers.find_hours(text)
            build = lambda v: v  # merged later into WorkingHoursValue
        elif spec.kind == "days":
            readings = readers.find_working_days(text)
            build = lambda v: v
        elif spec.kind == "money":
            readings, percentages = readers.find_money(text, require_currency=not from_label)
            if from_label and percentages and not readings:
                return [candidate(None, text, source, ["written as a percentage, not an amount"])]
            build = lambda v: MoneyValue(**v)
        elif spec.kind == "allowance":
            amounts, percentages = readers.find_money(text, require_currency=not from_label)
            readings = [*amounts, *percentages]
            build = lambda v: AllowanceValue(**v)
        else:
            return []
        if not readings:
            return [candidate(None, text, source, [f"no {spec.kind} value could be read"])] if from_label else []
        return [candidate(build(r.value) if r.value is not None else None, r.raw, source, r.notes) for r in readings]

    def _resolve(self, spec: FieldSpec, items: list[FieldCandidate | None]):
        if spec.kind == "hours":
            return resolve(spec.name, items, merge=_merge_hours)
        if spec.kind == "days":
            return resolve(spec.name, items, merge=_merge_days)
        if spec.kind == "money":
            return resolve(spec.name, items, merge=_merge_money)
        if spec.kind == "allowance":
            return resolve(spec.name, items, merge=_merge_allowance)
        if spec.kind == "text":
            return resolve(spec.name, items, key=matching_key)
        if spec.kind == "contract_type":
            return resolve(spec.name, items, key=lambda v: matching_key(v.raw))
        if spec.kind == "date":
            return resolve(spec.name, items, key=lambda v: v.iso_date or matching_key(v.raw))
        return resolve(spec.name, items)

    @staticmethod
    def _other_allowance(item: LabeledValue) -> tuple[AllowanceValue, SourceSpan] | None:
        if not _looks_like_allowance_label(item.keys):
            return None
        source = item.unit.source(item.method, item.source_text)
        amounts, percentages = readers.find_money(item.value, require_currency=False)
        readings = [*amounts, *percentages]
        if len(readings) == 1:
            return AllowanceValue(name=item.label, **readings[0].value), source
        return AllowanceValue(name=item.label), source


def _derive_probation_status(probation_period: ExtractedField) -> ExtractedField:
    """A new, explicit field for "does probation apply at all", alongside (never instead of)
    `probation_period`, which keeps carrying the duration itself unchanged.

    FOUND here means "the document states a probation period" - the value is a fixed marker
    ('applicable'), not the duration, so this field is never a second, competing source for the
    length. NOT_APPLICABLE, AMBIGUOUS and NOT_FOUND are mirrored directly from `probation_period`,
    so an explicit waiver ("not subject to a probationary period") is NOT_APPLICABLE here exactly as
    it already is for `probation_period`, and a genuine conflict stays AMBIGUOUS rather than being
    guessed one way or the other.
    """
    if probation_period.status is FieldStatus.NOT_APPLICABLE:
        return ExtractedField(name="probation_status", status=FieldStatus.NOT_APPLICABLE,
                              source_text=probation_period.source_text, page_number=probation_period.page_number,
                              sources=probation_period.sources, notes=list(probation_period.notes))
    if probation_period.status is FieldStatus.FOUND:
        return ExtractedField(name="probation_status", status=FieldStatus.FOUND, value="applicable",
                              raw_value=probation_period.raw_value, normalized_value="applicable",
                              source_text=probation_period.source_text, page_number=probation_period.page_number,
                              confidence=probation_period.confidence, sources=probation_period.sources,
                              notes=["derived from the stated probation period"])
    if probation_period.status is FieldStatus.AMBIGUOUS:
        return ExtractedField(name="probation_status", status=FieldStatus.AMBIGUOUS,
                              candidates=probation_period.candidates, notes=list(probation_period.notes))
    return ExtractedField(name="probation_status", status=FieldStatus.NOT_FOUND)


def _merge_money(values: list[MoneyValue]) -> tuple[MoneyValue | None, list[str]]:
    amounts = {v.amount for v in values}
    currencies = {v.currency for v in values if v.currency}
    periods = {v.period for v in values if v.period}
    if len(amounts) > 1:
        return None, [f"different amounts are written: {', '.join(f'{a:g}' for a in sorted(amounts))}"]
    if len(currencies) > 1 or len(periods) > 1:
        return None, ["the amount is written with different currencies or periods"]
    return MoneyValue(amount=amounts.pop(), currency=next(iter(currencies), None), period=next(iter(periods), None)), []


def _merge_allowance(values: list[AllowanceValue]) -> tuple[AllowanceValue | None, list[str]]:
    merged: dict[str, Any] = {}
    for attribute in ("amount", "currency", "period", "percentage", "percentage_basis"):
        written = {getattr(v, attribute) for v in values if getattr(v, attribute) is not None}
        if attribute == "percentage_basis":
            written = {matching_key(w) for w in written}
        if len(written) > 1:
            return None, [f"different allowance {attribute.replace('_', ' ')}s are written"]
        merged[attribute] = next((getattr(v, attribute) for v in values if getattr(v, attribute) is not None), None)
    return AllowanceValue(**merged), []


def _merge_hours(values: list[dict]) -> tuple[WorkingHoursValue | None, list[str]]:
    by_period: dict[str | None, set[float]] = {}
    for value in values:
        by_period.setdefault(value["period"], set()).add(value["hours"])
    if any(len(hours) > 1 for hours in by_period.values()):
        return None, ["different working hours are written for the same period"]
    pick = lambda period: next(iter(by_period[period])) if period in by_period else None  # noqa: E731
    return WorkingHoursValue(hours_per_day=pick("day"), hours_per_week=pick("week"), hours=pick(None)), []


def _merge_days(values: list[dict]) -> tuple[WorkingDaysValue | None, list[str]]:
    counts = {v["days_per_week"] for v in values if v["days_per_week"] is not None}
    day_sets = {tuple(v["days"]) for v in values if v["days"]}
    if len(counts) > 1 or len(day_sets) > 1:
        return None, ["different working days are written"]
    count = next(iter(counts), None)
    days = list(next(iter(day_sets), ()))
    if count is not None and days and count != len(days):
        return None, [f"{count} days per week are written, but the named days are {len(days)}"]
    return WorkingDaysValue(days_per_week=count, days=days), []


def extract_contract(document: ParsedDocument) -> ContractExtraction:
    return ContractExtractor().extract(document)
