"""CV field extraction (explicit information only, with provenance).

Contact details come from labels or unambiguous patterns (e-mail, phone). Skills, education,
certifications, experience and languages are only read from the matching CV sections: a skill
mentioned inside a job description is NOT listed as a skill, years of experience are NOT computed,
and job titles are only split from organisations when written as "X at Y" / "X في Y" / "X لدى Y".
No compatibility analysis happens here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sanad.extraction import values as readers
from sanad.extraction.fields import candidate, not_extracted, resolve, resolve_list
from sanad.extraction.text import BULLET_CHARS, TextUnit, iter_labeled_values, iter_lines, label_keys, matching_key, to_ascii_digits
from sanad.models.common import ErrorInfo, ResultStatus
from sanad.models.documents import DocumentStatus, ParsedDocument
from sanad.models.extraction import (
    CertificationEntry,
    CvExtraction,
    EducationEntry,
    ExperienceEntry,
    ExtractionMethodName,
    LanguageEntry,
    ListItem,
)

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "professional profile", "objective", "career objective",
                "about me", "نبذة", "نبذة مختصرة", "نبذة عني", "الملخص", "ملخص", "الملخص المهني", "الهدف الوظيفي"),
    "skills": ("skills", "technical skills", "core skills", "key skills", "core competencies", "competencies",
               "المهارات", "مهارات", "المهارات التقنية", "المهارات الفنية", "المهارات الشخصية"),
    "experience": ("experience", "work experience", "professional experience", "employment history", "work history",
                   "career history", "الخبرات", "الخبرة", "الخبرات العملية", "الخبرة العملية", "الخبرة المهنية",
                   "الخبرات المهنية", "الخبرات السابقة"),
    "education": ("education", "academic background", "academic qualifications", "educational background",
                  "التعليم", "المؤهلات العلمية", "المؤهل العلمي", "المؤهلات الأكاديمية"),
    "certifications": ("certifications", "certificates", "licenses", "licenses and certifications",
                       "certifications and licenses", "professional certifications", "courses and certifications",
                       "الشهادات", "الشهادات المهنية", "الدورات والشهادات", "الشهادات والدورات", "الرخص المهنية"),
    "languages": ("languages", "language skills", "اللغات"),
    "_other": ("references", "projects", "awards", "achievements", "hobbies", "interests", "volunteering",
               "volunteer experience", "publications", "personal information", "personal details", "contact",
               "contact information", "courses", "المراجع", "المعرفون", "المشاريع", "الجوائز", "الإنجازات",
               "الهوايات", "الاهتمامات", "الأنشطة", "العمل التطوعي", "المعلومات الشخصية", "البيانات الشخصية",
               "بيانات التواصل", "الدورات"),
}
_SECTION_BY_KEY = {matching_key(k): name for name, keywords in SECTION_KEYWORDS.items() for k in keywords}

NAME_LABELS = ("name", "full name", "الاسم", "الاسم الكامل")
PHONE_LABELS = ("phone", "mobile", "tel", "telephone", "cell", "phone number", "mobile number", "contact number",
                "الجوال", "الهاتف", "رقم الجوال", "رقم الهاتف", "الموبايل", "جوال", "هاتف")
LOCATION_LABELS = ("location", "address", "city", "current location", "residence", "العنوان", "المدينة", "الموقع",
                   "مكان الإقامة", "الإقامة")
EMAIL_LABELS = ("email", "e-mail", "email address", "البريد الإلكتروني", "البريد", "الإيميل")

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w@])\+?\d[\d \-().]{7,}\d(?![\w@])")
_YEAR_RANGE = re.compile(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\b")
_ITEM_SPLIT = re.compile(r"\s*(?:[,،;|•●▪]|\s-\s)\s*")
_PRESENT = r"present|current|now|to date|حتى الآن|حتى الان|الآن|الان|حاليا|حالياً|حتى تاريخه"
_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
           r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|"
           r"سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر")
_DATE = rf"(?:(?:{_MONTHS})\.?\s+)?(?:\d{{1,2}}/)?(?:19|20)\d{{2}}"
_DATE_RANGE = re.compile(rf"(?P<start>{_DATE})\s*(?:[-–—]|to|until|إلى|الى|حتى)\s*(?P<end>{_PRESENT}|{_DATE})", re.IGNORECASE)
_TITLE_AT = re.compile(r"^(?P<title>.+?)\s+(?:at|@|في|لدى)\s+(?P<org>.+)$", re.IGNORECASE)
_DEGREE = re.compile(r"\b(?:bachelor(?:'s)?|b\.?sc|master(?:'s)?|m\.?sc|mba|ph\.?d|doctorate|diploma|associate degree|"
                     r"high school)\b|بكالوريوس|ماجستير|دكتوراه|دبلوم|الثانوية|ثانوية عامة", re.IGNORECASE)
_INSTITUTION = re.compile(r"\b(?:university|college|institute|school|academy)\b|جامعة|كلية|معهد|مدرسة|أكاديمية",
                          re.IGNORECASE)
_LANGUAGE_LEVEL = re.compile(r"^(?P<language>[^()\-–:]+?)\s*(?:\((?P<paren>[^)]+)\)|[-–:]\s*(?P<rest>.+))$")
_CERT_YEAR = re.compile(r"\s*(?:\(\s*(?P<y1>(?:19|20)\d{2})\s*\)|[,\-–]\s*(?P<y2>(?:19|20)\d{2}))\s*$")


@dataclass
class _Group:
    name: str
    heading: TextUnit | None
    level: int | None
    lines: list[TextUnit] = field(default_factory=list)


class CvExtractor:
    def extract(self, document: ParsedDocument) -> CvExtraction:
        names = ("name", "email", "phone", "location", "summary", "skills", "education", "certifications",
                 "work_experience", "languages")
        if not document.status.has_text:
            reason = f"document text is not available (document status: {document.status.value})"
            return CvExtraction(
                document_id=document.document_id, filename=document.filename, document_status=document.status,
                status=ResultStatus.ERROR,
                errors=[ErrorInfo(code=f"document_{document.status.value}", stage="extraction",
                                  message=f"Fields could not be extracted: {reason}.")],
                **{name: not_extracted(name, reason) for name in names},
            )

        lines = list(iter_lines(document))
        groups, title = self._groups(lines)
        labeled = list(iter_labeled_values(document))

        fields = {
            "name": self._name(labeled, title),
            "email": self._email(lines),
            "phone": self._phone(lines, labeled),
            "location": resolve("location", [candidate(i.value, i.value, i.unit.source(i.method, i.source_text))
                                             for i in labeled if i.keys & _keys(LOCATION_LABELS)], key=matching_key),
            "summary": self._summary(groups, labeled),
            "skills": self._items("skills", groups, labeled),
            "education": self._education(groups),
            "certifications": self._certifications(groups),
            "work_experience": self._experience(groups),
            "languages": self._languages(groups, labeled),
        }
        status, warnings = ResultStatus.SUCCESS, []
        if document.status is DocumentStatus.PARTIAL:
            status = ResultStatus.PARTIAL
            warnings.append("Only part of the document could be read; fields on unread pages may be reported as not found.")
        return CvExtraction(document_id=document.document_id, filename=document.filename,
                            document_status=document.status, status=status, warnings=warnings, **fields)

    # ------------------------------------------------------------------ sections
    @staticmethod
    def _section_of(unit: TextUnit) -> str | None:
        text = unit.text.strip().rstrip(":：").strip()
        if len(text) > 60:
            return None
        return _SECTION_BY_KEY.get(matching_key(text))

    def _groups(self, lines: list[TextUnit]) -> tuple[list[_Group], TextUnit | None]:
        groups: list[_Group] = []
        current: _Group | None = None
        title: TextUnit | None = None
        for unit in lines:
            section = self._section_of(unit) if unit.table_id is None else None
            if section is not None:
                current = None if section == "_other" else _Group(section, unit, unit.heading_level)
                if current is not None:
                    groups.append(current)
                continue
            if unit.is_heading:
                if current is None and not groups and title is None:
                    title = unit  # first heading before any CV section, e.g. the candidate's name
                    continue
                if (current is not None and unit.heading_level is not None and current.level is not None
                        and unit.heading_level <= current.level):
                    current = None  # an unrecognised section at the same level ends the current one
                    continue
            if current is not None:
                current.lines.append(unit)
        return groups, title

    @staticmethod
    def _group_lines(groups: list[_Group], name: str) -> list[TextUnit]:
        return [line for group in groups if group.name == name for line in group.lines]

    # ------------------------------------------------------------------ contact
    @staticmethod
    def _name(labeled, title: TextUnit | None):
        items = [candidate(i.value, i.value, i.unit.source(i.method, i.source_text))
                 for i in labeled if i.keys & _keys(NAME_LABELS)]
        if not items and title is not None:
            text = title.text.strip()
            words = text.split()
            if 2 <= len(words) <= 5 and len(text) <= 60 and not re.search(r"[\d@:/]", text):
                items.append(candidate(text, text, title.source(ExtractionMethodName.DOCUMENT_TITLE),
                                       ["read from the document title"]))
        return resolve("name", items, key=matching_key)

    @staticmethod
    def _email(lines: list[TextUnit]):
        items = []
        for unit in lines:
            for match in _EMAIL.finditer(unit.text):
                items.append(candidate(match.group(), match.group(), unit.source(ExtractionMethodName.PATTERN)))
        return resolve("email", items, key=str.lower)

    @staticmethod
    def _phone(lines: list[TextUnit], labeled):
        items = []
        for item in labeled:
            if item.keys & _keys(PHONE_LABELS):
                digits = re.sub(r"\D", "", to_ascii_digits(item.value))
                value = item.value if 8 <= len(digits) <= 15 else None
                notes = [] if value else ["labeled phone value does not look like a phone number"]
                items.append(candidate(value, item.value, item.unit.source(item.method, item.source_text), notes))
        for unit in lines:
            ascii_text = to_ascii_digits(unit.text)
            for match in _PHONE.finditer(ascii_text):
                raw = unit.text[match.start():match.end()]
                digits = re.sub(r"\D", "", match.group())
                if not 9 <= len(digits) <= 15 or _YEAR_RANGE.search(match.group()) or "/" in match.group():
                    continue
                items.append(candidate(raw, raw, unit.source(ExtractionMethodName.PATTERN)))
        return resolve("phone", items, key=lambda value: re.sub(r"\D", "", to_ascii_digits(value)))

    # ------------------------------------------------------------------ section content
    def _summary(self, groups: list[_Group], labeled):
        lines = self._group_lines(groups, "summary")
        if lines:
            text = "\n".join(line.text for line in lines)
            sources = [line.source(ExtractionMethodName.SECTION_CONTENT) for line in lines]
            field_ = resolve_list("summary", [text], sources)
            return field_.model_copy(update={"value": text, "raw_value": text})
        items = [candidate(i.value, i.value, i.unit.source(i.method, i.source_text))
                 for i in labeled if self._label_section(i) == "summary"]
        return resolve("summary", items)

    @staticmethod
    def _label_section(item) -> str | None:
        return next((_SECTION_BY_KEY[key] for key in item.keys if key in _SECTION_BY_KEY), None)

    def _split_items(self, units: list[TextUnit]) -> list[ListItem]:
        items = []
        for unit in units:
            cells = unit.cells if unit.table_id is not None else [unit.text]
            for cell in cells:
                for part in _ITEM_SPLIT.split(cell):
                    text = part.strip().lstrip(BULLET_CHARS).strip()
                    if text:
                        items.append(ListItem(text=text, source=unit.source(ExtractionMethodName.SECTION_CONTENT)))
        return items

    def _items(self, name: str, groups: list[_Group], labeled):
        units = self._group_lines(groups, name)
        items = self._split_items(units)
        for item in labeled:
            if self._label_section(item) == name:
                unit = TextUnit(text=item.source_text, page_number=item.unit.page_number,
                                section_id=item.unit.section_id, table_id=item.unit.table_id)
                for part in _ITEM_SPLIT.split(item.value):
                    if part.strip():
                        items.append(ListItem(text=part.strip(), source=unit.source(ExtractionMethodName.LABELED_LINE)))
        return resolve_list(name, items, [i.source for i in items])

    def _languages(self, groups: list[_Group], labeled):
        base = self._items("languages", groups, labeled)
        entries = []
        for item in base.value or []:
            match = _LANGUAGE_LEVEL.match(item.text)
            if match:
                level = match.group("paren") or match.group("rest")
                entries.append(LanguageEntry(language=match.group("language").strip(), level=level.strip(), source=item.source))
            else:
                entries.append(LanguageEntry(language=item.text, source=item.source))
        return resolve_list("languages", entries, [e.source for e in entries])

    def _certifications(self, groups: list[_Group]):
        entries = []
        for unit in self._group_lines(groups, "certifications"):
            text = unit.text.strip().lstrip(BULLET_CHARS).strip()
            match = _CERT_YEAR.search(to_ascii_digits(text))
            year = int(match.group("y1") or match.group("y2")) if match else None
            name = text[: match.start()].strip() if match else text
            if name:
                entries.append(CertificationEntry(name=name, year=year, source=unit.source(ExtractionMethodName.SECTION_CONTENT)))
        return resolve_list("certifications", entries, [e.source for e in entries])

    def _education(self, groups: list[_Group]):
        blocks: list[list[TextUnit]] = []
        for unit in self._group_lines(groups, "education"):
            has_degree = bool(_DEGREE.search(unit.text))
            if not blocks or (has_degree and any(_DEGREE.search(u.text) for u in blocks[-1])):
                blocks.append([unit])
            else:
                blocks[-1].append(unit)
        entries = []
        for block in blocks:
            text = "\n".join(u.text for u in block)
            segments = [s.strip() for s in re.split(r"[,،|\n]|\s[-–]\s", text) if s.strip()]
            degree = next((s for s in segments if _DEGREE.search(s)), None)
            institution = next((s for s in segments if _INSTITUTION.search(s)), None)
            start_year = end_year = year = None
            range_match = re.search(r"((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})", to_ascii_digits(text))
            if range_match:
                start_year, end_year = int(range_match.group(1)), int(range_match.group(2))
            else:
                years = readers.years_in(text)
                year = years[0] if len(years) == 1 else None
            source = block[0].source(ExtractionMethodName.SECTION_CONTENT)  # the entry's first line, verbatim
            entries.append(EducationEntry(text=text, degree=degree, institution=institution, start_year=start_year,
                                          end_year=end_year, year=year, source=source))
        return resolve_list("education", entries, [e.source for e in entries])

    def _experience(self, groups: list[_Group]):
        units = self._group_lines(groups, "experience")
        if not units:
            return resolve_list("work_experience", [], [])
        date_rows = [i for i, u in enumerate(units) if _DATE_RANGE.search(u.text)]
        starts: list[int] = []
        if date_rows:
            for row in date_rows:
                header_row = row
                previous = row - 1
                if (previous >= 0 and previous not in date_rows and not units[previous].is_list_item
                        and (not starts or previous > starts[-1])):
                    header_row = previous  # "Title at Company" line written just before the dates
                if not starts or header_row > starts[-1]:
                    starts.append(header_row)
            if starts[0] != 0:
                starts.insert(0, 0)
        else:  # without dates, entries can only be told apart by sub-headings
            starts = [i for i, u in enumerate(units) if u.is_heading] or [0]
            if starts[0] != 0:
                starts.insert(0, 0)
        entries = []
        for position, start in enumerate(starts):
            stop = starts[position + 1] if position + 1 < len(starts) else len(units)
            block = units[start:stop]
            entries.append(self._experience_entry(block))
        return resolve_list("work_experience", entries, [e.source for e in entries])

    @staticmethod
    def _experience_entry(block: list[TextUnit]) -> ExperienceEntry:
        header = block[0].text.strip()
        start = end = None
        is_current = None
        details = []
        header_without_dates = header
        for unit in block:
            match = _DATE_RANGE.search(unit.text)
            if match and start is None:
                start, end = match.group("start"), match.group("end")
                is_current = bool(re.fullmatch(_PRESENT, end.strip(), re.IGNORECASE))
                if unit is block[0]:
                    header_without_dates = (unit.text[: match.start()] + unit.text[match.end():]).strip(" ()|,-–")
                continue
            if unit is not block[0]:
                details.append(unit.text.strip().lstrip(BULLET_CHARS).strip())
        title = organization = None
        at = _TITLE_AT.match(header_without_dates)
        if at:
            title, organization = at.group("title").strip(), at.group("org").strip(" ,")
        source = block[0].source(ExtractionMethodName.SECTION_CONTENT)  # the entry's first line, verbatim
        return ExperienceEntry(header=header, title=title, organization=organization, start=start, end=end,
                               is_current=is_current, details=[d for d in details if d], source=source)


def _keys(labels: tuple[str, ...]) -> set[str]:
    return {key for label in labels for key in label_keys(label)}


def extract_cv(document: ParsedDocument) -> CvExtraction:
    return CvExtractor().extract(document)
