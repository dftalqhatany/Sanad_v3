"""Value readers return only what is literally written; conflicts are surfaced, never resolved by guessing."""

from __future__ import annotations

import pytest

from sanad.extraction import values


def _durations(text, **kwargs):
    return [(r.value and (r.value["count"], r.value["unit"], r.value["qualifier"]), r.raw, r.notes)
            for r in values.find_durations(text, **kwargs)]


@pytest.mark.parametrize("text, expected", [
    ("a probation period of 90 days", [((90, "day", None), "90 days", [])]),
    ("ninety (90) days from the start date", [((90, "day", None), "ninety (90) days", [])]),
    ("twenty-one working days", [((21, "day", "working"), "twenty-one working days", [])]),
    ("one hundred and eighty days", [((180, "day", None), "one hundred and eighty days", [])]),
    ("فترة تجربة مدتها تسعون يوماً", [((90, "day", None), "تسعون يوماً", [])]),
    ("إجازة سنوية مدتها واحد وعشرون يوماً", [((21, "day", None), "واحد وعشرون يوماً", [])]),
    ("مدة الإشعار (٦٠) يوماً", [((60, "day", None), "(٦٠) يوماً", [])]),
    ("ثلاثة أشهر", [((3, "month", None), "ثلاثة أشهر", [])]),
    ("شهرين", [((2, "month", None), "شهرين", [])]),
    ("سنة واحدة", [((1, "year", None), "سنة واحدة", [])]),
    ("ثلاثة عشر يوماً", [((13, "day", None), "ثلاثة عشر يوماً", [])]),
])
def test_durations_written_in_digits_or_words(text, expected):
    assert _durations(text) == expected


def test_conflicting_or_alternative_numbers_are_not_resolved():
    [conflict] = values.find_durations("ninety (60) days")
    assert conflict.value is None and "conflicting numbers" in conflict.notes[0]
    readings = values.find_durations("between 90 to 180 days")
    assert readings[0].value is None and "range" in readings[0].notes[0]


def test_units_used_as_rates_or_without_numbers_are_not_durations():
    assert values.find_durations("8 hours per day and paid each month") == []
    assert values.find_durations("30 days per contract year")[0].value["count"] == 30
    assert len(values.find_durations("30 days per contract year")) == 1
    assert values.find_durations("مدة العقد سنة") == []
    [bare] = values.find_durations("سنة", allow_bare_singular=True)
    assert bare.value["count"] == 1 and bare.notes == ["singular unit written without a number"]


def test_money_amounts_currency_period_and_percentages():
    [amount], [] = values.find_money("12,000 SAR per month", require_currency=False)
    assert amount.value == {"amount": 12000.0, "currency": "SAR", "period": "monthly"} and amount.raw == "12,000 SAR"
    [amount], _ = values.find_money("٩٬٥٠٠ ريال سعودي شهرياً", require_currency=True)
    assert amount.value == {"amount": 9500.0, "currency": "SAR", "period": "monthly"} and amount.raw == "٩٬٥٠٠ ريال"
    [], [percent] = values.find_money("25% من الراتب الأساسي", require_currency=False)
    assert percent.value == {"percentage": 25.0, "percentage_basis": "من الراتب الأساسي"}
    assert values.find_money("Article 90 applies from 15/10/2026", require_currency=True) == ([], [])
    [no_currency], _ = values.find_money("5000", require_currency=False)
    assert no_currency.value["currency"] is None  # never assumed


@pytest.mark.parametrize("text, iso, calendar, note", [
    ("15/10/2026", "2026-10-15", "gregorian", None),
    ("01/03/2026", None, "gregorian", "day/month order cannot be determined"),
    ("2026-10-15", "2026-10-15", "gregorian", None),
    ("15 October 2026", "2026-10-15", "gregorian", None),
    ("١٥/١٠/٢٠٢٦", "2026-10-15", "gregorian", None),
    ("15/03/1448هـ", None, "hijri", "Hijri date; not converted"),
    ("31/02/2026", None, "gregorian", None),
])
def test_dates(text, iso, calendar, note):
    [reading] = values.find_dates(text)
    assert reading.value["iso_date"] == iso and reading.value["calendar"] == calendar and reading.raw == text
    if note:
        assert note in reading.notes


def test_hours_and_days():
    hours = [(r.value["hours"], r.value["period"]) for r in values.find_hours("8 hours per day, 40 hours per week")]
    assert hours == [(8, "day"), (40, "week")]
    assert [(r.value["hours"], r.value["period"]) for r in values.find_hours("ثماني ساعات يومياً")] == [(8, "day")]
    assert values.find_hours("8 or 9 hours per day")[0].value is None
    [days] = values.find_working_days("من الأحد إلى الخميس")
    assert days.value["days"] == ["sunday", "monday", "tuesday", "wednesday", "thursday"]
    [count] = values.find_working_days("5 days per week")
    assert count.value == {"days_per_week": 5, "days": []}
    [listed] = values.find_working_days("Sunday, Monday and Tuesday")
    assert listed.value["days"] == ["sunday", "monday", "tuesday"]
