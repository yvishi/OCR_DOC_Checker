"""Stretch guard: recompute the percentage from the subject rows printed on the sheet.

Rows look like "<code> <SUBJECT NAME> <max> <obtained> <grade>" on every board layout we
generate (codes vary: "301", "$1", "ENG", "40"). Only used when at least 3 rows parse cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ROW = re.compile(
    r"^\s*\S{1,4}\s+(?P<name>[A-Za-z][A-Za-z &.\-]{1,40}?)\s+(?P<max>\d{2,3})\s+(?P<got>\d{1,3})\s+\S{1,3}\s*$"
)
MIN_ROWS = 3
MAX_DIFF = 0.5  # percentage points


@dataclass(frozen=True)
class SubjectRecompute:
    rows: int
    obtained: int
    maximum: int

    @property
    def percentage(self) -> float:
        return round(self.obtained / self.maximum * 100, 2)


def recompute(text: str) -> SubjectRecompute | None:
    got_total = max_total = rows = 0
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        got, maximum = int(m.group("got")), int(m.group("max"))
        if maximum not in (50, 70, 80, 100, 150, 200) or got > maximum:
            continue
        rows += 1
        got_total += got
        max_total += maximum
    if rows < MIN_ROWS or max_total == 0:
        return None
    return SubjectRecompute(rows, got_total, max_total)


def disagrees(text: str, pct: float) -> SubjectRecompute | None:
    """The recompute, if it differs from the printed percentage by more than MAX_DIFF."""
    r = recompute(text)
    if r is not None and abs(r.percentage - pct) > MAX_DIFF:
        return r
    return None
