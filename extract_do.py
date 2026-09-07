#!/usr/bin/env python3
"""Extract a section or lettered item from a DO-297 PDF.

Examples
--------
    python extract_do297_reference.py DO-297.pdf "3.1"
    python extract_do297_reference.py DO-297.pdf "5.3 e"
    python extract_do297_reference.py DO-297.pdf "DO-297 5.3 e" --inspect-margins

The extractor does not rely on bookmarks or the printed table of contents.
Section numbers are recognized from their dedicated left column and the gap
before the indented title.  Printed TOC rows with dot leaders or a far-right
page number are excluded.  A whole-section request includes all descendants.

Dependency: PyMuPDF (``pip install pymupdf``)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fitz  # PyMuPDF


DEFAULT_TOP_MARGIN = 0.07
DEFAULT_BOTTOM_MARGIN = 0.07
DEFAULT_COLUMN_TOLERANCE = 0.025
DEFAULT_HEADING_GAP = 0.012
DEFAULT_ITEM_GAP = 0.010


class ReferenceExtractionError(RuntimeError):
    """Raised when the requested section or item cannot be located safely."""


@dataclass(frozen=True)
class ParsedReference:
    original: str
    section: str
    item: str | None

    @property
    def canonical(self) -> str:
        return self.section + (f" {self.item}" if self.item else "")


@dataclass(frozen=True)
class WordSegment:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class TextRow:
    page_index: int
    page_label: str
    segments: tuple[WordSegment, ...]
    page_width: float
    page_height: float

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()

    @property
    def x0(self) -> float:
        return self.segments[0].x0

    @property
    def x_ratio(self) -> float:
        return self.x0 / self.page_width if self.page_width else 0.0

    @property
    def y0(self) -> float:
        return min(segment.y0 for segment in self.segments)

    @property
    def y1(self) -> float:
        return max(segment.y1 for segment in self.segments)

    @property
    def y_ratio(self) -> float:
        center = (self.y0 + self.y1) / 2.0
        return center / self.page_height if self.page_height else 0.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def leading_gap_ratio(self) -> float:
        if len(self.segments) < 2 or not self.page_width:
            return 0.0
        gap = self.segments[1].x0 - self.segments[0].x1
        return max(0.0, gap / self.page_width)


@dataclass(frozen=True)
class SectionCandidate:
    row_index: int
    number: str


@dataclass(frozen=True)
class ItemCandidate:
    row_index: int
    letter: str


@dataclass(frozen=True)
class ExtractionResult:
    reference: str
    section: str
    item: str | None
    text: str
    pdf_pages: list[int]
    page_labels: list[str]
    next_boundary: str | None

    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_RE = re.compile(
    r"^\s*(?:DO[\s-]*297\s+)?"
    r"(?P<section>\d+(?:\.\d+)+)"
    r"(?:\s+(?P<item>[A-Za-z])\.?)?\s*$",
    re.IGNORECASE,
)
SECTION_TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?$")
LETTER_TOKEN_RE = re.compile(r"^([a-z])\.$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+(\d+)\b", re.IGNORECASE)
TOC_DOT_LEADER_RE = re.compile(r"(?:\.{2,}|\u2026)\s*\d*\s*$")
LIST_MARKER_RE = re.compile(
    r"^\s*(?:\(?\d+\)?[.)]?|\([a-z]\)|[a-z][.)]|"
    r"[\u2022\u25cf\u25aa\u2013\u2014])\s+",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*$")


def parse_reference(reference: str) -> ParsedReference:
    """Parse ``3.1``, ``5.3 e`` or the same values prefixed by DO-297."""
    match = REFERENCE_RE.fullmatch(reference)
    if not match:
        raise ValueError(
            f"Invalid reference {reference!r}; expected e.g. "
            '"3.1", "5.3 e", or "DO-297 5.3 e".'
        )
    item = match.group("item")
    return ParsedReference(
        original=reference.strip(),
        section=match.group("section"),
        item=item.lower() if item else None,
    )


def _page_label(page: fitz.Page) -> str:
    try:
        return page.get_label() or str(page.number + 1)
    except (AttributeError, RuntimeError):
        return str(page.number + 1)


def _merge_words_into_rows(
    words: Sequence[WordSegment],
    *,
    page_index: int,
    page_label: str,
    page_width: float,
    page_height: float,
) -> list[TextRow]:
    """Merge words from separate PDF blocks when they occupy one visual row."""
    sorted_words = sorted(words, key=lambda word: (word.center_y, word.x0))
    rows: list[list[WordSegment]] = []
    for word in sorted_words:
        placed = False
        for row in reversed(rows[-3:]):
            row_y0 = min(item.y0 for item in row)
            row_y1 = max(item.y1 for item in row)
            overlap = min(row_y1, word.y1) - max(row_y0, word.y0)
            if overlap >= min(word.height, row_y1 - row_y0) * 0.35:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])

    result: list[TextRow] = []
    for row in rows:
        row.sort(key=lambda word: word.x0)
        result.append(
            TextRow(
                page_index=page_index,
                page_label=page_label,
                segments=tuple(row),
                page_width=page_width,
                page_height=page_height,
            )
        )
    return sorted(result, key=lambda row: (row.y0, row.x0))


def _extract_page_rows(
    page: fitz.Page,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> list[TextRow]:
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    top = page_height * top_margin_ratio
    bottom = page_height * (1.0 - bottom_margin_ratio)
    words: list[WordSegment] = []
    for word in page.get_text("words", sort=True):
        x0, y0, x1, y1, text = word[:5]
        segment = WordSegment(
            text=str(text),
            x0=float(x0),
            y0=float(y0),
            x1=float(x1),
            y1=float(y1),
        )
        center = segment.center_y
        if segment.text.strip() and top <= center <= bottom:
            words.append(segment)
    return _merge_words_into_rows(
        words,
        page_index=page.number,
        page_label=_page_label(page),
        page_width=page_width,
        page_height=page_height,
    )


def _extract_rows(
    doc: fitz.Document,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    start_page: int = 0,
    end_page: int | None = None,
) -> list[TextRow]:
    if end_page is None:
        end_page = doc.page_count - 1
    rows: list[TextRow] = []
    for page_index in range(start_page, end_page + 1):
        rows.extend(
            _extract_page_rows(
                doc[page_index],
                top_margin_ratio=top_margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
            )
        )
    return rows


def _looks_like_toc_entry(row: TextRow) -> bool:
    if TOC_DOT_LEADER_RE.search(row.text):
        return True
    if len(row.segments) >= 3:
        last = row.segments[-1]
        if last.text.strip().isdigit() and last.x0 / row.page_width >= 0.65:
            return True
    return False


def _section_number(
    row: TextRow,
    *,
    minimum_heading_gap: float,
) -> str | None:
    if len(row.segments) < 2 or _looks_like_toc_entry(row):
        return None
    match = SECTION_TOKEN_RE.fullmatch(row.segments[0].text.strip())
    if not match or row.leading_gap_ratio < minimum_heading_gap:
        return None
    return match.group(1)


def _section_tuple(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def _section_candidates(
    rows: Sequence[TextRow],
    *,
    minimum_heading_gap: float,
) -> list[SectionCandidate]:
    result: list[SectionCandidate] = []
    for index, row in enumerate(rows):
        number = _section_number(row, minimum_heading_gap=minimum_heading_gap)
        if number is not None:
            result.append(SectionCandidate(index, number))
    return result


def _find_section_bounds(
    rows: Sequence[TextRow],
    target_section: str,
    *,
    column_tolerance: float,
    minimum_heading_gap: float,
) -> tuple[int, int, str | None]:
    candidates = _section_candidates(
        rows,
        minimum_heading_gap=minimum_heading_gap,
    )
    targets = [candidate for candidate in candidates if candidate.number == target_section]
    if not targets:
        raise ReferenceExtractionError(
            f"Section {target_section} was not found. Try --inspect-margins or "
            "reduce --minimum-heading-gap if the title is close to the number."
        )

    def column_support(candidate: SectionCandidate) -> tuple[int, int]:
        x = rows[candidate.row_index].x_ratio
        numbers = {
            other.number
            for other in candidates
            if abs(rows[other.row_index].x_ratio - x) <= column_tolerance
        }
        return len(numbers), -candidate.row_index

    start_candidate = max(targets, key=column_support)
    start_index = start_candidate.row_index
    heading_x = rows[start_index].x_ratio
    target_parts = _section_tuple(target_section)
    end_index = len(rows)
    next_boundary: str | None = None

    for candidate in candidates:
        if candidate.row_index <= start_index:
            continue
        if abs(rows[candidate.row_index].x_ratio - heading_x) > column_tolerance:
            continue
        candidate_parts = _section_tuple(candidate.number)
        if candidate.number != target_section and len(candidate_parts) <= len(target_parts):
            end_index = candidate.row_index
            next_boundary = candidate.number
            break

    # If the requested section is the final one in a chapter, stop before the
    # next CHAPTER heading instead of including it until the first x.1 heading.
    target_chapter = target_parts[0]
    for index in range(start_index + 1, end_index):
        match = CHAPTER_RE.match(rows[index].text)
        if match and int(match.group(1)) != target_chapter:
            end_index = index
            next_boundary = f"CHAPTER {match.group(1)}"
            break
    return start_index, end_index, next_boundary


def _item_letter(row: TextRow, *, minimum_item_gap: float) -> str | None:
    if len(row.segments) < 2 or row.leading_gap_ratio < minimum_item_gap:
        return None
    match = LETTER_TOKEN_RE.fullmatch(row.segments[0].text.strip())
    return match.group(1).lower() if match else None


def _find_item_bounds(
    rows: Sequence[TextRow],
    section_start: int,
    section_end: int,
    target_item: str,
    *,
    column_tolerance: float,
    minimum_item_gap: float,
) -> tuple[int, int, str | None]:
    candidates: list[ItemCandidate] = []
    for index in range(section_start + 1, section_end):
        letter = _item_letter(rows[index], minimum_item_gap=minimum_item_gap)
        if letter is not None:
            candidates.append(ItemCandidate(index, letter))
    targets = [candidate for candidate in candidates if candidate.letter == target_item]
    if not targets:
        raise ReferenceExtractionError(
            f"Item {target_item}. was not found inside the requested section. "
            "Try reducing --minimum-item-gap if the text is close to the marker."
        )

    def column_support(candidate: ItemCandidate) -> tuple[int, int]:
        x = rows[candidate.row_index].x_ratio
        letters = {
            other.letter
            for other in candidates
            if abs(rows[other.row_index].x_ratio - x) <= column_tolerance
        }
        return len(letters), -candidate.row_index

    start_candidate = max(targets, key=column_support)
    start_index = start_candidate.row_index
    marker_x = rows[start_index].x_ratio
    end_index = section_end
    next_item: str | None = None
    for candidate in candidates:
        if candidate.row_index <= start_index:
            continue
        if abs(rows[candidate.row_index].x_ratio - marker_x) <= column_tolerance:
            end_index = candidate.row_index
            next_item = candidate.letter
            break
    return start_index, end_index, next_item


def _render_rows(rows: Sequence[TextRow], *, join_wrapped_lines: bool) -> str:
    if not rows:
        return ""
    if not join_wrapped_lines:
        return "\n".join(row.text for row in rows).strip()

    typical_height = statistics.median(row.height for row in rows)
    chunks: list[str] = [rows[0].text]
    previous = rows[0]
    for row in rows[1:]:
        page_changed = row.page_index != previous.page_index
        vertical_gap = row.y0 - previous.y1 if not page_changed else 0.0
        indented = row.x_ratio > previous.x_ratio + 0.018
        current_is_structure = bool(
            _section_number(row, minimum_heading_gap=0.0)
            or _item_letter(row, minimum_item_gap=0.0)
            or LIST_MARKER_RE.match(row.text)
        )
        previous_is_heading = bool(
            _section_number(previous, minimum_heading_gap=0.0)
        )
        previous_ends_sentence = bool(
            SENTENCE_END_RE.search(previous.text.rstrip())
        )
        new_paragraph = (
            current_is_structure
            or previous_is_heading
            or (
                previous_ends_sentence
                and (indented or vertical_gap > typical_height * 0.9)
            )
        )
        if new_paragraph:
            chunks.append("\n" + row.text)
        elif chunks[-1].endswith("-") and row.text[:1].islower():
            chunks[-1] = chunks[-1][:-1] + row.text
        else:
            chunks.append(" " + row.text)
        previous = row
    return "".join(chunks).strip()


def _authenticate(doc: fitz.Document, password: str | None) -> None:
    if doc.needs_pass and (not password or not doc.authenticate(password)):
        raise ReferenceExtractionError(
            "The PDF is encrypted. Supply the correct password."
        )


def extract_reference(
    pdf_path: str | Path,
    reference: str,
    *,
    password: str | None = None,
    top_margin_ratio: float = DEFAULT_TOP_MARGIN,
    bottom_margin_ratio: float = DEFAULT_BOTTOM_MARGIN,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    minimum_heading_gap: float = DEFAULT_HEADING_GAP,
    minimum_item_gap: float = DEFAULT_ITEM_GAP,
    join_wrapped_lines: bool = False,
) -> ExtractionResult:
    """Extract an entire DO-297 section or one lettered item within it."""
    for name, value in (
        ("top_margin_ratio", top_margin_ratio),
        ("bottom_margin_ratio", bottom_margin_ratio),
        ("column_tolerance", column_tolerance),
        ("minimum_heading_gap", minimum_heading_gap),
        ("minimum_item_gap", minimum_item_gap),
    ):
        if not 0 <= value < 0.25:
            raise ValueError(f"{name} must be between 0 and 0.25")

    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        _authenticate(doc, password)
        rows = _extract_rows(
            doc,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        section_start, section_end, next_section = _find_section_bounds(
            rows,
            parsed.section,
            column_tolerance=column_tolerance,
            minimum_heading_gap=minimum_heading_gap,
        )
        selected_start, selected_end = section_start, section_end
        next_boundary = next_section
        if parsed.item is not None:
            selected_start, selected_end, next_item = _find_item_bounds(
                rows,
                section_start,
                section_end,
                parsed.item,
                column_tolerance=column_tolerance,
                minimum_item_gap=minimum_item_gap,
            )
            next_boundary = f"{next_item}." if next_item else next_section

        selected = rows[selected_start:selected_end]
        return ExtractionResult(
            reference=parsed.canonical,
            section=parsed.section,
            item=parsed.item,
            text=_render_rows(selected, join_wrapped_lines=join_wrapped_lines),
            pdf_pages=list(
                dict.fromkeys(row.page_index + 1 for row in selected)
            ),
            page_labels=list(dict.fromkeys(row.page_label for row in selected)),
            next_boundary=next_boundary,
        )
    finally:
        doc.close()


def _edge_fingerprint(text: str) -> str:
    normalized = re.sub(r"\d+", "#", text.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _suggest_margins(
    rows: Sequence[TextRow], inspection_band: float
) -> tuple[float | None, float | None]:
    fingerprint_pages: dict[str, set[int]] = {}
    for row in rows:
        if row.y_ratio <= inspection_band or row.y_ratio >= 1.0 - inspection_band:
            fingerprint_pages.setdefault(_edge_fingerprint(row.text), set()).add(
                row.page_index
            )
    repeated = {
        fingerprint
        for fingerprint, pages in fingerprint_pages.items()
        if len(pages) >= 2
    }
    top_headers = [
        row for row in rows
        if row.y_ratio <= inspection_band
        and _edge_fingerprint(row.text) in repeated
    ]
    bottom_footers = [
        row for row in rows
        if row.y_ratio >= 1.0 - inspection_band
        and _edge_fingerprint(row.text) in repeated
    ]

    suggested_top: float | None = None
    if top_headers:
        last_header = max(row.y_ratio for row in top_headers)
        body = [
            row.y_ratio for row in rows
            if last_header < row.y_ratio <= inspection_band
            and _edge_fingerprint(row.text) not in repeated
        ]
        first_body = min(body, default=min(inspection_band, last_header + 0.03))
        suggested_top = round((last_header + first_body) / 2.0, 3)

    suggested_bottom: float | None = None
    if bottom_footers:
        first_footer = min(row.y_ratio for row in bottom_footers)
        body = [
            row.y_ratio for row in rows
            if 1.0 - inspection_band <= row.y_ratio < first_footer
            and _edge_fingerprint(row.text) not in repeated
        ]
        last_body = max(
            body,
            default=max(1.0 - inspection_band, first_footer - 0.03),
        )
        suggested_bottom = round(
            1.0 - (last_body + first_footer) / 2.0,
            3,
        )
    return suggested_top, suggested_bottom


def inspect_margins(
    pdf_path: str | Path,
    reference: str,
    *,
    password: str | None = None,
    top_margin_ratio: float = DEFAULT_TOP_MARGIN,
    bottom_margin_ratio: float = DEFAULT_BOTTOM_MARGIN,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
    minimum_heading_gap: float = DEFAULT_HEADING_GAP,
    minimum_item_gap: float = DEFAULT_ITEM_GAP,
    inspection_band: float = 0.20,
    inspection_page_count: int = 5,
) -> dict:
    """Inspect edge text on pages beginning with the requested reference."""
    if not 0 < inspection_band < 0.5:
        raise ValueError("inspection_band must be between 0 and 0.5")
    if inspection_page_count < 1:
        raise ValueError("inspection_page_count must be at least 1")

    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        _authenticate(doc, password)
        locating_rows = _extract_rows(
            doc,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        try:
            section_start, section_end, _next = _find_section_bounds(
                locating_rows,
                parsed.section,
                column_tolerance=column_tolerance,
                minimum_heading_gap=minimum_heading_gap,
            )
            target_index = section_start
            if parsed.item is not None:
                target_index, _end, _next_item = _find_item_bounds(
                    locating_rows,
                    section_start,
                    section_end,
                    parsed.item,
                    column_tolerance=column_tolerance,
                    minimum_item_gap=minimum_item_gap,
                )
            first_page = locating_rows[target_index].page_index
        except ReferenceExtractionError:
            # A too-large current margin can hide the target. Retry without
            # filtering solely so the diagnostic report can still be produced.
            raw_locating_rows = _extract_rows(
                doc,
                top_margin_ratio=0.0,
                bottom_margin_ratio=0.0,
            )
            section_start, section_end, _next = _find_section_bounds(
                raw_locating_rows,
                parsed.section,
                column_tolerance=column_tolerance,
                minimum_heading_gap=minimum_heading_gap,
            )
            target_index = section_start
            if parsed.item is not None:
                target_index, _end, _next_item = _find_item_bounds(
                    raw_locating_rows,
                    section_start,
                    section_end,
                    parsed.item,
                    column_tolerance=column_tolerance,
                    minimum_item_gap=minimum_item_gap,
                )
            first_page = raw_locating_rows[target_index].page_index

        last_page = min(doc.page_count - 1, first_page + inspection_page_count - 1)
        rows = _extract_rows(
            doc,
            top_margin_ratio=0.0,
            bottom_margin_ratio=0.0,
            start_page=first_page,
            end_page=last_page,
        )
        edge_rows = [
            row for row in rows
            if row.y_ratio <= inspection_band
            or row.y_ratio >= 1.0 - inspection_band
        ]
        suggested_top, suggested_bottom = _suggest_margins(rows, inspection_band)
        entries = []
        for row in edge_rows:
            if row.y_ratio <= inspection_band:
                edge = "top"
                distance = row.y_ratio
                filtered = row.y_ratio < top_margin_ratio
            else:
                edge = "bottom"
                distance = 1.0 - row.y_ratio
                filtered = row.y_ratio > 1.0 - bottom_margin_ratio
            entries.append(
                {
                    "pdf_page": row.page_index + 1,
                    "page_label": row.page_label,
                    "edge": edge,
                    "distance_ratio": round(distance, 4),
                    "status": "FILTERED" if filtered else "KEPT",
                    "text": row.text,
                }
            )
        return {
            "reference": parsed.canonical,
            "pdf_page_range": [first_page + 1, last_page + 1],
            "current_top_margin": top_margin_ratio,
            "current_bottom_margin": bottom_margin_ratio,
            "suggested_top_margin": suggested_top,
            "suggested_bottom_margin": suggested_bottom,
            "entries": entries,
        }
    finally:
        doc.close()


def _format_margin_report(report: dict) -> str:
    lines = [
        f"Reference: {report['reference']}",
        f"PDF pages inspected: {report['pdf_page_range'][0]}-"
        f"{report['pdf_page_range'][1]}",
        f"Current margins: top={report['current_top_margin']:.3f}, "
        f"bottom={report['current_bottom_margin']:.3f}",
        "",
        "distance = line center's distance from that page edge.",
        "A line is filtered when margin > distance.",
        "",
    ]
    for entry in report["entries"]:
        lines.append(
            f"{entry['edge'].upper():6} pdf={entry['pdf_page']:>4} "
            f"label={entry['page_label']!s:<6} "
            f"distance={entry['distance_ratio']:.4f} "
            f"{entry['status']:<8} {entry['text']}"
        )
    lines.extend(["", "Automatic suggestion from repeated edge text:"])
    top = report["suggested_top_margin"]
    bottom = report["suggested_bottom_margin"]
    lines.append(
        f"  --top-margin {top:.3f}" if top is not None
        else "  top: no reliable suggestion (fewer than two matching headers)"
    )
    lines.append(
        f"  --bottom-margin {bottom:.3f}" if bottom is not None
        else "  bottom: no reliable suggestion (fewer than two matching footers)"
    )
    lines.extend(
        [
            "",
            "Verify that every header/footer line says FILTERED and every real",
            "body line says KEPT before using the suggested values.",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a section or lettered item from a DO-297 PDF."
    )
    parser.add_argument("pdf", type=Path, help="path to the DO-297 PDF")
    parser.add_argument("reference", help='e.g. "3.1" or "5.3 e"')
    parser.add_argument("--password", help="PDF password, if required")
    parser.add_argument(
        "--top-margin",
        type=float,
        default=DEFAULT_TOP_MARGIN,
        help=f"page-height fraction ignored at top (default: {DEFAULT_TOP_MARGIN})",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=DEFAULT_BOTTOM_MARGIN,
        help=f"page-height fraction ignored at bottom (default: {DEFAULT_BOTTOM_MARGIN})",
    )
    parser.add_argument(
        "--column-tolerance",
        type=float,
        default=DEFAULT_COLUMN_TOLERANCE,
        help=(
            "page-width fraction for matching peer marker columns "
            f"(default: {DEFAULT_COLUMN_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--minimum-heading-gap",
        type=float,
        default=DEFAULT_HEADING_GAP,
        help=(
            "minimum page-width gap between a section number and title "
            f"(default: {DEFAULT_HEADING_GAP})"
        ),
    )
    parser.add_argument(
        "--minimum-item-gap",
        type=float,
        default=DEFAULT_ITEM_GAP,
        help=(
            "minimum page-width gap between e. and its text "
            f"(default: {DEFAULT_ITEM_GAP})"
        ),
    )
    parser.add_argument(
        "--join-wrapped-lines",
        action="store_true",
        help="join visual wraps while preserving detected structure",
    )
    parser.add_argument(
        "--inspect-margins",
        action="store_true",
        help="show edge text, filter decisions, and suggested margins",
    )
    parser.add_argument(
        "--inspection-band",
        type=float,
        default=0.20,
        help="fraction of each page edge shown in margin inspection (default: 0.20)",
    )
    parser.add_argument(
        "--inspection-pages",
        type=int,
        default=5,
        help="number of pages inspected from the target page (default: 5)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.inspect_margins:
            report = inspect_margins(
                args.pdf,
                args.reference,
                password=args.password,
                top_margin_ratio=args.top_margin,
                bottom_margin_ratio=args.bottom_margin,
                column_tolerance=args.column_tolerance,
                minimum_heading_gap=args.minimum_heading_gap,
                minimum_item_gap=args.minimum_item_gap,
                inspection_band=args.inspection_band,
                inspection_page_count=args.inspection_pages,
            )
            print(_format_margin_report(report))
            return 0
        result = extract_reference(
            args.pdf,
            args.reference,
            password=args.password,
            top_margin_ratio=args.top_margin,
            bottom_margin_ratio=args.bottom_margin,
            column_tolerance=args.column_tolerance,
            minimum_heading_gap=args.minimum_heading_gap,
            minimum_item_gap=args.minimum_item_gap,
            join_wrapped_lines=args.join_wrapped_lines,
        )
    except (OSError, ValueError, ReferenceExtractionError, fitz.FileDataError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())