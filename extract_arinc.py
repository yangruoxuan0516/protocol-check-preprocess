#!/usr/bin/env python3
"""Preprocess an ARINC 664 PDF into retrieval-friendly JSONL chunks.

The extractor processes the complete PDF and emits three content types only:

    text
    table
    figure_caption

It is layout-aware.  It uses PyMuPDF text dictionaries, font/bbox information,
repeated edge-text fingerprints, visible TOC evidence, optional PDF bookmarks,
Page.find_tables(), and raster-image blocks.  Table text and figure captions are
kept separate from normal body text.

Examples
--------
    python extract_arinc.py ARINC664P2.pdf --document-id ARINC664P2
    python extract_arinc.py ARINC664P2.pdf --document-id ARINC664P2 -o out.jsonl
    python extract_arinc.py ARINC664P2.pdf --document-id ARINC664P2 --inspect-margins
    python extract_arinc.py ARINC664P2.pdf --document-id ARINC664P2 --inspect-sections

Dependency
----------
    pip install "pymupdf>=1.23"

Python 3.9+.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Constants and regular expressions
# ---------------------------------------------------------------------------

SECTION_NUMBER_PATTERN = r"\d+(?:\.\d+)*"
SECTION_ROW_RE = re.compile(
    r"^\s*(?P<number>" + SECTION_NUMBER_PATTERN + r")\s+(?P<title>\S.*)$"
)
SECTION_NUMBER_ONLY_RE = re.compile(
    r"^\s*(?P<number>" + SECTION_NUMBER_PATTERN + r")\s*$"
)
TOC_ENTRY_RE = re.compile(
    r"^\s*(?P<number>" + SECTION_NUMBER_PATTERN + r")\s+(?P<rest>.+?)\s*$"
)
APPENDIX_RE = re.compile(
    r"^\s*(?P<kind>APPENDIX|ATTACHMENT)\s+"
    r"(?P<identifier>[A-Z0-9]+)\b(?P<title>.*)$",
    re.IGNORECASE,
)
FIGURE_CAPTION_RE = re.compile(
    r"^\s*FIGURE\s+[A-Z0-9]+(?:[.\-][A-Z0-9]+)*\b.*$",
    re.IGNORECASE,
)
TABLE_CAPTION_RE = re.compile(
    r"^\s*TABLE\s+[A-Z0-9]+(?:[.\-][A-Z0-9]+)*\b.*$",
    re.IGNORECASE,
)
LIST_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[a-zA-Z][.)]|"
    r"\((?:[a-zA-Z]|\d+)\)|"
    r"\d+[.)]|"
    r"[\u2022\u25cf\u25aa\u25e6\u2013\u2014-]"
    r")\s+"
)
ROMAN_PAGE_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
PLAIN_PAGE_RE = re.compile(r"^\d+$")
RUNNING_ARINC_RE = re.compile(
    r"^ARINC\s+(?:SPECIFICATION|REPORT|CHARACTERISTIC)\b.*\bPAGE\s+\S+\s*$",
    re.IGNORECASE,
)
RUNNING_SECTION_RE = re.compile(
    r"^\s*" + SECTION_NUMBER_PATTERN + r"\s+[A-Z][A-Z0-9 /&(),.:'\-\u2013\u2014]+\s*$"
)
FRONT_TITLE_WORDS = {
    "DISCLAIMER",
    "FOREWORD",
    "PREFACE",
    "INTRODUCTION TO ARINC STANDARDS",
}
NAVIGATION_TITLES = {
    "TABLE OF CONTENTS",
    "CONTENTS",
    "LIST OF TABLES",
    "LIST OF FIGURES",
    "LIST OF ILLUSTRATIONS",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ExtractionError(RuntimeError):
    """Raised when the PDF cannot be safely converted into a usable corpus."""


@dataclass(frozen=True)
class TextLine:
    page_index: int
    block_index: int
    line_index: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    font_size: float
    bold: bool
    italic: bool

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.page_index, self.block_index, self.line_index)

    @property
    def x_ratio(self) -> float:
        if not self.page_width:
            return 0.0
        return self.x0 / self.page_width

    @property
    def y_ratio(self) -> float:
        if not self.page_height:
            return 0.0
        return ((self.y0 + self.y1) / 2.0) / self.page_height

    @property
    def center_x_ratio(self) -> float:
        if not self.page_width:
            return 0.0
        return ((self.x0 + self.x1) / 2.0) / self.page_width

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass
class TableRegion:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    caption_keys: Set[Tuple[int, int, int]] = field(default_factory=set)
    fallback_used: bool = False


@dataclass(frozen=True)
class ImageRegion:
    page_index: int
    bbox: Tuple[float, float, float, float]


@dataclass
class FigureCaption:
    page_index: int
    bbox: Tuple[float, float, float, float]
    text: str
    line_keys: Set[Tuple[int, int, int]]
    associated_image: bool


@dataclass
class HeadingCandidate:
    page_index: int
    section_number: Optional[str]
    section_title: str
    line_keys: Set[Tuple[int, int, int]]
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    bold: bool
    x_ratio: float = 0.0
    score: float = 0.0
    evidence: Tuple[str, ...] = ()
    evidence_similarity: float = 0.0
    kind: str = "numeric"
    source_typo: Optional[str] = None


@dataclass
class PageData:
    page_index: int
    width: float
    height: float
    lines: List[TextLine]
    is_navigation: bool = False
    tables: List[TableRegion] = field(default_factory=list)
    images: List[ImageRegion] = field(default_factory=list)
    figure_captions: List[FigureCaption] = field(default_factory=list)


@dataclass
class ContentUnit:
    content_type: str
    section_number: Optional[str]
    section_title: Optional[str]
    page_start: int
    page_end: int
    text: str
    text_run_id: int


@dataclass
class ChunkRecord:
    chunk_id: str
    document: str
    content_type: str
    section_number: Optional[str]
    section_title: Optional[str]
    page_start: int
    page_end: int
    text: str

    def to_ordered_dict(self) -> Dict[str, object]:
        # Dict insertion order is part of the output contract here.
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "content_type": self.content_type,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }


class WarningCollector:
    def __init__(self) -> None:
        self._warnings: List[str] = []
        self._seen: Set[str] = set()

    def add(self, message: str) -> None:
        if message not in self._seen:
            self._seen.add(message)
            self._warnings.append(message)

    @property
    def items(self) -> List[str]:
        return list(self._warnings)


# ---------------------------------------------------------------------------
# Basic text/layout helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Clean extraction artifacts without rewriting document wording."""
    text = text.replace("\x00", "")
    text = text.replace("\ufffe", "").replace("\uffff", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_for_match(text: str) -> str:
    text = _clean_text(text).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_similarity(a: str, b: str) -> float:
    a_tokens = set(_normalize_for_match(a).split())
    b_tokens = set(_normalize_for_match(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / float(len(a_tokens | b_tokens))


def _section_tuple(number: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def _repair_missing_dot_section_number(
    number: str,
    title: str,
    toc_evidence: Dict[str, List[str]],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
) -> Tuple[str, str, Optional[str]]:
    """Repair a source heading like ``4.7.2 2 Title`` conservatively.

    A repair is made only when the first title token is an integer and the
    resulting dotted number (for example ``4.7.2.2``) is corroborated by the
    document's visible TOC and/or PDF bookmarks with a reasonably similar
    title.  This is intended for obvious source/PDF numbering typos, not as a
    general rewrite rule.
    """
    match = re.match(r"^(?P<child>\d+)\s+(?P<rest>\S.*)$", title)
    if not match:
        return number, title, None

    repaired = number + "." + match.group("child")
    repaired_title = _clean_text(match.group("rest"))
    supporting_titles: List[str] = []
    supporting_titles.extend(toc_evidence.get(repaired, []))
    supporting_titles.extend(
        item[0] for item in bookmark_numeric.get(repaired, [])
    )
    if not supporting_titles:
        return number, title, None

    best_similarity = max(
        (_title_similarity(repaired_title, item) for item in supporting_titles),
        default=0.0,
    )
    if best_similarity < 0.35:
        return number, title, None

    return repaired, repaired_title, number + " " + match.group("child")


def _bbox_union(boxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _bbox_area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _line_overlaps_region(
    line: TextLine,
    region: Tuple[float, float, float, float],
    threshold: float = 0.45,
) -> bool:
    line_box = (line.x0, line.y0, line.x1, line.y1)
    area = _bbox_area(line_box)
    if area <= 0.0:
        return False
    return _bbox_intersection_area(line_box, region) / area >= threshold


def _horizontal_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return overlap / denom


def _page_label(page: fitz.Page) -> str:
    try:
        return page.get_label() or str(page.number + 1)
    except (AttributeError, RuntimeError):
        return str(page.number + 1)


def _extract_page_lines(page: fitz.Page) -> List[TextLine]:
    page_dict = page.get_text("dict", sort=True)
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    result: List[TextLine] = []

    for block_index, block in enumerate(page_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            text = _clean_text("".join(str(span.get("text", "")) for span in spans))
            if not text:
                continue
            x0, y0, x1, y1 = map(float, line.get("bbox", (0, 0, 0, 0)))
            font_size = max(
                (float(span.get("size", 0.0)) for span in spans),
                default=0.0,
            )
            bold = any(
                "bold" in str(span.get("font", "")).casefold()
                or bool(int(span.get("flags", 0)) & 16)
                for span in spans
            )
            italic = any(
                "italic" in str(span.get("font", "")).casefold()
                or bool(int(span.get("flags", 0)) & 2)
                for span in spans
            )
            result.append(
                TextLine(
                    page_index=page.number,
                    block_index=block_index,
                    line_index=line_index,
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_width=page_width,
                    page_height=page_height,
                    font_size=font_size,
                    bold=bold,
                    italic=italic,
                )
            )
    return result


def _estimate_body_font(pages: Sequence[PageData]) -> float:
    samples: List[float] = []
    for page in pages:
        if page.is_navigation:
            continue
        for line in page.lines:
            if line.y_ratio < 0.10 or line.y_ratio > 0.92:
                continue
            if line.bold or len(line.text) < 20:
                continue
            if line.font_size > 0:
                samples.append(line.font_size)
    if samples:
        return float(statistics.median(samples))

    fallback = [
        line.font_size
        for page in pages
        for line in page.lines
        if line.font_size > 0
    ]
    return float(statistics.median(fallback)) if fallback else 10.0


# ---------------------------------------------------------------------------
# Navigation / visible TOC evidence
# ---------------------------------------------------------------------------


def _looks_like_navigation_page(lines: Sequence[TextLine]) -> bool:
    normalized = {_normalize_for_match(line.text).upper() for line in lines}
    for title in NAVIGATION_TITLES:
        if _normalize_for_match(title).upper() in normalized:
            return True

    leader_lines = 0
    toc_like_lines = 0
    for line in lines:
        if re.search(r"\.{5,}\s*\d+\s*$", line.text):
            leader_lines += 1
        if re.match(r"^\s*" + SECTION_NUMBER_PATTERN + r"\b", line.text):
            toc_like_lines += 1
    return leader_lines >= 4 and toc_like_lines >= 6


def _group_visual_rows(lines: Sequence[TextLine]) -> List[List[TextLine]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: (item.y0, item.x0, item.block_index, item.line_index))
    median_height = statistics.median([line.height for line in ordered])
    tolerance = max(1.5, median_height * 0.30)
    rows: List[List[TextLine]] = []

    for line in ordered:
        center = (line.y0 + line.y1) / 2.0
        if not rows:
            rows.append([line])
            continue
        previous_centers = [
            (item.y0 + item.y1) / 2.0 for item in rows[-1]
        ]
        row_center = statistics.mean(previous_centers)
        if abs(center - row_center) <= tolerance:
            rows[-1].append(line)
            rows[-1].sort(key=lambda item: item.x0)
        else:
            rows.append([line])
    return rows


def _extract_visible_toc_evidence(
    pages: Sequence[PageData],
) -> Tuple[Dict[str, List[str]], List[str]]:
    evidence: Dict[str, List[str]] = {}
    order: List[str] = []

    for page in pages:
        if not page.is_navigation:
            continue
        for row in _group_visual_rows(page.lines):
            combined = _clean_text(" ".join(line.text for line in row))
            match = TOC_ENTRY_RE.match(combined)
            if not match:
                continue
            number = match.group("number")
            rest = match.group("rest")
            # Remove dotted leaders and printed page number.  On wrapped TOC
            # entries this may retain only the first title line, which is still
            # useful as supporting evidence.
            title = re.split(r"\.{3,}", rest, maxsplit=1)[0].strip()
            title = re.sub(r"\s+\d+\s*$", "", title).strip()
            if not title:
                continue
            evidence.setdefault(number, [])
            if title not in evidence[number]:
                evidence[number].append(title)
            if number not in order:
                order.append(number)
    return evidence, order


def _bookmark_evidence(
    doc: fitz.Document,
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, List[Tuple[str, int]]]]:
    numeric: Dict[str, List[Tuple[str, int]]] = {}
    nonnumeric: Dict[str, List[Tuple[str, int]]] = {}
    try:
        toc = doc.get_toc(simple=True)
    except (AttributeError, RuntimeError, ValueError):
        toc = []

    for entry in toc:
        if len(entry) < 3:
            continue
        _level, raw_title, page_number = entry[:3]
        title = _clean_text(str(raw_title))
        if int(page_number) < 1:
            continue
        match = SECTION_ROW_RE.match(title)
        if match:
            number = match.group("number")
            numeric.setdefault(number, []).append(
                (match.group("title").strip(), int(page_number) - 1)
            )
            continue
        app = APPENDIX_RE.match(title)
        if app:
            key = "%s %s" % (app.group("kind").upper(), app.group("identifier").upper())
            nonnumeric.setdefault(key, []).append((title, int(page_number) - 1))
    return numeric, nonnumeric


# ---------------------------------------------------------------------------
# Running headers / margins
# ---------------------------------------------------------------------------


def _edge_fingerprint(text: str) -> str:
    normalized = _clean_text(text).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _detect_repeated_edge_lines(
    pages: Sequence[PageData],
    edge_band: float = 0.105,
) -> Set[Tuple[int, int, int]]:
    occurrences: Dict[Tuple[str, str, int], List[TextLine]] = {}
    page_count = len(pages)
    min_pages = 3 if page_count >= 5 else 2

    for page in pages:
        for line in page.lines:
            edge: Optional[str] = None
            location_ratio = line.y_ratio
            if line.y_ratio <= edge_band:
                edge = "top"
            elif line.y_ratio >= 1.0 - edge_band:
                edge = "bottom"
                location_ratio = 1.0 - line.y_ratio
            if edge is None:
                continue
            fingerprint = _edge_fingerprint(line.text)
            if not fingerprint:
                continue
            # Bucket by edge-distance so the repeated current-section header at
            # the top is distinguishable from the real heading slightly lower.
            bucket = int(round(location_ratio / 0.01))
            occurrences.setdefault((edge, fingerprint, bucket), []).append(line)

    repeated: Set[Tuple[int, int, int]] = set()
    for (_edge, _fingerprint, _bucket), lines in occurrences.items():
        distinct_pages = {line.page_index for line in lines}
        if len(distinct_pages) >= min_pages:
            repeated.update(line.key for line in lines)
    return repeated


def _looks_like_page_footer(line: TextLine) -> bool:
    if line.y_ratio < 0.92:
        return False
    centered = abs(line.center_x_ratio - 0.5) <= 0.12
    if not centered:
        return False
    stripped = line.text.strip()
    return bool(ROMAN_PAGE_RE.fullmatch(stripped) or PLAIN_PAGE_RE.fullmatch(stripped))


def _looks_like_running_header(line: TextLine, body_font: float) -> bool:
    if line.y_ratio > 0.105:
        return False
    if RUNNING_ARINC_RE.match(line.text):
        return True
    # ARINC body pages commonly repeat the current top-level section centered
    # below the specification/page header.  In the supplied preview this is
    # smaller than the real 11-point heading and is horizontally centered.
    if (
        line.font_size <= body_font - 0.75
        and line.x_ratio >= 0.30
        and RUNNING_SECTION_RE.match(line.text)
    ):
        return True
    return False


def _line_margin_filtered(
    line: TextLine,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> bool:
    return (
        line.y_ratio < top_margin_ratio
        or line.y_ratio > 1.0 - bottom_margin_ratio
    )


def _suggest_margins(
    pages: Sequence[PageData],
    repeated_keys: Set[Tuple[int, int, int]],
    inspection_band: float,
) -> Tuple[Optional[float], Optional[float]]:
    top_repeated: List[TextLine] = []
    bottom_repeated: List[TextLine] = []
    non_repeated_top: List[float] = []
    non_repeated_bottom: List[float] = []

    for page in pages:
        for line in page.lines:
            if line.y_ratio <= inspection_band:
                if line.key in repeated_keys or _looks_like_running_header(line, 11.0):
                    top_repeated.append(line)
                else:
                    non_repeated_top.append(line.y_ratio)
            if line.y_ratio >= 1.0 - inspection_band:
                if line.key in repeated_keys or _looks_like_page_footer(line):
                    bottom_repeated.append(line)
                else:
                    non_repeated_bottom.append(line.y_ratio)

    suggested_top: Optional[float] = None
    if top_repeated:
        last_header = max(line.y_ratio for line in top_repeated)
        later_body = [value for value in non_repeated_top if value > last_header]
        if later_body:
            suggested_top = round((last_header + min(later_body)) / 2.0, 3)
        else:
            suggested_top = round(min(inspection_band, last_header + 0.02), 3)

    suggested_bottom: Optional[float] = None
    if bottom_repeated:
        first_footer = min(line.y_ratio for line in bottom_repeated)
        earlier_body = [value for value in non_repeated_bottom if value < first_footer]
        if earlier_body:
            cutoff = (max(earlier_body) + first_footer) / 2.0
        else:
            cutoff = max(1.0 - inspection_band, first_footer - 0.02)
        suggested_bottom = round(1.0 - cutoff, 3)

    return suggested_top, suggested_bottom


def _format_margin_report(
    pages: Sequence[PageData],
    repeated_keys: Set[Tuple[int, int, int]],
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    inspection_band: float,
    body_font: float,
) -> str:
    suggested_top, suggested_bottom = _suggest_margins(
        pages, repeated_keys, inspection_band
    )
    output = [
        "Physical PDF pages: %d" % len(pages),
        "Current margins: top=%.3f, bottom=%.3f"
        % (top_margin_ratio, bottom_margin_ratio),
        "Inspection band: %.3f" % inspection_band,
        "",
        "Status includes geometric margin filtering plus repeated/fallback running-header filtering.",
        "",
    ]

    for page in pages:
        for line in page.lines:
            if not (
                line.y_ratio <= inspection_band
                or line.y_ratio >= 1.0 - inspection_band
            ):
                continue
            if _line_margin_filtered(line, top_margin_ratio, bottom_margin_ratio):
                status = "FILTERED margin"
            elif line.key in repeated_keys:
                status = "FILTERED repeated"
            elif _looks_like_running_header(line, body_font):
                status = "FILTERED header"
            elif _looks_like_page_footer(line):
                status = "FILTERED footer"
            else:
                status = "KEPT"
            edge = "TOP" if line.y_ratio <= inspection_band else "BOTTOM"
            distance = line.y_ratio if edge == "TOP" else 1.0 - line.y_ratio
            output.append(
                "%s pdf=%4d distance=%.4f %-18s %s"
                % (edge, page.page_index + 1, distance, status, line.text)
            )

    output.extend(["", "Automatic suggestion from repeated edge text:"])
    if suggested_top is not None:
        output.append("  --top-margin %.3f" % suggested_top)
    else:
        output.append("  top: no reliable suggestion")
    if suggested_bottom is not None:
        output.append("  --bottom-margin %.3f" % suggested_bottom)
    else:
        output.append("  bottom: no reliable suggestion")
    output.extend(
        [
            "",
            "Verify that real body headings remain KEPT before adopting suggested margins.",
        ]
    )
    return "\n".join(output)


# ---------------------------------------------------------------------------
# Tables and raster images / figure captions
# ---------------------------------------------------------------------------


def _render_table_rows(rows: Sequence[Sequence[object]]) -> str:
    rendered: List[str] = []
    for row in rows:
        cells: List[str] = []
        for cell in row:
            if cell is None:
                value = ""
            else:
                value = _clean_text(str(cell).replace("\n", " "))
            value = value.replace("|", "\\|")
            cells.append(value)
        if any(cell for cell in cells):
            rendered.append(" | ".join(cells).rstrip())
    return "\n".join(rendered).strip()


def _fallback_table_text(
    lines: Sequence[TextLine],
    bbox: Tuple[float, float, float, float],
) -> str:
    inside = [line for line in lines if _line_overlaps_region(line, bbox, 0.25)]
    rows = _group_visual_rows(inside)
    rendered: List[str] = []
    for row in rows:
        row_text = " | ".join(line.text for line in sorted(row, key=lambda item: item.x0))
        if row_text.strip():
            rendered.append(row_text.strip())
    return "\n".join(rendered).strip()


def _find_nearby_caption_keys(
    lines: Sequence[TextLine],
    bbox: Tuple[float, float, float, float],
    caption_re: re.Pattern,
) -> Set[Tuple[int, int, int]]:
    if not lines:
        return set()
    typical_height = statistics.median([line.height for line in lines])
    candidates: List[Tuple[float, TextLine]] = []
    for line in lines:
        if not caption_re.match(line.text):
            continue
        line_box = (line.x0, line.y0, line.x1, line.y1)
        if line.y1 <= bbox[1]:
            distance = bbox[1] - line.y1
        elif line.y0 >= bbox[3]:
            distance = line.y0 - bbox[3]
        else:
            distance = 0.0
        if distance <= typical_height * 3.0 and _horizontal_overlap_ratio(line_box, bbox) >= 0.15:
            candidates.append((distance, line))
    if not candidates:
        return set()
    _distance, first = min(candidates, key=lambda pair: (pair[0], pair[1].y0))
    keys = {first.key}
    # Wrapped caption lines are usually in the same PDF text block.
    for line in lines:
        if line.block_index == first.block_index and line.line_index > first.line_index:
            if line.y0 - first.y1 <= typical_height * 3.0:
                keys.add(line.key)
    return keys


def _detect_tables(
    page: fitz.Page,
    lines: Sequence[TextLine],
    warnings: WarningCollector,
) -> List[TableRegion]:
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            finder = page.find_tables()
        raw_tables = list(getattr(finder, "tables", []) or [])
    except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
        warnings.add(
            "PDF %d: table detection unavailable/failed (%s); page text is preserved as normal text where possible."
            % (page.number + 1, exc)
        )
        return []

    regions: List[TableRegion] = []
    for table_index, table in enumerate(raw_tables, start=1):
        try:
            bbox = tuple(float(value) for value in table.bbox)
        except (AttributeError, TypeError, ValueError):
            warnings.add(
                "PDF %d table %d: invalid table bbox; content was not spatially removed from body text."
                % (page.number + 1, table_index)
            )
            continue

        text = ""
        fallback_used = False
        try:
            rows = table.extract()
            text = _render_table_rows(rows or [])
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            warnings.add(
                "PDF %d table %d: structured extraction failed (%s); using bbox text fallback."
                % (page.number + 1, table_index, exc)
            )

        if not text:
            text = _fallback_table_text(lines, bbox)
            fallback_used = True
            if text:
                warnings.add(
                    "PDF %d table %d: structured cells were empty; preserved table-region text with a layout fallback."
                    % (page.number + 1, table_index)
                )
            else:
                warnings.add(
                    "PDF %d table %d: table region was detected but no extractable text was available."
                    % (page.number + 1, table_index)
                )

        caption_keys = _find_nearby_caption_keys(lines, bbox, TABLE_CAPTION_RE)
        if caption_keys:
            caption_lines = [line for line in lines if line.key in caption_keys]
            caption_text = _render_lines_inline(caption_lines)
            if caption_text:
                text = caption_text + ("\n" + text if text else "")

        regions.append(
            TableRegion(
                page_index=page.number,
                bbox=bbox,
                text=text,
                caption_keys=caption_keys,
                fallback_used=fallback_used,
            )
        )

        page_height = float(page.rect.height)
        if bbox[3] >= page_height * 0.92:
            warnings.add(
                "PDF %d table %d reaches the lower page edge; it may be one part of a cross-page table and is kept page-local."
                % (page.number + 1, table_index)
            )

    # Remove near-identical nested duplicates if the table finder reports them.
    deduped: List[TableRegion] = []
    for region in regions:
        duplicate = False
        for existing in deduped:
            inter = _bbox_intersection_area(region.bbox, existing.bbox)
            smaller = min(_bbox_area(region.bbox), _bbox_area(existing.bbox))
            if smaller > 0 and inter / smaller >= 0.95:
                duplicate = True
                if len(region.text) > len(existing.text):
                    existing.text = region.text
                    existing.caption_keys.update(region.caption_keys)
                break
        if not duplicate:
            deduped.append(region)
    return sorted(deduped, key=lambda item: (item.bbox[1], item.bbox[0]))


def _detect_raster_images(page: fitz.Page) -> List[ImageRegion]:
    page_dict = page.get_text("dict", sort=True)
    page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
    images: List[ImageRegion] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue
        try:
            bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
        except (TypeError, ValueError):
            continue
        if _bbox_area(bbox) / page_area < 0.005:
            # Ignore very small raster decorations/logos.
            continue
        if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
            continue
        images.append(ImageRegion(page_index=page.number, bbox=bbox))
    return images


def _caption_block_lines(
    first: TextLine,
    lines: Sequence[TextLine],
) -> List[TextLine]:
    same_block = [
        line
        for line in lines
        if line.block_index == first.block_index
        and line.line_index >= first.line_index
    ]
    if not same_block:
        return [first]
    same_block.sort(key=lambda line: line.line_index)
    result = [same_block[0]]
    typical = max(1.0, first.height)
    for line in same_block[1:]:
        if line.y0 - result[-1].y1 > typical * 0.8:
            break
        if SECTION_ROW_RE.match(line.text) or SECTION_NUMBER_ONLY_RE.match(line.text):
            break
        result.append(line)
    return result


def _detect_figure_captions(
    page_data: PageData,
    usable_lines: Sequence[TextLine],
    warnings: WarningCollector,
) -> List[FigureCaption]:
    captions: List[FigureCaption] = []
    consumed: Set[Tuple[int, int, int]] = set()

    for line in usable_lines:
        if line.key in consumed or not FIGURE_CAPTION_RE.match(line.text):
            continue
        block_lines = _caption_block_lines(line, usable_lines)
        keys = {item.key for item in block_lines}
        consumed.update(keys)
        text = _render_lines_inline(block_lines)
        bbox = _bbox_union([(item.x0, item.y0, item.x1, item.y1) for item in block_lines])

        associated = False
        for image in page_data.images:
            if bbox[3] <= image.bbox[1]:
                distance = image.bbox[1] - bbox[3]
            elif bbox[1] >= image.bbox[3]:
                distance = bbox[1] - image.bbox[3]
            else:
                distance = 0.0
            if (
                distance <= page_data.height * 0.10
                and _horizontal_overlap_ratio(bbox, image.bbox) >= 0.10
            ):
                associated = True
                break

        captions.append(
            FigureCaption(
                page_index=page_data.page_index,
                bbox=bbox,
                text=text,
                line_keys=keys,
                associated_image=associated,
            )
        )

    for image_index, image in enumerate(page_data.images, start=1):
        nearby = False
        for caption in captions:
            if caption.associated_image:
                if caption.bbox[3] <= image.bbox[1]:
                    distance = image.bbox[1] - caption.bbox[3]
                elif caption.bbox[1] >= image.bbox[3]:
                    distance = caption.bbox[1] - image.bbox[3]
                else:
                    distance = 0.0
                if distance <= page_data.height * 0.10:
                    nearby = True
                    break
        if not nearby:
            warnings.add(
                "PDF %d raster image %d has no reliable extractable figure caption; no image text was invented."
                % (page_data.page_index + 1, image_index)
            )

    return captions


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------


def _line_is_excluded_from_heading(
    line: TextLine,
    *,
    repeated_keys: Set[Tuple[int, int, int]],
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    body_font: float,
    table_regions: Sequence[TableRegion],
    image_regions: Sequence[ImageRegion],
    figure_caption_keys: Set[Tuple[int, int, int]],
) -> bool:
    if _line_margin_filtered(line, top_margin_ratio, bottom_margin_ratio):
        return True
    if line.key in repeated_keys:
        return True
    if _looks_like_running_header(line, body_font) or _looks_like_page_footer(line):
        return True
    if line.key in figure_caption_keys:
        return True
    for table in table_regions:
        if _line_overlaps_region(line, table.bbox):
            return True
    for image in image_regions:
        if _line_overlaps_region(line, image.bbox, 0.60):
            return True
    return False


def _supporting_section_titles(
    number: str,
    toc_evidence: Dict[str, List[str]],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
) -> List[str]:
    titles: List[str] = []
    titles.extend(toc_evidence.get(number, []))
    titles.extend(item[0] for item in bookmark_numeric.get(number, []))
    return titles


def _section_evidence(
    number: str,
    title: str,
    toc_evidence: Dict[str, List[str]],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
    *,
    similarity_threshold: float = 0.35,
) -> Tuple[Tuple[str, ...], float]:
    """Return strong external evidence labels and the best title similarity.

    A matching section number alone is deliberately not treated as strong
    corroboration.  This prevents figure labels such as ``2.4 Something`` from
    becoming headings merely because section 2.4 exists elsewhere in the
    document.  The title must also be reasonably similar to the TOC/bookmark
    title.
    """
    evidence_names: List[str] = []
    best_similarity = 0.0

    toc_titles = toc_evidence.get(number, [])
    if toc_titles:
        toc_similarity = max(
            (_title_similarity(title, item) for item in toc_titles),
            default=0.0,
        )
        best_similarity = max(best_similarity, toc_similarity)
        if toc_similarity >= similarity_threshold:
            evidence_names.append("visible_toc")

    bookmark_titles = bookmark_numeric.get(number, [])
    if bookmark_titles:
        bookmark_similarity = max(
            (_title_similarity(title, item[0]) for item in bookmark_titles),
            default=0.0,
        )
        best_similarity = max(best_similarity, bookmark_similarity)
        if bookmark_similarity >= similarity_threshold:
            evidence_names.append("bookmark")

    return tuple(evidence_names), best_similarity


def _heading_candidates_on_page(
    page_data: PageData,
    candidate_lines: Sequence[TextLine],
    body_font: float,
    toc_evidence: Dict[str, List[str]],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
) -> List[HeadingCandidate]:
    rows = _group_visual_rows(candidate_lines)
    candidates: List[HeadingCandidate] = []
    used_rows: Set[int] = set()

    for row_index, row in enumerate(rows):
        if row_index in used_rows:
            continue
        row = sorted(row, key=lambda item: item.x0)
        if not row:
            continue
        combined = _clean_text(" ".join(line.text for line in row))
        number: Optional[str] = None
        title = ""
        selected_lines: List[TextLine] = []

        # Preferred case: a left numeric fragment and a title fragment share a
        # visual row (for example, ``1.1`` + ``Purpose``).
        first_number = SECTION_NUMBER_ONLY_RE.match(row[0].text)
        if first_number and len(row) >= 2:
            number = first_number.group("number")
            title = _clean_text(" ".join(line.text for line in row[1:]))
            selected_lines = list(row)
        else:
            direct = SECTION_ROW_RE.match(combined)
            if direct:
                number = direct.group("number")
                title = _clean_text(direct.group("title"))
                selected_lines = list(row)
            elif first_number and len(row) == 1:
                # Rare split-row heading: number on one line and title just
                # below.  Require strong heading-like style.
                if row_index + 1 < len(rows):
                    next_row = sorted(rows[row_index + 1], key=lambda item: item.x0)
                    gap = min(item.y0 for item in next_row) - max(item.y1 for item in row)
                    next_text = _clean_text(" ".join(item.text for item in next_row))
                    next_x_ratio = (
                        min(item.x0 for item in next_row) / page_data.width
                        if page_data.width
                        else 0.0
                    )
                    if (
                        next_text
                        and gap <= max(4.0, row[0].height * 0.65)
                        and any(item.bold for item in next_row)
                        and next_x_ratio <= 0.20
                        and not SECTION_ROW_RE.match(next_text)
                        and not SECTION_NUMBER_ONLY_RE.match(next_text)
                        and not APPENDIX_RE.match(next_text)
                    ):
                        number = first_number.group("number")
                        title = next_text
                        selected_lines = list(row) + list(next_row)
                        used_rows.add(row_index + 1)

        if not number or not title:
            continue

        # Source typo special case: some ARINC PDFs contain a heading such as
        # ``4.7.2 2`` with the title on the next visual row, where the intended
        # number is clearly ``4.7.2.2``.  Only consume that next row when the
        # repaired number and the next-row title are corroborated by the
        # visible TOC and/or PDF bookmarks.
        if re.fullmatch(r"\d+", title) and row_index + 1 < len(rows):
            child = title
            repaired_number = number + "." + child
            next_row = sorted(rows[row_index + 1], key=lambda item: item.x0)
            next_text = _clean_text(" ".join(item.text for item in next_row))
            support_titles = _supporting_section_titles(
                repaired_number, toc_evidence, bookmark_numeric
            )
            next_similarity = max(
                (_title_similarity(next_text, item) for item in support_titles),
                default=0.0,
            )
            next_x_ratio = (
                min(item.x0 for item in next_row) / page_data.width
                if next_row and page_data.width
                else 1.0
            )
            gap = (
                min(item.y0 for item in next_row) - max(item.y1 for item in selected_lines)
                if next_row and selected_lines
                else math.inf
            )
            next_style_ok = bool(next_row) and (
                any(item.bold for item in next_row)
                or max(item.font_size for item in next_row) >= body_font - 0.25
            )
            if (
                support_titles
                and next_text
                and next_similarity >= 0.35
                and next_x_ratio <= 0.20
                and gap <= max(5.0, statistics.median([item.height for item in selected_lines]) * 0.90)
                and next_style_ok
                and not SECTION_ROW_RE.match(next_text)
                and not SECTION_NUMBER_ONLY_RE.match(next_text)
                and not APPENDIX_RE.match(next_text)
            ):
                selected_lines.extend(next_row)
                title = child + " " + next_text
                used_rows.add(row_index + 1)

        source_typo: Optional[str] = None
        number, title, source_typo = _repair_missing_dot_section_number(
            number, title, toc_evidence, bookmark_numeric
        )

        # A TOC-like leader line should never become a body heading.
        if re.search(r"\.{3,}\s*\d+\s*$", title):
            continue

        evidence_names, evidence_similarity = _section_evidence(
            number, title, toc_evidence, bookmark_numeric
        )
        has_strong_evidence = bool(evidence_names)

        # If strongly corroborated, allow one wrapped bold title row just below
        # the first heading row.  Number-only evidence is not sufficient.
        if (
            has_strong_evidence
            and selected_lines
            and row_index + 1 < len(rows)
            and (row_index + 1) not in used_rows
        ):
            next_row = sorted(rows[row_index + 1], key=lambda item: item.x0)
            next_text = _clean_text(" ".join(item.text for item in next_row))
            gap = min(item.y0 for item in next_row) - max(item.y1 for item in selected_lines)
            title_x0 = min(item.x0 for item in selected_lines[1:] or selected_lines)
            if (
                next_text
                and gap <= max(
                    3.0,
                    statistics.median([item.height for item in selected_lines]) * 0.55,
                )
                and all(item.bold for item in next_row)
                and min(item.x0 for item in next_row) >= title_x0 - page_data.width * 0.03
                and min(item.x0 for item in next_row) <= page_data.width * 0.20
                and not SECTION_ROW_RE.match(next_text)
                and not SECTION_NUMBER_ONLY_RE.match(next_text)
                and not APPENDIX_RE.match(next_text)
            ):
                proposed_title = _clean_text(title + " " + next_text)
                proposed_evidence, proposed_similarity = _section_evidence(
                    number, proposed_title, toc_evidence, bookmark_numeric
                )
                if proposed_evidence:
                    selected_lines.extend(next_row)
                    title = proposed_title
                    evidence_names = proposed_evidence
                    evidence_similarity = proposed_similarity
                    used_rows.add(row_index + 1)

        boxes = [(line.x0, line.y0, line.x1, line.y1) for line in selected_lines]
        bbox = _bbox_union(boxes)
        font_size = max(line.font_size for line in selected_lines)
        bold = any(line.bold for line in selected_lines)
        x_ratio = bbox[0] / page_data.width if page_data.width else 0.0
        y_ratio = (
            ((bbox[1] + bbox[3]) / 2.0) / page_data.height
            if page_data.height
            else 0.0
        )

        # Numeric ARINC body headings are anchored in the leftmost fifth of the
        # page.  Treat this as a hard geometric requirement, even if the same
        # section number exists in a TOC/bookmark.  This blocks bold labels
        # embedded inside figures and diagrams farther to the right.
        if x_ratio > 0.20:
            continue

        # Bare integer labels (``1``, ``2``, ``3``) are common inside figures.
        # ARINC main-body top-level headings are normally ``1.0``, ``2.0``, ...
        # so a bare integer is accepted only when its title is strongly
        # corroborated by the TOC/bookmarks.
        if "." not in number and not has_strong_evidence:
            continue

        score = 0.0
        if bold:
            score += 2.0
        if font_size >= body_font - 0.25:
            score += 1.0
        if x_ratio <= 0.20:
            score += 1.0
        if "visible_toc" in evidence_names:
            score += 3.0
        if "bookmark" in evidence_names:
            score += 3.0
        if evidence_similarity >= 0.60:
            score += 0.5
        if title.isupper() and len(title) <= 120:
            score += 0.25
        if y_ratio < 0.095 and font_size < body_font - 0.5:
            score -= 2.0
        if len(title) > 220:
            score -= 1.5

        candidates.append(
            HeadingCandidate(
                page_index=page_data.page_index,
                section_number=number,
                section_title=title,
                line_keys={line.key for line in selected_lines},
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                font_size=font_size,
                bold=bold,
                x_ratio=x_ratio,
                score=score,
                evidence=evidence_names,
                evidence_similarity=evidence_similarity,
                kind="numeric",
                source_typo=source_typo,
            )
        )

    return candidates

def _appendix_candidates_on_page(
    page_data: PageData,
    candidate_lines: Sequence[TextLine],
    body_font: float,
) -> List[HeadingCandidate]:
    rows = _group_visual_rows(candidate_lines)
    result: List[HeadingCandidate] = []
    used_rows: Set[int] = set()

    for row_index, row in enumerate(rows):
        if row_index in used_rows:
            continue
        combined = _clean_text(" ".join(line.text for line in sorted(row, key=lambda item: item.x0)))
        match = APPENDIX_RE.match(combined)
        if not match:
            continue
        selected = list(row)
        title_tail = _clean_text(match.group("title"))

        if not title_tail and row_index + 1 < len(rows):
            next_row = rows[row_index + 1]
            next_text = _clean_text(" ".join(line.text for line in sorted(next_row, key=lambda item: item.x0)))
            gap = min(line.y0 for line in next_row) - max(line.y1 for line in row)
            if (
                next_text
                and gap <= max(4.0, statistics.median([line.height for line in row]) * 0.75)
                and any(line.bold for line in next_row)
            ):
                selected.extend(next_row)
                combined = _clean_text(combined + " " + next_text)
                used_rows.add(row_index + 1)

        bbox = _bbox_union([(line.x0, line.y0, line.x1, line.y1) for line in selected])
        bold = any(line.bold for line in selected)
        font_size = max(line.font_size for line in selected)
        x_ratio = bbox[0] / page_data.width if page_data.width else 0.0
        strong = (
            x_ratio <= 0.22
            and font_size >= body_font - 0.25
            and (bold or combined.isupper())
        )
        if not strong:
            continue
        result.append(
            HeadingCandidate(
                page_index=page_data.page_index,
                section_number=None,
                section_title=combined,
                line_keys={line.key for line in selected},
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                font_size=font_size,
                bold=bold,
                x_ratio=x_ratio,
                score=4.0,
                evidence=(),
                kind=match.group("kind").lower(),
            )
        )
    return result


def _detect_headings(
    pages: Sequence[PageData],
    *,
    repeated_keys: Set[Tuple[int, int, int]],
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    body_font: float,
    toc_evidence: Dict[str, List[str]],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
    warnings: WarningCollector,
) -> Tuple[
    List[HeadingCandidate],
    Set[Tuple[int, int, int]],
    Optional[int],
    Optional[Tuple[int, float]],
]:
    navigation_pages = [page.page_index for page in pages if page.is_navigation]
    earliest_body_candidate = max(navigation_pages) + 1 if navigation_pages else 0

    # First identify reliable ATTACHMENT / APPENDIX boundaries.  Numeric
    # numbering inside those regions belongs to a different namespace and is
    # deliberately not promoted to document-level ARINC sections in v0.
    appendix_candidates: List[HeadingCandidate] = []
    page_candidate_lines: Dict[int, List[TextLine]] = {}
    for page in pages:
        if page.page_index < earliest_body_candidate or page.is_navigation:
            continue
        figure_keys: Set[Tuple[int, int, int]] = set()
        for caption in page.figure_captions:
            figure_keys.update(caption.line_keys)
        candidate_lines = [
            line
            for line in page.lines
            if not _line_is_excluded_from_heading(
                line,
                repeated_keys=repeated_keys,
                top_margin_ratio=top_margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
                body_font=body_font,
                table_regions=page.tables,
                image_regions=page.images,
                figure_caption_keys=figure_keys,
            )
        ]
        page_candidate_lines[page.page_index] = candidate_lines
        appendix_candidates.extend(
            _appendix_candidates_on_page(page, candidate_lines, body_font)
        )

    appendix_candidates.sort(key=lambda item: (item.page_index, item.y0, item.x0))
    first_appendix_position: Optional[Tuple[int, float]] = None
    if appendix_candidates:
        first_appendix_position = (
            appendix_candidates[0].page_index,
            appendix_candidates[0].y0,
        )

    # Only bookmarks pointing to the main numeric body may corroborate numeric
    # headings.  Repeated numbers inside attachments/appendices must not leak
    # back into the main-body namespace.  A numeric bookmark on the exact page
    # where the first ATTACHMENT/APPENDIX begins is kept only when the visible
    # main TOC also knows that section number; this avoids using appendix-local
    # bookmark numbers as body evidence.
    main_bookmark_numeric: Dict[str, List[Tuple[str, int]]] = {}
    for number, entries in bookmark_numeric.items():
        kept: List[Tuple[str, int]] = []
        for title, page_index in entries:
            if first_appendix_position is None:
                kept.append((title, page_index))
            elif page_index < first_appendix_position[0]:
                kept.append((title, page_index))
            elif (
                page_index == first_appendix_position[0]
                and number in toc_evidence
            ):
                kept.append((title, page_index))
        if kept:
            main_bookmark_numeric[number] = kept

    raw_candidates: List[HeadingCandidate] = []
    for page in pages:
        if page.page_index < earliest_body_candidate or page.is_navigation:
            continue
        candidate_lines = page_candidate_lines.get(page.page_index, [])
        numeric_candidates = _heading_candidates_on_page(
            page,
            candidate_lines,
            body_font,
            toc_evidence,
            main_bookmark_numeric,
        )
        for candidate in numeric_candidates:
            if first_appendix_position is not None:
                if candidate.page_index > first_appendix_position[0]:
                    continue
                if (
                    candidate.page_index == first_appendix_position[0]
                    and candidate.y0 >= first_appendix_position[1]
                ):
                    continue
            raw_candidates.append(candidate)

    # Learn the left-column anchor from strongly corroborated main-body
    # headings.  Layout-only candidates must sit close to that anchor.  This is
    # stricter than the absolute leftmost-fifth rule and filters bold numeric
    # labels inside figures even when they are still technically left of 20%.
    corroborated_x = [
        candidate.x_ratio
        for candidate in raw_candidates
        if candidate.evidence
        and candidate.score >= 4.0
        and candidate.x_ratio <= 0.20
    ]
    layout_heading_x_cutoff = 0.20
    if len(corroborated_x) >= 3:
        anchor = statistics.median(corroborated_x)
        deviations = [abs(value - anchor) for value in corroborated_x]
        mad = statistics.median(deviations) if deviations else 0.0
        adaptive_tolerance = max(0.025, min(0.050, 0.020 + 3.0 * mad))
        layout_heading_x_cutoff = min(
            0.20,
            max(0.12, anchor + adaptive_tolerance),
        )

    accepted_numeric = [
        candidate
        for candidate in raw_candidates
        if candidate.score >= 4.0
        and (candidate.evidence or candidate.x_ratio <= layout_heading_x_cutoff)
    ]
    accepted_numeric.sort(key=lambda item: (item.page_index, item.y0, item.x0))

    duplicate_running_keys: Set[Tuple[int, int, int]] = set()
    deduped: List[HeadingCandidate] = []
    by_number: Dict[str, HeadingCandidate] = {}
    previous_numeric: Optional[HeadingCandidate] = None

    for candidate in accepted_numeric:
        assert candidate.section_number is not None
        existing = by_number.get(candidate.section_number)
        if existing is not None:
            similarity = _title_similarity(existing.section_title, candidate.section_title)
            if similarity >= 0.70:
                duplicate_running_keys.update(candidate.line_keys)
                warnings.add(
                    "PDF %d: duplicate heading %s (%s) was ignored as probable repeated heading text."
                    % (
                        candidate.page_index + 1,
                        candidate.section_number,
                        candidate.section_title,
                    )
                )
                continue
            warnings.add(
                "PDF %d: duplicate section number %s has a different title; it was left as ordinary text rather than creating a second section."
                % (candidate.page_index + 1, candidate.section_number)
            )
            continue

        if previous_numeric is not None and previous_numeric.section_number is not None:
            try:
                if _section_tuple(candidate.section_number) < _section_tuple(previous_numeric.section_number):
                    warnings.add(
                        "PDF %d: heading %s appears out of numeric order after %s; verify --inspect-sections output."
                        % (
                            candidate.page_index + 1,
                            candidate.section_number,
                            previous_numeric.section_number,
                        )
                    )
            except ValueError:
                pass

        if candidate.source_typo:
            warnings.add(
                'PDF %d: probable source numbering typo "%s" was normalized to section %s using TOC/bookmark evidence.'
                % (
                    candidate.page_index + 1,
                    candidate.source_typo,
                    candidate.section_number,
                )
            )

        if not candidate.evidence and candidate.score < 4.75:
            warnings.add(
                "PDF %d: section %s was accepted mainly from layout/style rather than TOC/bookmark corroboration."
                % (candidate.page_index + 1, candidate.section_number)
            )

        deduped.append(candidate)
        by_number[candidate.section_number] = candidate
        previous_numeric = candidate

    first_numeric_page = min((item.page_index for item in deduped), default=None)

    # Keep one enclosing heading per attachment/appendix identifier.  Repeated
    # page-top labels for the same attachment/appendix are treated as running
    # text and removed from body prose.
    deduped_appendices: List[HeadingCandidate] = []
    seen_appendix_keys: Dict[str, HeadingCandidate] = {}
    for candidate in appendix_candidates:
        if first_numeric_page is not None and candidate.page_index < first_numeric_page:
            continue
        match = APPENDIX_RE.match(candidate.section_title)
        if not match:
            continue
        namespace_key = "%s %s" % (
            match.group("kind").upper(),
            match.group("identifier").upper(),
        )
        existing = seen_appendix_keys.get(namespace_key)
        if existing is not None:
            duplicate_running_keys.update(candidate.line_keys)
            continue
        seen_appendix_keys[namespace_key] = candidate
        deduped_appendices.append(candidate)

    all_headings = deduped + deduped_appendices
    all_headings.sort(key=lambda item: (item.page_index, item.y0, item.x0))
    return (
        all_headings,
        duplicate_running_keys,
        first_numeric_page,
        first_appendix_position,
    )

def _warn_bookmark_disagreement(
    headings: Sequence[HeadingCandidate],
    bookmark_numeric: Dict[str, List[Tuple[str, int]]],
    warnings: WarningCollector,
    first_appendix_position: Optional[Tuple[int, float]],
) -> None:
    if not bookmark_numeric:
        return

    body_by_number = {
        heading.section_number: heading
        for heading in headings
        if heading.section_number is not None
    }
    for number, all_entries in bookmark_numeric.items():
        heading = body_by_number.get(number)
        if first_appendix_position is None:
            entries = list(all_entries)
        else:
            entries = []
            for title, page_index in all_entries:
                if page_index < first_appendix_position[0]:
                    entries.append((title, page_index))
                    continue
                if (
                    page_index == first_appendix_position[0]
                    and heading is not None
                    and heading.page_index == first_appendix_position[0]
                    and heading.y0 < first_appendix_position[1]
                ):
                    entries.append((title, page_index))
        if not entries:
            # Numeric bookmarks inside ATTACHMENT / APPENDIX namespaces are
            # intentionally outside the main-body disagreement check.
            continue

        if heading is None:
            warnings.add(
                "Bookmark/body-heading disagreement: bookmark section %s was not detected as a main-body heading."
                % number
            )
            continue
        best_title_similarity = max(
            (_title_similarity(heading.section_title, title) for title, _page in entries),
            default=0.0,
        )
        if best_title_similarity < 0.35:
            warnings.add(
                "Bookmark/body-heading disagreement: main-body section %s title differs substantially from bookmark title."
                % number
            )
        nearest_page = min(abs(heading.page_index - page_index) for _title, page_index in entries)
        if nearest_page > 2:
            warnings.add(
                "Bookmark/body-heading disagreement: main-body section %s body heading is %d physical pages away from its nearest bookmark target."
                % (number, nearest_page)
            )


# ---------------------------------------------------------------------------
# Body text reconstruction
# ---------------------------------------------------------------------------


def _render_lines_inline(lines: Sequence[TextLine]) -> str:
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda line: (line.y0, line.x0, line.block_index, line.line_index))
    result = ordered[0].text
    previous = ordered[0]
    for line in ordered[1:]:
        if previous.text.endswith("-") and line.text[:1].islower():
            result += line.text
        else:
            result += " " + line.text
        previous = line
    return _clean_text(result)


def _merge_same_row_fragments(lines: Sequence[TextLine]) -> List[TextLine]:
    """Merge separate PDF text objects that occupy one visual baseline."""
    rows = _group_visual_rows(lines)
    merged: List[TextLine] = []
    for row in rows:
        ordered = sorted(row, key=lambda line: line.x0)
        if len(ordered) == 1:
            merged.append(ordered[0])
            continue
        bbox = _bbox_union([(line.x0, line.y0, line.x1, line.y1) for line in ordered])
        merged.append(
            TextLine(
                page_index=ordered[0].page_index,
                block_index=min(line.block_index for line in ordered),
                line_index=min(line.line_index for line in ordered),
                text=_clean_text(" ".join(line.text for line in ordered)),
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
                page_width=ordered[0].page_width,
                page_height=ordered[0].page_height,
                font_size=max(line.font_size for line in ordered),
                bold=any(line.bold for line in ordered),
                italic=any(line.italic for line in ordered),
            )
        )
    return merged


def _lines_to_paragraphs(lines: Sequence[TextLine]) -> List[str]:
    if not lines:
        return []
    ordered = _merge_same_row_fragments(lines)
    ordered.sort(key=lambda line: (line.y0, line.x0, line.block_index, line.line_index))
    typical_height = statistics.median([line.height for line in ordered])
    paragraphs: List[str] = []
    current = ordered[0].text
    previous = ordered[0]

    for line in ordered[1:]:
        vertical_gap = line.y0 - previous.y1
        indent_delta = (line.x0 - previous.x0) / max(1.0, line.page_width)
        starts_list = bool(LIST_MARKER_RE.match(line.text))
        previous_starts_list = bool(LIST_MARKER_RE.match(previous.text))
        commentary_boundary = (
            previous.text.strip().upper() == "COMMENTARY"
            or line.text.strip().upper() == "COMMENTARY"
        )
        strong_gap = vertical_gap > typical_height * 0.38
        strong_indent_change = abs(indent_delta) > 0.045
        hanging_list_continuation = previous_starts_list and indent_delta > 0.0
        block_break = line.block_index != previous.block_index

        new_paragraph = (
            starts_list
            or commentary_boundary
            or block_break
            or strong_gap
            or (strong_indent_change and not hanging_list_continuation)
        )

        if new_paragraph:
            if current.strip():
                paragraphs.append(current.strip())
            current = line.text
        elif current.endswith("-") and line.text[:1].islower():
            # Preserve the visible hyphen while removing only the PDF line break:
            # "off-the-" + "shelf" -> "off-the-shelf".
            current += line.text
        else:
            current += " " + line.text
        previous = line

    if current.strip():
        paragraphs.append(current.strip())
    return paragraphs


def _front_matter_title(lines: Sequence[TextLine]) -> Optional[str]:
    for line in lines:
        normalized = _normalize_for_match(line.text).upper()
        for title in FRONT_TITLE_WORDS:
            if normalized == _normalize_for_match(title).upper():
                return line.text.strip()
    return None


def _record_suspicious_margin_drops(
    page: PageData,
    repeated_keys: Set[Tuple[int, int, int]],
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    body_font: float,
    warnings: WarningCollector,
) -> None:
    suspicious: List[str] = []
    for line in page.lines:
        if not _line_margin_filtered(line, top_margin_ratio, bottom_margin_ratio):
            continue
        if line.key in repeated_keys:
            continue
        if _looks_like_running_header(line, body_font) or _looks_like_page_footer(line):
            continue
        if page.is_navigation:
            continue
        if len(line.text) >= 24:
            suspicious.append(line.text)
    if suspicious:
        preview = " / ".join(suspicious[:2])
        warnings.add(
            "PDF %d: non-repeated text was removed by the configured page margin; inspect margins if this is real content: %s"
            % (page.page_index + 1, preview)
        )


def _build_content_units(
    pages: Sequence[PageData],
    headings: Sequence[HeadingCandidate],
    *,
    repeated_keys: Set[Tuple[int, int, int]],
    duplicate_running_keys: Set[Tuple[int, int, int]],
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    body_font: float,
    first_numeric_page: Optional[int],
    warnings: WarningCollector,
) -> List[ContentUnit]:
    headings_by_page: Dict[int, List[HeadingCandidate]] = {}
    for heading in headings:
        headings_by_page.setdefault(heading.page_index, []).append(heading)
    for values in headings_by_page.values():
        values.sort(key=lambda item: (item.y0, item.x0))

    units: List[ContentUnit] = []
    current_section_number: Optional[str] = None
    current_section_title: Optional[str] = None
    text_run_id = 0

    all_running_keys = repeated_keys | duplicate_running_keys

    for page in pages:
        if page.is_navigation:
            text_run_id += 1
            continue

        _record_suspicious_margin_drops(
            page,
            all_running_keys,
            top_margin_ratio,
            bottom_margin_ratio,
            body_font,
            warnings,
        )

        accepted_headings = headings_by_page.get(page.page_index, [])
        heading_keys: Set[Tuple[int, int, int]] = set()
        for heading in accepted_headings:
            heading_keys.update(heading.line_keys)

        table_keys: Set[Tuple[int, int, int]] = set()
        table_caption_keys: Set[Tuple[int, int, int]] = set()
        for table in page.tables:
            table_caption_keys.update(table.caption_keys)
            for line in page.lines:
                if _line_overlaps_region(line, table.bbox):
                    table_keys.add(line.key)

        figure_keys: Set[Tuple[int, int, int]] = set()
        for caption in page.figure_captions:
            figure_keys.update(caption.line_keys)

        image_text_keys: Set[Tuple[int, int, int]] = set()
        for image in page.images:
            for line in page.lines:
                if _line_overlaps_region(line, image.bbox, 0.60):
                    image_text_keys.add(line.key)
        if image_text_keys:
            warnings.add(
                "PDF %d: %d extractable text line(s) overlap raster-image regions and were excluded from normal body text."
                % (page.page_index + 1, len(image_text_keys))
            )

        usable_lines: List[TextLine] = []
        for line in page.lines:
            if _line_margin_filtered(line, top_margin_ratio, bottom_margin_ratio):
                continue
            if line.key in all_running_keys:
                continue
            if _looks_like_running_header(line, body_font) or _looks_like_page_footer(line):
                continue
            if line.key in heading_keys or line.key in table_keys:
                continue
            if line.key in table_caption_keys or line.key in figure_keys:
                continue
            if line.key in image_text_keys:
                continue
            usable_lines.append(line)

        page_front_title: Optional[str] = None
        if first_numeric_page is None or page.page_index < first_numeric_page:
            page_front_title = _front_matter_title(usable_lines)

        events: List[Tuple[float, int, str, object]] = []
        for heading in accepted_headings:
            events.append((heading.y0, 0, "heading", heading))
        for table in page.tables:
            events.append((table.bbox[1], 1, "table", table))
        for caption in page.figure_captions:
            events.append((caption.bbox[1], 1, "figure_caption", caption))
        for image in page.images:
            events.append((image.bbox[1], 2, "image_barrier", image))
        for line in usable_lines:
            events.append((line.y0, 3, "line", line))
        events.sort(key=lambda item: (item[0], item[1]))

        pending_lines: List[TextLine] = []

        def flush_pending() -> None:
            nonlocal pending_lines
            if not pending_lines:
                return
            title = current_section_title
            if current_section_number is None and current_section_title is None:
                title = page_front_title
            for paragraph in _lines_to_paragraphs(pending_lines):
                if paragraph:
                    units.append(
                        ContentUnit(
                            content_type="text",
                            section_number=current_section_number,
                            section_title=title,
                            page_start=page.page_index + 1,
                            page_end=page.page_index + 1,
                            text=paragraph,
                            text_run_id=text_run_id,
                        )
                    )
            pending_lines = []

        for _position, _priority, kind, payload in events:
            if kind == "line":
                pending_lines.append(payload)  # type: ignore[arg-type]
                continue

            flush_pending()
            text_run_id += 1

            if kind == "heading":
                heading = payload  # type: ignore[assignment]
                current_section_number = heading.section_number
                current_section_title = heading.section_title
                continue

            if kind == "table":
                table = payload  # type: ignore[assignment]
                if table.text.strip():
                    units.append(
                        ContentUnit(
                            content_type="table",
                            section_number=current_section_number,
                            section_title=current_section_title or page_front_title,
                            page_start=page.page_index + 1,
                            page_end=page.page_index + 1,
                            text=table.text.strip(),
                            text_run_id=text_run_id,
                        )
                    )
                else:
                    warnings.add(
                        "PDF %d: an identified table region produced no textual chunk; inspect the source page manually."
                        % (page.page_index + 1)
                    )
                text_run_id += 1
                continue

            if kind == "figure_caption":
                caption = payload  # type: ignore[assignment]
                if caption.text.strip():
                    units.append(
                        ContentUnit(
                            content_type="figure_caption",
                            section_number=current_section_number,
                            section_title=current_section_title or page_front_title,
                            page_start=page.page_index + 1,
                            page_end=page.page_index + 1,
                            text=caption.text.strip(),
                            text_run_id=text_run_id,
                        )
                    )
                text_run_id += 1
                continue

            # image_barrier: no OCR / no invented text; the barrier only keeps
            # prose on opposite sides of a figure from being merged blindly.
            if kind == "image_barrier":
                text_run_id += 1

        flush_pending()

    return units


# ---------------------------------------------------------------------------
# Structural chunking and deterministic IDs
# ---------------------------------------------------------------------------


def _split_oversize_paragraph(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            word_piece = ""
            for word in words:
                proposed = word if not word_piece else word_piece + " " + word
                if len(proposed) <= max_chars:
                    word_piece = proposed
                else:
                    if word_piece:
                        pieces.append(word_piece)
                    if len(word) <= max_chars:
                        word_piece = word
                    else:
                        # Last-resort deterministic hard split for pathological
                        # unbroken tokens.  There is no overlap.
                        for offset in range(0, len(word), max_chars):
                            part = word[offset : offset + max_chars]
                            if len(part) == max_chars:
                                pieces.append(part)
                            else:
                                word_piece = part
            if word_piece:
                current = word_piece
            continue

        proposed = sentence if not current else current + " " + sentence
        if len(proposed) <= max_chars:
            current = proposed
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)
    return pieces or [text]


def _section_id_component(
    section_number: Optional[str],
    section_title: Optional[str],
) -> str:
    if section_number:
        return section_number
    if section_title:
        match = APPENDIX_RE.match(section_title)
        if match:
            return "%s-%s" % (
                match.group("kind").upper(),
                match.group("identifier").upper(),
            )
    return "UNSECTIONED"


def _sanitize_id_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("_.-")
    return cleaned or "DOCUMENT"


def _build_chunks(
    units: Sequence[ContentUnit],
    document_id: str,
    max_chars: int,
) -> List[ChunkRecord]:
    preliminary: List[Tuple[str, Optional[str], Optional[str], int, int, str]] = []
    index = 0

    while index < len(units):
        unit = units[index]
        if unit.content_type != "text":
            preliminary.append(
                (
                    unit.content_type,
                    unit.section_number,
                    unit.section_title,
                    unit.page_start,
                    unit.page_end,
                    unit.text,
                )
            )
            index += 1
            continue

        run_id = unit.text_run_id
        number = unit.section_number
        title = unit.section_title
        run: List[ContentUnit] = []
        while index < len(units):
            candidate = units[index]
            if (
                candidate.content_type != "text"
                or candidate.text_run_id != run_id
                or candidate.section_number != number
                or candidate.section_title != title
            ):
                break
            run.append(candidate)
            index += 1

        current_parts: List[str] = []
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        def flush_text_chunk() -> None:
            nonlocal current_parts, current_start, current_end
            if not current_parts or current_start is None or current_end is None:
                return
            preliminary.append(
                (
                    "text",
                    number,
                    title,
                    current_start,
                    current_end,
                    "\n\n".join(current_parts).strip(),
                )
            )
            current_parts = []
            current_start = None
            current_end = None

        for paragraph_unit in run:
            parts = _split_oversize_paragraph(paragraph_unit.text, max_chars)
            for part in parts:
                if current_parts:
                    proposed = "\n\n".join(current_parts + [part])
                else:
                    proposed = part
                if len(proposed) <= max_chars:
                    current_parts.append(part)
                    if current_start is None:
                        current_start = paragraph_unit.page_start
                    current_end = paragraph_unit.page_end
                else:
                    flush_text_chunk()
                    current_parts = [part]
                    current_start = paragraph_unit.page_start
                    current_end = paragraph_unit.page_end
        flush_text_chunk()

    doc_component = _sanitize_id_component(document_id)
    counters: Dict[Tuple[str, str], int] = {}
    type_tokens = {
        "text": "TEXT",
        "table": "TABLE",
        "figure_caption": "FIGURE",
    }
    records: List[ChunkRecord] = []

    for content_type, number, title, page_start, page_end, text in preliminary:
        section_component = _section_id_component(number, title)
        counter_key = (section_component, content_type)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        chunk_id = "%s-%s-%s-%03d" % (
            doc_component,
            section_component,
            type_tokens[content_type],
            counters[counter_key],
        )
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document=document_id,
                content_type=content_type,
                section_number=number,
                section_title=title,
                page_start=page_start,
                page_end=page_end,
                text=text,
            )
        )
    return records


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    pages: List[PageData]
    headings: List[HeadingCandidate]
    repeated_keys: Set[Tuple[int, int, int]]
    duplicate_running_keys: Set[Tuple[int, int, int]]
    first_numeric_page: Optional[int]
    first_appendix_position: Optional[Tuple[int, float]]
    body_font: float
    warnings: WarningCollector


def _load_pages(doc: fitz.Document) -> List[PageData]:
    pages: List[PageData] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        lines = _extract_page_lines(page)
        pages.append(
            PageData(
                page_index=page_index,
                width=float(page.rect.width),
                height=float(page.rect.height),
                lines=lines,
                is_navigation=_looks_like_navigation_page(lines),
            )
        )
    return pages


def _analyze_document(
    doc: fitz.Document,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> AnalysisResult:
    warnings = WarningCollector()
    pages = _load_pages(doc)
    body_font = _estimate_body_font(pages)
    repeated_keys = _detect_repeated_edge_lines(pages)
    toc_evidence, _toc_order = _extract_visible_toc_evidence(pages)
    bookmark_numeric, _bookmark_nonnumeric = _bookmark_evidence(doc)

    if not any(page.is_navigation for page in pages):
        warnings.add(
            "No visible TOC/navigation pages were confidently detected; section detection relies on layout and any bookmarks."
        )
    if not bookmark_numeric:
        warnings.add(
            "No numeric PDF bookmarks were available; bookmarks were not used as section evidence."
        )

    # Detect tables and raster images only after navigation classification.
    for page_data in pages:
        page = doc[page_data.page_index]
        if page_data.is_navigation:
            # TOC tables, if any, are intentionally not emitted.
            continue
        page_data.tables = _detect_tables(page, page_data.lines, warnings)
        page_data.images = _detect_raster_images(page)

    # Figure captions are detected from margin/header-filtered lines and kept
    # separate from body text.  Table regions are not allowed to generate figure
    # captions.
    for page_data in pages:
        if page_data.is_navigation:
            continue
        usable: List[TextLine] = []
        for line in page_data.lines:
            if _line_margin_filtered(line, top_margin_ratio, bottom_margin_ratio):
                continue
            if line.key in repeated_keys:
                continue
            if _looks_like_running_header(line, body_font) or _looks_like_page_footer(line):
                continue
            if any(_line_overlaps_region(line, table.bbox) for table in page_data.tables):
                continue
            if any(_line_overlaps_region(line, image.bbox, 0.60) for image in page_data.images):
                continue
            if any(line.key in table.caption_keys for table in page_data.tables):
                continue
            usable.append(line)
        page_data.figure_captions = _detect_figure_captions(
            page_data, usable, warnings
        )

    (
        headings,
        duplicate_running_keys,
        first_numeric_page,
        first_appendix_position,
    ) = _detect_headings(
        pages,
        repeated_keys=repeated_keys,
        top_margin_ratio=top_margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
        body_font=body_font,
        toc_evidence=toc_evidence,
        bookmark_numeric=bookmark_numeric,
        warnings=warnings,
    )
    _warn_bookmark_disagreement(
        headings, bookmark_numeric, warnings, first_appendix_position
    )

    if first_numeric_page is None:
        warnings.add(
            "No numeric ARINC body section heading was confidently detected. Unsectioned extractable text will be preserved, but verify the PDF layout."
        )

    return AnalysisResult(
        pages=pages,
        headings=headings,
        repeated_keys=repeated_keys,
        duplicate_running_keys=duplicate_running_keys,
        first_numeric_page=first_numeric_page,
        first_appendix_position=first_appendix_position,
        body_font=body_font,
        warnings=warnings,
    )


def _open_pdf(path: Path, password: Optional[str]) -> fitz.Document:
    try:
        doc = fitz.open(str(path))
    except (OSError, RuntimeError, fitz.FileDataError) as exc:
        raise ExtractionError("Could not open PDF %s: %s" % (path, exc))

    if doc.needs_pass:
        if not password or not doc.authenticate(password):
            doc.close()
            raise ExtractionError("The PDF is encrypted. Supply the correct --password.")
    if doc.page_count <= 0:
        doc.close()
        raise ExtractionError("The PDF contains no physical pages.")
    return doc


def extract_arinc(
    pdf_path: Path,
    *,
    document_id: str,
    max_chars: int,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    password: Optional[str],
) -> Tuple[List[ChunkRecord], AnalysisResult]:
    doc = _open_pdf(pdf_path, password)
    try:
        analysis = _analyze_document(
            doc,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        units = _build_content_units(
            analysis.pages,
            analysis.headings,
            repeated_keys=analysis.repeated_keys,
            duplicate_running_keys=analysis.duplicate_running_keys,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
            body_font=analysis.body_font,
            first_numeric_page=analysis.first_numeric_page,
            warnings=analysis.warnings,
        )
        records = _build_chunks(units, document_id, max_chars)
        if not records:
            raise ExtractionError(
                "Extraction produced an empty corpus; no JSONL was written. "
                "Inspect margins/sections and verify the PDF contains extractable text."
            )
        if not any(record.text.strip() for record in records):
            raise ExtractionError(
                "Extraction produced only empty chunks; no JSONL was written."
            )
        return records, analysis
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------


def _validate_ratio(name: str, value: float, upper: float = 0.30) -> None:
    if value < 0.0 or value >= upper:
        raise ValueError("%s must be >= 0 and < %.2f" % (name, upper))


def _write_jsonl(
    output_path: Path,
    records: Sequence[ChunkRecord],
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise ExtractionError(
            "Output already exists: %s (use --overwrite to replace it)" % output_path
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record.to_ordered_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
    except OSError as exc:
        raise ExtractionError("Could not write JSONL %s: %s" % (output_path, exc))
    return output_path.resolve()


def _print_warnings(warnings: Sequence[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print("- " + warning)


def _print_summary(
    pdf_path: Path,
    output_path: Path,
    records: Sequence[ChunkRecord],
    analysis: AnalysisResult,
) -> None:
    text_count = sum(record.content_type == "text" for record in records)
    table_count = sum(record.content_type == "table" for record in records)
    figure_count = sum(record.content_type == "figure_caption" for record in records)
    if analysis.first_numeric_page is None:
        body_pages = 0
    else:
        body_pages = sum(
            1
            for page in analysis.pages
            if page.page_index >= analysis.first_numeric_page and not page.is_navigation
        )

    print("Physical PDF pages: %d" % len(analysis.pages))
    print("Body pages processed: %d" % body_pages)
    print("Sections detected: %d" % len(analysis.headings))
    print("Text chunks: %d" % text_count)
    print("Table chunks: %d" % table_count)
    print("Figure-caption chunks: %d" % figure_count)
    print("Total chunks: %d" % len(records))
    print("JSONL: %s" % output_path)
    if analysis.warnings.items:
        print()
        _print_warnings(analysis.warnings.items)


def _inspect_sections_report(analysis: AnalysisResult) -> str:
    lines: List[str] = []
    for heading in analysis.headings:
        number = heading.section_number if heading.section_number is not None else "-"
        lines.append(
            "PDF %-4d %-12s %s"
            % (heading.page_index + 1, number, heading.section_title)
        )
    if not lines:
        lines.append("No body headings detected.")
    if analysis.warnings.items:
        lines.append("")
        lines.append("Warnings:")
        lines.extend("- " + warning for warning in analysis.warnings.items)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a complete ARINC 664 PDF into retrieval-friendly JSONL "
            "chunks with body text, tables, and figure captions kept separate."
        )
    )
    parser.add_argument("pdf", type=Path, help="path to the ARINC PDF")
    parser.add_argument(
        "--document-id",
        help="logical document identifier stored in every chunk, e.g. ARINC664P2 (required for normal extraction)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output JSONL path (default: <document_stem>_chunks.jsonl in current directory)",
    )
    parser.add_argument("--password", help="PDF password, if required")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="maximum body-text chunk size before structural subdivision (default: 3000)",
    )
    parser.add_argument(
        "--top-margin",
        type=float,
        default=0.09,
        help="fraction of page height removed geometrically at top (default: 0.04)",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=0.03,
        help="fraction of page height removed geometrically at bottom (default: 0.03)",
    )
    parser.add_argument(
        "--inspection-band",
        type=float,
        default=0.20,
        help="fraction of each page edge displayed by --inspect-margins (default: 0.20)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output JSONL file",
    )
    inspect_group = parser.add_mutually_exclusive_group()
    inspect_group.add_argument(
        "--inspect-margins",
        action="store_true",
        help="print edge-text filtering diagnostics and do not write JSONL",
    )
    inspect_group.add_argument(
        "--inspect-sections",
        action="store_true",
        help="print detected real body headings and do not write JSONL",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_ratio("--top-margin", args.top_margin)
        _validate_ratio("--bottom-margin", args.bottom_margin)
        _validate_ratio("--inspection-band", args.inspection_band, 0.50)
        if args.inspection_band <= 0.0:
            raise ValueError("--inspection-band must be > 0")
        if args.max_chars < 500:
            raise ValueError("--max-chars must be at least 500")
        if args.document_id is not None and not args.document_id.strip():
            raise ValueError("--document-id must not be empty")
        if not args.pdf.exists() or not args.pdf.is_file():
            raise ExtractionError("Input PDF does not exist: %s" % args.pdf)

        if args.inspect_margins:
            doc = _open_pdf(args.pdf, args.password)
            try:
                pages = _load_pages(doc)
                body_font = _estimate_body_font(pages)
                repeated = _detect_repeated_edge_lines(pages)
                print(
                    _format_margin_report(
                        pages,
                        repeated,
                        top_margin_ratio=args.top_margin,
                        bottom_margin_ratio=args.bottom_margin,
                        inspection_band=args.inspection_band,
                        body_font=body_font,
                    )
                )
            finally:
                doc.close()
            return 0

        if args.inspect_sections:
            doc = _open_pdf(args.pdf, args.password)
            try:
                analysis = _analyze_document(
                    doc,
                    top_margin_ratio=args.top_margin,
                    bottom_margin_ratio=args.bottom_margin,
                )
                print(_inspect_sections_report(analysis))
            finally:
                doc.close()
            return 0

        if args.document_id is None:
            raise ValueError("--document-id is required for normal extraction")

        output_path = args.output
        if output_path is None:
            output_path = Path(args.pdf.stem + "_chunks.jsonl")

        # Refuse before doing the expensive extraction where possible.
        if output_path.exists() and not args.overwrite:
            raise ExtractionError(
                "Output already exists: %s (use --overwrite to replace it)"
                % output_path
            )

        records, analysis = extract_arinc(
            args.pdf,
            document_id=args.document_id.strip(),
            max_chars=args.max_chars,
            top_margin_ratio=args.top_margin,
            bottom_margin_ratio=args.bottom_margin,
            password=args.password,
        )
        resolved_output = _write_jsonl(output_path, records, args.overwrite)
        _print_summary(args.pdf, resolved_output, records, analysis)
        return 0

    except (ExtractionError, ValueError, OSError, fitz.FileDataError) as exc:
        parser.exit(2, "Error: %s\n" % exc)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
