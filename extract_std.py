#!/usr/bin/env python3
"""Extract one RS requirement from an ENG-STD-011 PDF.

Examples
--------
    python extract_std_rs_item.py ENG-STD-011.pdf RS36
    python extract_std_rs_item.py ENG-STD-011.pdf "ENG-STD-011 RS36"
    python extract_std_rs_item.py ENG-STD-011.pdf RS36 --inspect-margins

The PDF does not need bookmarks.  A real RS item is recognized as a bold,
line-leading ``RS<number>`` marker in the document's fixed RS column.  The
item ends immediately before the next bold RS marker in the same column.

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
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

DEFAULT_TOP_MARGIN = 0.07
DEFAULT_BOTTOM_MARGIN = 0.09
DEFAULT_MARKER_COLUMN_TOLERANCE = 0.025


class ReferenceExtractionError(RuntimeError):
    """Raised when an RS item cannot be located safely."""

@dataclass(frozen=True)
class ContentFlags:
    contains_table: bool | None
    contains_image: bool
    
@dataclass(frozen=True)
class ParsedReference:
    original: str
    number: int

    @property
    def canonical(self) -> str:
        return f"RS{self.number}"


@dataclass(frozen=True)
class TextLine:
    page_index: int
    page_label: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    leading_bold: bool

    @property
    def x_ratio(self) -> float:
        return self.x0 / self.page_width if self.page_width else 0.0

    @property
    def y_ratio(self) -> float:
        center = (self.y0 + self.y1) / 2.0
        return center / self.page_height if self.page_height else 0.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class MarkerCandidate:
    line_index: int
    number: int


@dataclass(frozen=True)
class ExtractionResult:
    reference: str
    text: str
    pdf_pages: list[int]
    page_labels: list[str]
    next_reference: str | None
    content_flags: ContentFlags
    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_RE = re.compile(
    r"^\s*(?:ENG[\s-]*STD[\s-]*011\s+)?RS[\s-]*(?P<number>\d+)\s*$",
    re.IGNORECASE,
)
RS_MARKER_RE = re.compile(
    r"^\s*RS[\s-]*(?P<number>\d+)(?=\s|$|[:.])",
    re.IGNORECASE,
)
LIST_MARKER_RE = re.compile(
    r"^\s*(?:\(?\d+\)?[.)]?|\([a-z]\)|[a-z][.)]|"
    r"[\u2022\u25cf\u25aa\u2013\u2014])\s+",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)+)\.?(?=\s|$)",
    re.IGNORECASE,
)

CHAPTER_HEADING_RE = re.compile(
    r"^\s*CHAPTER\s+(?P<number>\d+)\b",
    re.IGNORECASE,
)

def _detect_content_flags(
    doc: fitz.Document,
    lines: Sequence[TextLine],
    selected_start: int,
    selected_end: int,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> ContentFlags:
    """Detect tables and raster images intersecting the selected RS region."""
    if selected_start >= selected_end or not lines:
        return ContentFlags(contains_table=False, contains_image=False)

    start_line = lines[selected_start]
    boundary = lines[selected_end] if selected_end < len(lines) else None
    end_page = boundary.page_index if boundary else lines[-1].page_index
    contains_table: bool | None = False
    contains_image = False

    for page_index in range(start_line.page_index, end_page + 1):
        page = doc[page_index]
        region_top = page.rect.height * top_margin_ratio
        region_bottom = page.rect.height * (1.0 - bottom_margin_ratio)
        if page_index == start_line.page_index:
            region_top = max(region_top, start_line.y0)
        if boundary is not None and page_index == boundary.page_index:
            region_bottom = min(region_bottom, boundary.y0)
        if region_bottom <= region_top:
            continue

        if not contains_image:
            for block in page.get_text("dict", sort=True).get("blocks", []):
                if block.get("type") != 1:
                    continue
                _x0, y0, _x1, y1 = map(
                    float, block.get("bbox", (0, 0, 0, 0))
                )
                if y1 > region_top and y0 < region_bottom:
                    contains_image = True
                    break

        if contains_table is not True:
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    tables = page.find_tables().tables
                if any(
                    float(table.bbox[3]) > region_top
                    and float(table.bbox[1]) < region_bottom
                    for table in tables
                ):
                    contains_table = True
            except (AttributeError, RuntimeError, ValueError):
                contains_table = None

    return ContentFlags(
        contains_table=contains_table,
        contains_image=contains_image,
    )
    
def parse_reference(reference: str) -> ParsedReference:
    """Parse ``RS36`` or ``ENG-STD-011 RS36``."""
    match = REFERENCE_RE.fullmatch(reference)
    if not match:
        raise ValueError(
            f"Invalid reference {reference!r}; expected e.g. "
            '"RS36" or "ENG-STD-011 RS36".'
        )
    return ParsedReference(reference.strip(), int(match.group("number")))


def _page_label(page: fitz.Page) -> str:
    try:
        return page.get_label() or str(page.number + 1)
    except (AttributeError, RuntimeError):
        return str(page.number + 1)


def _span_is_bold(span: dict) -> bool:
    return (
        "bold" in str(span.get("font", "")).lower()
        or bool(int(span.get("flags", 0)) & 16)
    )


def _join_span_text(spans: Sequence[dict]) -> str:
    """Join spans while restoring a space across visibly separate columns."""
    pieces: list[str] = []
    previous_x1: float | None = None
    previous_size = 0.0
    for span in spans:
        text = str(span.get("text", ""))
        if not text:
            continue
        x0, _y0, x1, _y1 = map(float, span.get("bbox", (0, 0, 0, 0)))
        size = float(span.get("size", 0.0))
        if (
            pieces
            and previous_x1 is not None
            and x0 - previous_x1 > max(1.5, previous_size * 0.18)
            and not pieces[-1].endswith(" ")
            and not text.startswith(" ")
        ):
            pieces.append(" ")
        pieces.append(text)
        previous_x1 = x1
        previous_size = size
    return "".join(pieces).strip()


def _merge_same_visual_rows(lines: Sequence[TextLine]) -> list[TextLine]:
    """Merge a separate RS column and body column occupying the same row."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda line: (line.center_y, line.x0))
    rows: list[list[TextLine]] = []

    for line in sorted_lines:
        placed = False
        for row in reversed(rows[-2:]):
            row_y0 = min(item.y0 for item in row)
            row_y1 = max(item.y1 for item in row)
            overlap = min(row_y1, line.y1) - max(row_y0, line.y0)
            if overlap >= min(line.height, row_y1 - row_y0) * 0.35:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    merged: list[TextLine] = []
    for row in rows:
        row.sort(key=lambda line: line.x0)
        first = row[0]
        text = " ".join(item.text for item in row if item.text).strip()
        merged.append(
            TextLine(
                page_index=first.page_index,
                page_label=first.page_label,
                text=text,
                x0=min(item.x0 for item in row),
                y0=min(item.y0 for item in row),
                x1=max(item.x1 for item in row),
                y1=max(item.y1 for item in row),
                page_width=first.page_width,
                page_height=first.page_height,
                leading_bold=first.leading_bold,
            )
        )
    return sorted(merged, key=lambda line: (line.y0, line.x0))


def _extract_page_lines(
    page: fitz.Page,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> list[TextLine]:
    page_height = float(page.rect.height)
    page_width = float(page.rect.width)
    top = page_height * top_margin_ratio
    bottom = page_height * (1.0 - bottom_margin_ratio)
    raw_lines: list[TextLine] = []

    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = _join_span_text(spans)
            if not text:
                continue
            x0, y0, x1, y1 = map(float, line.get("bbox", (0, 0, 0, 0)))
            center = (y0 + y1) / 2.0
            if center < top or center > bottom:
                continue
            first_span = next(
                (span for span in spans if str(span.get("text", "")).strip()),
                {},
            )
            raw_lines.append(
                TextLine(
                    page_index=page.number,
                    page_label=_page_label(page),
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_width=page_width,
                    page_height=page_height,
                    leading_bold=_span_is_bold(first_span),
                )
            )
    return _merge_same_visual_rows(raw_lines)


def _extract_lines(
    doc: fitz.Document,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
    start_page: int = 0,
    end_page: int | None = None,
) -> list[TextLine]:
    if end_page is None:
        end_page = doc.page_count - 1
    lines: list[TextLine] = []
    for page_index in range(start_page, end_page + 1):
        lines.extend(
            _extract_page_lines(
                doc[page_index],
                top_margin_ratio=top_margin_ratio,
                bottom_margin_ratio=bottom_margin_ratio,
            )
        )
    return lines


def _rs_marker_number(text: str) -> int | None:
    match = RS_MARKER_RE.match(text)
    return int(match.group("number")) if match else None

def _section_heading_label(text: str) -> str | None:
    match = SECTION_HEADING_RE.match(text)
    if match:
        return match.group("number")

    match = CHAPTER_HEADING_RE.match(text)
    if match:
        return f"CHAPTER {match.group('number')}"

    return None

def _marker_candidates(
    lines: Sequence[TextLine], *, require_bold: bool
) -> list[MarkerCandidate]:
    candidates: list[MarkerCandidate] = []
    for index, line in enumerate(lines):
        number = _rs_marker_number(line.text)
        if number is not None and (line.leading_bold or not require_bold):
            candidates.append(MarkerCandidate(index, number))
    return candidates


def _find_item_bounds(
    lines: Sequence[TextLine],
    target_number: int,
    *,
    require_bold: bool,
    marker_column_tolerance: float,
) -> tuple[int, int, int | None]:
    candidates = _marker_candidates(lines, require_bold=require_bold)
    targets = [candidate for candidate in candidates if candidate.number == target_number]
    if not targets:
        unbold_matches = [
            line
            for line in lines
            if _rs_marker_number(line.text) == target_number
        ]
        hint = (
            " Line-leading matches exist but are not detected as bold; "
            "inspect the PDF or try --allow-nonbold-markers."
            if unbold_matches and require_bold
            else ""
        )
        raise ReferenceExtractionError(f"RS{target_number} was not found.{hint}")

    # If the same text appears more than once, prefer the occurrence in the
    # column containing the greatest number of distinct bold RS markers.
    def column_support(candidate: MarkerCandidate) -> tuple[int, int]:
        x = lines[candidate.line_index].x_ratio
        numbers = {
            other.number
            for other in candidates
            if abs(lines[other.line_index].x_ratio - x) <= marker_column_tolerance
        }
        return len(numbers), -candidate.line_index

    start_candidate = max(targets, key=column_support)
    start_index = start_candidate.line_index
    marker_x = lines[start_index].x_ratio
    end_index = len(lines)
    next_number: int | None = None
    for candidate in candidates:
        if candidate.line_index <= start_index:
            continue
        if abs(lines[candidate.line_index].x_ratio - marker_x) <= marker_column_tolerance:
            end_index = candidate.line_index
            next_number = candidate.number
            break
    for index in range(start_index + 1, end_index):
        line = lines[index]

        is_section_heading = (
            line.leading_bold
            and _section_heading_label(line.text) is not None
            and line.x_ratio <= marker_x + marker_column_tolerance
        )

        if is_section_heading:
            end_index = index
            next_number = None
            break
    return start_index, end_index, next_number


def _render_lines(lines: Sequence[TextLine], *, join_wrapped_lines: bool) -> str:
    if not lines:
        return ""
    if not join_wrapped_lines:
        return "\n".join(line.text for line in lines).strip()

    typical_height = statistics.median(line.height for line in lines)
    chunks: list[str] = [lines[0].text]
    previous = lines[0]
    for line in lines[1:]:
        page_changed = line.page_index != previous.page_index
        vertical_gap = line.y0 - previous.y1 if not page_changed else 0.0
        indented = line.x_ratio > previous.x_ratio + 0.018
        prev_ends_sentences = bool(
            re.search(r'[.!?]["”’)\]]*$', previous.text.rstrip())
        )
        new_list_item = bool(LIST_MARKER_RE.match(line.text))
        new_paragraph = (
            new_list_item
            or (
                prev_ends_sentences
            )
            and {
                indented
                or vertical_gap > typical_height * 0.9
            }
        )
        if new_paragraph:
            chunks.append("\n" + line.text)
        elif chunks[-1].endswith("-") and line.text[:1].islower():
            chunks[-1] = chunks[-1][:-1] + line.text
        else:
            chunks.append(" " + line.text)
        previous = line
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
    marker_column_tolerance: float = DEFAULT_MARKER_COLUMN_TOLERANCE,
    require_bold: bool = True,
    join_wrapped_lines: bool = False,
) -> ExtractionResult:
    """Extract one RS item and its physical/labelled PDF pages."""
    if not 0 <= top_margin_ratio < 0.25:
        raise ValueError("top_margin_ratio must be between 0 and 0.25")
    if not 0 <= bottom_margin_ratio < 0.25:
        raise ValueError("bottom_margin_ratio must be between 0 and 0.25")
    if not 0 <= marker_column_tolerance < 0.25:
        raise ValueError("marker_column_tolerance must be between 0 and 0.25")

    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        _authenticate(doc, password)
        lines = _extract_lines(
            doc,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        start, end, next_number = _find_item_bounds(
            lines,
            parsed.number,
            require_bold=require_bold,
            marker_column_tolerance=marker_column_tolerance,
        )
        content_flags = _detect_content_flags(
            doc,
            lines,
            start,
            end,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        selected = lines[start:end]
        return ExtractionResult(
            reference=parsed.canonical,
            text=_render_lines(selected, join_wrapped_lines=join_wrapped_lines),
            pdf_pages=list(
                dict.fromkeys(line.page_index + 1 for line in selected)
            ),
            page_labels=list(dict.fromkeys(line.page_label for line in selected)),
            next_reference=f"RS{next_number}" if next_number is not None else None,
            content_flags=content_flags,
        )
    finally:
        doc.close()


def _edge_fingerprint(text: str) -> str:
    normalized = re.sub(r"\d+", "#", text.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _suggest_margins(
    lines: Sequence[TextLine], inspection_band: float
) -> tuple[float | None, float | None]:
    fingerprint_pages: dict[str, set[int]] = {}
    for line in lines:
        if line.y_ratio <= inspection_band or line.y_ratio >= 1.0 - inspection_band:
            fingerprint_pages.setdefault(_edge_fingerprint(line.text), set()).add(
                line.page_index
            )
    repeated = {
        fingerprint
        for fingerprint, pages in fingerprint_pages.items()
        if len(pages) >= 2
    }

    top_headers = [
        line for line in lines
        if line.y_ratio <= inspection_band
        and _edge_fingerprint(line.text) in repeated
    ]
    bottom_footers = [
        line for line in lines
        if line.y_ratio >= 1.0 - inspection_band
        and _edge_fingerprint(line.text) in repeated
    ]

    suggested_top: float | None = None
    if top_headers:
        last_header = max(line.y_ratio for line in top_headers)
        possible_body = [
            line.y_ratio for line in lines
            if last_header < line.y_ratio <= inspection_band
            and _edge_fingerprint(line.text) not in repeated
        ]
        first_body = min(
            possible_body,
            default=min(inspection_band, last_header + 0.03),
        )
        suggested_top = round((last_header + first_body) / 2.0, 3)

    suggested_bottom: float | None = None
    if bottom_footers:
        first_footer = min(line.y_ratio for line in bottom_footers)
        possible_body = [
            line.y_ratio for line in lines
            if 1.0 - inspection_band <= line.y_ratio < first_footer
            and _edge_fingerprint(line.text) not in repeated
        ]
        last_body = max(
            possible_body,
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
    marker_column_tolerance: float = DEFAULT_MARKER_COLUMN_TOLERANCE,
    require_bold: bool = True,
    inspection_band: float = 0.20,
    inspection_page_count: int = 5,
) -> dict:
    """Inspect edge lines around the target RS page without applying margins."""
    if not 0 < inspection_band < 0.5:
        raise ValueError("inspection_band must be between 0 and 0.5")
    if inspection_page_count < 1:
        raise ValueError("inspection_page_count must be at least 1")

    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        _authenticate(doc, password)
        all_raw_lines = _extract_lines(
            doc,
            top_margin_ratio=0.0,
            bottom_margin_ratio=0.0,
        )
        start, _end, _next = _find_item_bounds(
            all_raw_lines,
            parsed.number,
            require_bold=require_bold,
            marker_column_tolerance=marker_column_tolerance,
        )
        first_page = all_raw_lines[start].page_index
        last_page = min(doc.page_count - 1, first_page + inspection_page_count - 1)
        lines = [
            line for line in all_raw_lines
            if first_page <= line.page_index <= last_page
        ]
        edge_lines = [
            line for line in lines
            if line.y_ratio <= inspection_band
            or line.y_ratio >= 1.0 - inspection_band
        ]
        suggested_top, suggested_bottom = _suggest_margins(lines, inspection_band)
        entries = []
        for line in edge_lines:
            if line.y_ratio <= inspection_band:
                edge = "top"
                distance = line.y_ratio
                filtered = line.y_ratio < top_margin_ratio
            else:
                edge = "bottom"
                distance = 1.0 - line.y_ratio
                filtered = line.y_ratio > 1.0 - bottom_margin_ratio
            entries.append(
                {
                    "pdf_page": line.page_index + 1,
                    "page_label": line.page_label,
                    "edge": edge,
                    "distance_ratio": round(distance, 4),
                    "status": "FILTERED" if filtered else "KEPT",
                    "text": line.text,
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
        description="Extract a bold, column-aligned RS item from ENG-STD-011."
    )
    parser.add_argument("pdf", type=Path, help="path to the ENG-STD-011 PDF")
    parser.add_argument("reference", help='e.g. "RS36"')
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
        "--marker-column-tolerance",
        type=float,
        default=DEFAULT_MARKER_COLUMN_TOLERANCE,
        help=(
            "page-width fraction used to decide whether RS markers share a "
            f"column (default: {DEFAULT_MARKER_COLUMN_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--allow-nonbold-markers",
        action="store_true",
        help="accept line-leading RS markers even if PDF font metadata is not bold",
    )
    parser.add_argument(
        "--join-wrapped-lines",
        action="store_true",
        default=True,
        help="join visual PDF line wraps while preserving detected list items",
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
        help="number of pages inspected starting at the target RS page (default: 5)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        require_bold = not args.allow_nonbold_markers
        if args.inspect_margins:
            report = inspect_margins(
                args.pdf,
                args.reference,
                password=args.password,
                top_margin_ratio=args.top_margin,
                bottom_margin_ratio=args.bottom_margin,
                marker_column_tolerance=args.marker_column_tolerance,
                require_bold=require_bold,
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
            marker_column_tolerance=args.marker_column_tolerance,
            require_bold=require_bold,
            join_wrapped_lines=args.join_wrapped_lines,
        )
    except (OSError, ValueError, ReferenceExtractionError, fitz.FileDataError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())