#!/usr/bin/env python3
"""Extract a lettered item (or one of its numeric children) from ARP4754.

Example
-------
    python extract_arp4754_item.py ARP4754.pdf "ARP4754 5.4.4.1 a"
    python extract_arp4754_item.py ARP4754.pdf "ARP4754 5.4.3 a(8)"
    python extract_arp4754_item.py ARP4754.pdf "5.4.3 a(8)" --inspect-margins

The extractor uses the PDF bookmarks / TOC to find the nearest indexed
ancestor (for example, 5.4.4), then finds the deeper heading (5.4.4.1) in
the page text.  A lettered item ends at the next peer item (b., c., ...)
or at the next peer / ancestor section heading.  For ``a(8)``, extraction is
further narrowed to numeric child ``(8)`` and stops before the next peer child.

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


class ReferenceExtractionError(RuntimeError):
    """Raised when the requested section or item cannot be located safely."""


@dataclass(frozen=True)
class ParsedReference:
    original: str
    section: str
    item: str
    subitem: int | None


@dataclass(frozen=True)
class TextLine:
    page_index: int          # zero-based physical PDF page index
    page_label: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float
    font_size: float
    bold: bool

    @property
    def x_ratio(self) -> float:
        return self.x0 / self.page_width if self.page_width else 0.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def y_ratio(self) -> float:
        center = (self.y0 + self.y1) / 2.0
        return center / self.page_height if self.page_height else 0.0


@dataclass(frozen=True)
class ExtractionResult:
    reference: str
    section: str
    item: str
    subitem: int | None
    text: str
    pdf_pages: list[int]     # one-based physical PDF page numbers
    page_labels: list[str]   # labels printed / defined by the PDF, if present

    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_RE = re.compile(
    r"^\s*(?:(?:ARP\s*)?4754[A-Z]?\s+)?"
    r"(?P<section>\d+(?:\.\d+)+)\s+(?P<item>[A-Za-z])\.?\s*"
    r"(?:\(\s*(?P<subitem>\d+)\s*\))?\s*$",
    re.IGNORECASE,
)
LEADING_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)(?=\s|$|[\-\u2013\u2014:])")
LETTERED_ITEM_RE = re.compile(r"^\s*([a-z])\.\s*(?:\S.*)?$", re.IGNORECASE)
NUMERIC_SUBITEM_RE = re.compile(
    r"^\s*(?:\((\d+)\)|(\d+)[.)])\s*(?:\S.*)?$"
)
CHILD_ITEM_RE = re.compile(
    r"^\s*(?:\(?\d+\)?[.)]?|\([a-z]\)|[\u2022\u25cf\u25aa\u2013\u2014])\s+",
    re.IGNORECASE,
)


def parse_reference(reference: str) -> ParsedReference:
    """Parse ``5.4.4.1 a`` or ``5.4.3 a(8)``; ARP4754 is optional."""
    match = REFERENCE_RE.fullmatch(reference)
    if not match:
        raise ValueError(
            f"Invalid reference {reference!r}; expected e.g. "
            '"ARP4754 5.4.4.1 a" or "ARP4754 5.4.3 a(8)".'
        )
    return ParsedReference(
        original=reference.strip(),
        section=match.group("section"),
        item=match.group("item").lower(),
        subitem=int(match.group("subitem")) if match.group("subitem") else None,
    )


def _section_tuple(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def _leading_section_number(text: str) -> str | None:
    match = LEADING_SECTION_RE.match(text)
    return match.group(1) if match else None


def _find_toc_window(
    doc: fitz.Document, target_section: str
) -> tuple[int, int, str]:
    """Return inclusive page window and the indexed ancestor section."""
    toc = doc.get_toc(simple=True)
    if not toc:
        raise ReferenceExtractionError(
            "The PDF has no readable TOC/bookmarks; cannot narrow the search range."
        )

    target = _section_tuple(target_section)
    candidates: list[tuple[int, int, int, str]] = []
    # tuple: (numeric depth, TOC index, page number, section string)
    for index, entry in enumerate(toc):
        _level, title, page_number = entry[:3]
        number = _leading_section_number(title)
        if not number or page_number < 1:
            continue
        parts = _section_tuple(number)
        if len(parts) <= len(target) and target[: len(parts)] == parts:
            candidates.append((len(parts), index, page_number, number))

    if not candidates:
        raise ReferenceExtractionError(
            f"No TOC ancestor was found for section {target_section}."
        )

    _depth, toc_index, start_page_number, ancestor = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    selected_level = int(toc[toc_index][0])
    start_index = start_page_number - 1
    end_index = doc.page_count - 1

    # Include the page on which the next bookmarked peer begins.  The precise
    # body heading detector below will stop before that heading; inclusion is
    # important when two sections share a physical page.
    for later in toc[toc_index + 1 :]:
        later_level, _title, later_page_number = later[:3]
        if later_page_number >= 1 and int(later_level) <= selected_level:
            end_index = min(doc.page_count - 1, later_page_number - 1)
            break

    if start_index < 0 or start_index >= doc.page_count:
        raise ReferenceExtractionError(
            f"TOC points section {ancestor} to invalid PDF page {start_page_number}."
        )
    return start_index, max(start_index, end_index), ancestor


def _page_label(page: fitz.Page) -> str:
    try:
        return page.get_label() or str(page.number + 1)
    except (AttributeError, RuntimeError):
        return str(page.number + 1)


def _extract_page_lines(
    page: fitz.Page,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> list[TextLine]:
    """Extract reading-order lines while dropping header/footer margin bands."""
    page_height = float(page.rect.height)
    page_width = float(page.rect.width)
    top = page_height * top_margin_ratio
    bottom = page_height * (1.0 - bottom_margin_ratio)
    page_dict = page.get_text("dict", sort=True)
    result: list[TextLine] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = map(float, line.get("bbox", (0, 0, 0, 0)))
            vertical_center = (y0 + y1) / 2.0
            if vertical_center < top or vertical_center > bottom:
                continue
            font_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
            bold = any(
                "bold" in str(span.get("font", "")).lower()
                or bool(int(span.get("flags", 0)) & 16)
                for span in spans
            )
            result.append(
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
                    font_size=font_size,
                    bold=bold,
                )
            )
    return result


def _extract_lines(
    doc: fitz.Document,
    start_page: int,
    end_page: int,
    *,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> list[TextLine]:
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


def _find_section_bounds(
    lines: Sequence[TextLine], target_section: str
) -> tuple[int, int]:
    target_parts = _section_tuple(target_section)
    heading_index: int | None = None

    for index, line in enumerate(lines):
        if _leading_section_number(line.text) == target_section:
            heading_index = index
            break
    if heading_index is None:
        raise ReferenceExtractionError(
            f"Section heading {target_section} was not found in the TOC-derived pages. "
            "Try reducing --top-margin if the heading is close to the page top."
        )

    target_heading = lines[heading_index]
    end_index = len(lines)
    for index in range(heading_index + 1, len(lines)):
        candidate = _leading_section_number(lines[index].text)
        if not candidate:
            continue
        candidate_parts = _section_tuple(candidate)
        if candidate_parts == target_parts or len(candidate_parts) > len(target_parts):
            continue

        # Section headings in these standards are normally bold and/or at
        # least as large as body text. Matching the target heading's style
        # prevents a body sentence beginning with a cross-reference from
        # becoming a false boundary.
        line = lines[index]
        style_matches = (
            line.font_size >= target_heading.font_size - 0.5
            and (not target_heading.bold or line.bold)
        )
        if style_matches:
            end_index = index
            break
    return heading_index, end_index


def _lettered_marker(text: str) -> str | None:
    match = LETTERED_ITEM_RE.match(text)
    return match.group(1).lower() if match else None


def _find_item_bounds(
    lines: Sequence[TextLine],
    section_start: int,
    section_end: int,
    item: str,
    *,
    indent_tolerance_ratio: float,
) -> tuple[int, int]:
    item_start: int | None = None
    for index in range(section_start + 1, section_end):
        if _lettered_marker(lines[index].text) == item:
            item_start = index
            break
    if item_start is None:
        raise ReferenceExtractionError(
            f"Item {item}. was not found inside the requested section."
        )

    start_indent = lines[item_start].x_ratio
    item_end = section_end
    for index in range(item_start + 1, section_end):
        marker = _lettered_marker(lines[index].text)
        if marker and abs(lines[index].x_ratio - start_indent) <= indent_tolerance_ratio:
            item_end = index
            break
    return item_start, item_end


def _numeric_subitem_marker(text: str) -> int | None:
    """Return the number from ``(8)``, ``8.`` or ``8)`` at line start."""
    match = NUMERIC_SUBITEM_RE.match(text)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _find_numeric_subitem_bounds(
    lines: Sequence[TextLine],
    item_start: int,
    item_end: int,
    subitem: int,
    *,
    indent_tolerance_ratio: float,
) -> tuple[int, int]:
    subitem_start: int | None = None
    for index in range(item_start + 1, item_end):
        if _numeric_subitem_marker(lines[index].text) == subitem:
            subitem_start = index
            break
    if subitem_start is None:
        raise ReferenceExtractionError(
            f"Numeric subitem ({subitem}) was not found inside the requested item."
        )

    start_indent = lines[subitem_start].x_ratio
    subitem_end = item_end
    for index in range(subitem_start + 1, item_end):
        marker = _numeric_subitem_marker(lines[index].text)
        if marker is not None and abs(
            lines[index].x_ratio - start_indent
        ) <= indent_tolerance_ratio:
            subitem_end = index
            break
    return subitem_start, subitem_end


def _edge_fingerprint(text: str) -> str:
    """Normalize changing page numbers so repeated headers can be recognized."""
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
        line
        for line in lines
        if line.y_ratio <= inspection_band
        and _edge_fingerprint(line.text) in repeated
    ]
    bottom_footers = [
        line
        for line in lines
        if line.y_ratio >= 1.0 - inspection_band
        and _edge_fingerprint(line.text) in repeated
    ]

    suggested_top: float | None = None
    if top_headers:
        last_header = max(line.y_ratio for line in top_headers)
        body_candidates = [
            line.y_ratio
            for line in lines
            if last_header < line.y_ratio <= inspection_band
            and _edge_fingerprint(line.text) not in repeated
        ]
        first_body = min(body_candidates, default=min(inspection_band, last_header + 0.03))
        suggested_top = round((last_header + first_body) / 2.0, 3)

    suggested_bottom: float | None = None
    if bottom_footers:
        first_footer = min(line.y_ratio for line in bottom_footers)
        body_candidates = [
            line.y_ratio
            for line in lines
            if 1.0 - inspection_band <= line.y_ratio < first_footer
            and _edge_fingerprint(line.text) not in repeated
        ]
        last_body = max(
            body_candidates,
            default=max(1.0 - inspection_band, first_footer - 0.03),
        )
        cutoff = (last_body + first_footer) / 2.0
        suggested_bottom = round(1.0 - cutoff, 3)
    return suggested_top, suggested_bottom


def inspect_margins(
    pdf_path: str | Path,
    reference: str,
    *,
    password: str | None = None,
    top_margin_ratio: float = 0.07,
    bottom_margin_ratio: float = 0.07,
    inspection_band: float = 0.20,
) -> dict:
    """Show which edge lines current margins keep/filter and suggest cutoffs."""
    if not 0 < inspection_band < 0.5:
        raise ValueError("inspection_band must be between 0 and 0.5")
    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass and (not password or not doc.authenticate(password)):
            raise ReferenceExtractionError(
                "The PDF is encrypted. Supply the correct password."
            )
        start_page, end_page, ancestor = _find_toc_window(doc, parsed.section)
        raw_lines = _extract_lines(
            doc,
            start_page,
            end_page,
            top_margin_ratio=0.0,
            bottom_margin_ratio=0.0,
        )
        edge_lines = [
            line
            for line in raw_lines
            if line.y_ratio <= inspection_band
            or line.y_ratio >= 1.0 - inspection_band
        ]
        suggested_top, suggested_bottom = _suggest_margins(
            raw_lines, inspection_band
        )
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
            "reference": parsed.original,
            "toc_ancestor": ancestor,
            "pdf_page_range": [start_page + 1, end_page + 1],
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
        f"TOC ancestor: {report['toc_ancestor']}",
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


def _render_lines(lines: Sequence[TextLine], *, join_wrapped_lines: bool) -> str:
    if not lines:
        return ""
    if not join_wrapped_lines:
        return "\n".join(line.text for line in lines).strip()

    heights = [line.height for line in lines]
    typical_height = statistics.median(heights) if heights else 10.0
    chunks: list[str] = [lines[0].text]
    previous = lines[0]

    for line in lines[1:]:
        page_changed = line.page_index != previous.page_index
        vertical_gap = line.y0 - previous.y1 if not page_changed else 0.0
        indented = line.x_ratio > previous.x_ratio + 0.018
        new_list_item = bool(CHILD_ITEM_RE.match(line.text) or _lettered_marker(line.text))
        new_paragraph = new_list_item or indented or vertical_gap > typical_height * 0.9

        if new_paragraph:
            chunks.append("\n" + line.text)
        elif chunks[-1].endswith("-") and line.text[:1].islower():
            # Rejoin words hyphenated only because of a visual PDF line break.
            chunks[-1] = chunks[-1][:-1] + line.text
        else:
            chunks.append(" " + line.text)
        previous = line
    return "".join(chunks).strip()


def extract_reference(
    pdf_path: str | Path,
    reference: str,
    *,
    password: str | None = None,
    top_margin_ratio: float = 0.07,
    bottom_margin_ratio: float = 0.07,
    indent_tolerance_ratio: float = 0.03,
    join_wrapped_lines: bool = False,
) -> ExtractionResult:
    """Extract a lettered ARP4754 item and the pages containing it.

    ``pdf_pages`` are one-based physical PDF page numbers. ``page_labels`` are
    the labels defined inside the PDF (often the printed document page number).
    Header/footer removal is geometric: adjust the margin ratios when necessary.
    """
    if not 0 <= top_margin_ratio < 0.25:
        raise ValueError("top_margin_ratio must be between 0 and 0.25")
    if not 0 <= bottom_margin_ratio < 0.25:
        raise ValueError("bottom_margin_ratio must be between 0 and 0.25")

    parsed = parse_reference(reference)
    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass:
            if not password or not doc.authenticate(password):
                raise ReferenceExtractionError(
                    "The PDF is encrypted. Supply the correct password."
                )

        start_page, end_page, _ancestor = _find_toc_window(doc, parsed.section)
        lines = _extract_lines(
            doc,
            start_page,
            end_page,
            top_margin_ratio=top_margin_ratio,
            bottom_margin_ratio=bottom_margin_ratio,
        )
        section_start, section_end = _find_section_bounds(lines, parsed.section)
        item_start, item_end = _find_item_bounds(
            lines,
            section_start,
            section_end,
            parsed.item,
            indent_tolerance_ratio=indent_tolerance_ratio,
        )
        selected_start, selected_end = item_start, item_end
        if parsed.subitem is not None:
            selected_start, selected_end = _find_numeric_subitem_bounds(
                lines,
                item_start,
                item_end,
                parsed.subitem,
                indent_tolerance_ratio=indent_tolerance_ratio,
            )
        selected = lines[selected_start:selected_end]
        pages = list(dict.fromkeys(line.page_index + 1 for line in selected))
        labels = list(dict.fromkeys(line.page_label for line in selected))
        return ExtractionResult(
            reference=parsed.original,
            section=parsed.section,
            item=parsed.item,
            subitem=parsed.subitem,
            text=_render_lines(selected, join_wrapped_lines=join_wrapped_lines),
            pdf_pages=pages,
            page_labels=labels,
        )
    finally:
        doc.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an item such as ARP4754 5.4.4.1 a or "
            "ARP4754 5.4.3 a(8) from a PDF."
        )
    )
    parser.add_argument("pdf", type=Path, help="path to the ARP4754 PDF")
    parser.add_argument(
        "reference",
        help='e.g. "ARP4754 5.4.4.1 a" or "ARP4754 5.4.3 a(8)"',
    )
    parser.add_argument("--password", help="PDF password, if required")
    parser.add_argument(
        "--top-margin",
        type=float,
        default=0.07,
        help="fraction of page height ignored at top (default: 0.07)",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=0.07,
        help="fraction of page height ignored at bottom (default: 0.07)",
    )
    parser.add_argument(
        "--indent-tolerance",
        type=float,
        default=0.03,
        help="page-width fraction for deciding whether b. is a peer of a. (default: 0.03)",
    )
    parser.add_argument(
        "--join-wrapped-lines",
        action="store_true",
        help="join visual PDF line wraps while preserving detected list items",
    )
    parser.add_argument(
        "--inspect-margins",
        action="store_true",
        help=(
            "show edge text, current filter decisions, and suggested top/bottom "
            "margins instead of extracting the item"
        ),
    )
    parser.add_argument(
        "--inspection-band",
        type=float,
        default=0.20,
        help="fraction of each page edge shown by --inspect-margins (default: 0.20)",
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
                inspection_band=args.inspection_band,
            )
            print(_format_margin_report(report))
            return 0
        result = extract_reference(
            args.pdf,
            args.reference,
            password=args.password,
            top_margin_ratio=args.top_margin,
            bottom_margin_ratio=args.bottom_margin,
            indent_tolerance_ratio=args.indent_tolerance,
            join_wrapped_lines=args.join_wrapped_lines,
        )
    except (OSError, ValueError, ReferenceExtractionError, fitz.FileDataError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())