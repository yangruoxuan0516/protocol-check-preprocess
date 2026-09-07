#!/usr/bin/env python3
"""Extract four-column NIO tables from a DOCX file.

Expected columns, by position:
    1. ID (NIO-<digits>)
    2. Specification
    3. Rationale (may be empty)
    4. Req (Yes/No, or empty)

Each extracted record is classified as either:
    - section_header
    - requirement

A section header is identified only when the Specification begins with a
hierarchical numeric section number, for example:
    1 Introduction
    1.1 General
    4.3.2 Hosted Functions

Output:
    <document_stem>_records.jsonl

Usage:
    python extract_protocol.py protocol.docx
    python extract_protocol.py protocol.docx --output-dir extracted
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from docx import Document


ID_PATTERN = re.compile(r"^NIO-\d+$", re.IGNORECASE)

# A valid section header must begin with:
#
#     <number>(.<number>)* <text>
#
# Examples:
#     1 Introduction
#     1.1 General
#     4.3.2 Hosted Functions
#
# Leading whitespace is allowed.
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*)\s+\S"
)

SECTION_NUMBER_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")

HEADER_ID_NAMES = {"id", "identifier", "nio id", "nio-id"}


def clean_cell_text(text: str) -> str:
    """Normalize Word whitespace while retaining meaningful line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def canonical_req(value: str) -> Optional[str]:
    """Normalize Req values while preserving unexpected non-empty values."""
    value_lower = value.strip().lower()

    if not value_lower:
        return None
    if value_lower == "yes":
        return "Yes"
    if value_lower == "no":
        return "No"

    return value.strip()


def is_header_row(first_cell: str) -> bool:
    """Return True for a table's column-header row."""
    return first_cell.strip().lower() in HEADER_ID_NAMES


def get_section_number(specification: str) -> Optional[str]:
    """Return the hierarchical section number if Specification is a header."""
    match = SECTION_HEADER_PATTERN.match(specification)

    if match is None:
        return None

    return match.group("number")


def classify_record(specification: str) -> str:
    """Classify a record using only its leading numeric section prefix."""
    if get_section_number(specification) is not None:
        return "section_header"

    return "requirement"


def parse_section_number(section_number: str) -> Tuple[int, ...]:
    """Convert '4.3.2' into the comparable tuple (4, 3, 2)."""
    return tuple(int(part) for part in section_number.split("."))


def record_location(record: Dict[str, Any]) -> str:
    """Return a compact human-readable source location for diagnostics."""
    source = record["source"]
    return "table {}, row {}".format(source["table"], source["word_row"])


def detect_malformed_numeric_prefix(
    specification: str,
) -> Optional[str]:
    """Detect text that resembles a malformed hierarchical section prefix.

    This does not affect classification. It is only a diagnostic intended to
    catch likely section headers that failed the strict section-header regex.
    """
    stripped = specification.lstrip()

    if not stripped or not stripped[0].isdigit():
        return None

    parts = stripped.split(None, 1)
    first_token = parts[0]

    # A valid token such as "4.3.2" followed by text would already have been
    # classified as a section header. If the token is valid but has no title,
    # it is suspicious.
    if SECTION_NUMBER_TOKEN_PATTERN.fullmatch(first_token):
        if len(parts) == 1:
            return first_token
        return None

    # Only flag malformed numeric-looking tokens containing a dot. This keeps
    # ordinary requirements beginning with values such as "100 Mbps" from
    # generating unnecessary warnings.
    if "." in first_token:
        return first_token

    return None


def extract_records(
    docx_path: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract records from all tables and return records plus warnings."""
    document = Document(docx_path)

    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen_ids: Set[str] = set()

    for table_index, table in enumerate(document.tables, start=1):
        for row_index, row in enumerate(table.rows, start=1):
            cells = [clean_cell_text(cell.text) for cell in row.cells]

            if not any(cells):
                continue

            if is_header_row(cells[0]):
                continue

            if len(cells) != 4:
                warnings.append(
                    "Table {}, row {}: expected 4 columns, found {}; "
                    "row skipped.".format(
                        table_index,
                        row_index,
                        len(cells),
                    )
                )
                continue

            nio_id, specification, rationale, req_text = cells

            nio_id = nio_id.upper()
            req = canonical_req(req_text)
            record_type = classify_record(specification)

            row_errors: List[str] = []

            if not ID_PATTERN.fullmatch(nio_id):
                row_errors.append("unexpected ID {!r}".format(nio_id))

            if not specification:
                row_errors.append("empty specification")

            # An empty Req is valid and stored as null.
            if req is not None and req not in {"Yes", "No"}:
                row_errors.append(
                    "Req is not Yes/No: {!r}".format(req)
                )

            if nio_id in seen_ids:
                row_errors.append(
                    "duplicate ID {!r}".format(nio_id)
                )

            if nio_id:
                seen_ids.add(nio_id)

            if row_errors:
                warnings.append(
                    "Table {}, row {}: {}".format(
                        table_index,
                        row_index,
                        "; ".join(row_errors),
                    )
                )

            records.append(
                {
                    "id": nio_id,
                    "specification": specification,
                    "rationale": rationale or None,
                    "req": req,
                    "type": record_type,
                    "source": {
                        "file": docx_path.name,
                        "table": table_index,
                        "word_row": row_index,
                    },
                }
            )

    if not document.tables:
        warnings.append("The document contains no tables.")

    return records, warnings


def format_section_number(parts: Tuple[int, ...]) -> str:
    """Convert a parsed section tuple back into dotted notation."""
    return ".".join(str(part) for part in parts)


def format_missing_siblings(
    parent: Tuple[int, ...],
    first_missing: int,
    last_missing: int,
) -> str:
    """Format one or more missing sibling section numbers."""
    missing_count = last_missing - first_missing + 1

    if missing_count <= 5:
        numbers = []

        for number in range(first_missing, last_missing + 1):
            numbers.append(
                format_section_number(parent + (number,))
            )

        return ", ".join(numbers)

    first = format_section_number(parent + (first_missing,))
    last = format_section_number(parent + (last_missing,))

    return "{} through {}".format(first, last)


def validate_section_headers(
    records: List[Dict[str, Any]],
) -> List[str]:
    """Perform diagnostic checks on detected section-header numbering.

    No records are changed or removed by these checks.
    """
    warnings: List[str] = []

    section_headers: List[
        Tuple[str, Tuple[int, ...], Dict[str, Any]]
    ] = []

    # Collect valid detected section headers in document order.
    for record in records:
        if record["type"] != "section_header":
            continue

        section_number = get_section_number(record["specification"])

        if section_number is None:
            # This should not normally be possible because classification uses
            # the same parser, but keep the validation defensive.
            warnings.append(
                "{}: record classified as section_header but no valid "
                "section number could be parsed.".format(
                    record_location(record)
                )
            )
            continue

        try:
            parsed = parse_section_number(section_number)
        except ValueError:
            warnings.append(
                "{}: malformed hierarchical section number {!r}.".format(
                    record_location(record),
                    section_number,
                )
            )
            continue

        section_headers.append(
            (section_number, parsed, record)
        )

    # Also look for specifications that resemble malformed section headers but
    # were correctly classified as requirements because they failed the strict
    # header regex.
    for record in records:
        if record["type"] == "section_header":
            continue

        malformed_prefix = detect_malformed_numeric_prefix(
            record["specification"]
        )

        if malformed_prefix is not None:
            warnings.append(
                "{}: numeric-looking prefix {!r} does not match the "
                "section-header pattern; record remains classified as "
                "requirement.".format(
                    record_location(record),
                    malformed_prefix,
                )
            )

    if not section_headers:
        return warnings

    # Check exact duplicate section numbers.
    seen_sections: Dict[
        Tuple[int, ...],
        Dict[str, Any],
    ] = {}

    for section_number, parsed, record in section_headers:
        previous_record = seen_sections.get(parsed)

        if previous_record is not None:
            warnings.append(
                "{}: duplicate/extra section number {} "
                "(previously detected at {}).".format(
                    record_location(record),
                    section_number,
                    record_location(previous_record),
                )
            )
        else:
            seen_sections[parsed] = record

    # Check overall document ordering.
    previous_number = section_headers[0][0]
    previous_parsed = section_headers[0][1]
    previous_record = section_headers[0][2]

    for section_number, parsed, record in section_headers[1:]:
        if parsed < previous_parsed:
            warnings.append(
                "{}: section {} appears after section {} at {}; "
                "numbering is descending/out of order.".format(
                    record_location(record),
                    section_number,
                    previous_number,
                    record_location(previous_record),
                )
            )

        previous_number = section_number
        previous_parsed = parsed
        previous_record = record

    # Group sections by immediate parent and check sibling numbering.
    #
    # For example:
    #     4.3.2.1
    #     4.3.2.2
    #     4.3.2.4
    #
    # are siblings under parent 4.3.2, so 4.3.2.3 is suspiciously absent.
    sibling_groups: Dict[
        Tuple[int, ...],
        List[Tuple[int, str, Dict[str, Any]]],
    ] = {}

    for section_number, parsed, record in section_headers:
        parent = parsed[:-1]
        sibling_number = parsed[-1]

        sibling_groups.setdefault(parent, []).append(
            (sibling_number, section_number, record)
        )

    for parent, siblings in sibling_groups.items():
        if len(siblings) < 2:
            continue

        previous_sibling = siblings[0][0]
        previous_section = siblings[0][1]
        previous_record = siblings[0][2]

        for sibling_number, section_number, record in siblings[1:]:
            if sibling_number > previous_sibling + 1:
                missing = format_missing_siblings(
                    parent,
                    previous_sibling + 1,
                    sibling_number - 1,
                )

                warnings.append(
                    "{}: section {} follows sibling section {} at {}; "
                    "possible missing intermediate section(s): {}.".format(
                        record_location(record),
                        section_number,
                        previous_section,
                        record_location(previous_record),
                        missing,
                    )
                )

            previous_sibling = sibling_number
            previous_section = section_number
            previous_record = record

    # Check for missing immediate parent headers.
    #
    # Example:
    #     4.3.2
    #     4.3.2.1.1
    #
    # without 4.3.2.1 is suspicious and can indicate that a header was not
    # classified correctly.
    seen_before: Set[Tuple[int, ...]] = set()

    for section_number, parsed, record in section_headers:
        if len(parsed) > 1:
            parent = parsed[:-1]

            if parent not in seen_before:
                # Avoid warning for the first detected section if the document
                # starts part-way through a hierarchy. Once related numbering
                # is established, a missing immediate parent is more useful
                # diagnostically.
                related_seen = any(
                    parsed[: min(len(parsed), len(previous))]
                    == previous[: min(len(parsed), len(previous))]
                    for previous in seen_before
                )

                if related_seen:
                    warnings.append(
                        "{}: section {} was detected, but its immediate "
                        "parent section {} was not detected earlier; "
                        "hierarchy may be inconsistent.".format(
                            record_location(record),
                            section_number,
                            format_section_number(parent),
                        )
                    )

        seen_before.add(parsed)

    return warnings


def write_jsonl(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write extracted records as UTF-8 JSON Lines."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract four-column NIO tables from a DOCX file "
            "into JSONL."
        )
    )

    parser.add_argument(
        "docx",
        type=Path,
        help="Path to the input .docx file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory "
            "(default: next to the input document)"
        ),
    )

    return parser.parse_args()


def print_warnings(
    heading: str,
    warnings: List[str],
) -> None:
    """Print a warning group."""
    if not warnings:
        return

    print()
    print(heading)

    for warning in warnings:
        print("- {}".format(warning))


def main() -> int:
    args = parse_args()

    docx_path = args.docx.expanduser().resolve()

    if not docx_path.is_file():
        print(
            "Error: file not found: {}".format(docx_path),
            file=sys.stderr,
        )
        return 2

    if docx_path.suffix.lower() != ".docx":
        print(
            "Error: input file must have a .docx extension",
            file=sys.stderr,
        )
        return 2

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = docx_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    records, extraction_warnings = extract_records(docx_path)

    section_warnings = validate_section_headers(records)

    base_name = docx_path.stem
    jsonl_path = output_dir / "{}_records.jsonl".format(base_name)

    write_jsonl(records, jsonl_path)

    section_header_count = sum(
        record["type"] == "section_header"
        for record in records
    )

    print("Extracted {} records.".format(len(records)))
    print(
        "Detected {} section headers.".format(
            section_header_count
        )
    )
    print("JSONL: {}".format(jsonl_path))

    print_warnings(
        "Extraction warnings:",
        extraction_warnings,
    )

    if section_warnings:
        print_warnings(
            "Section numbering warnings:",
            section_warnings,
        )
    else:
        print()
        print(
            "Section numbering check: "
            "no obvious issues found."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())