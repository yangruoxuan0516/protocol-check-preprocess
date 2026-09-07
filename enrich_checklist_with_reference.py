#!/usr/bin/env python3
"""Retrieve every parsed checklist reference and enrich checklist JSONL.

The input is the JSONL produced by the updated ``extract_checklist.py``.
Every input record remains one output record. Retrieval failures and unparsed
reference fragments are recorded in place instead of aborting the full run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from extract_arp import extract_reference as extract_arp4754_reference
from extract_std import extract_reference as extract_std_reference
from extract_do import extract_reference as extract_do297_reference


Extractor = Callable[..., Any]


class ChecklistEnrichmentError(RuntimeError):
    """Raised when checklist JSONL cannot be enriched safely."""


def _document_key(value: object) -> str:
    normalized = "".join(
        character for character in str(value or "").upper() if character.isalnum()
    )
    aliases = {
        "ARP4754": "ARP4754",
        "ARP4754A": "ARP4754",
        "ARP4754B": "ARP4754",
        "ENGSTD011": "ENG-STD-011",
        "DO297": "DO-297",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ChecklistEnrichmentError(
            f"Unsupported reference document: {value!r}"
        ) from exc


def _result_as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        converted = result
    elif hasattr(result, "to_dict"):
        converted = result.to_dict()
    elif is_dataclass(result) and not isinstance(result, type):
        converted = asdict(result)
    else:
        raise ChecklistEnrichmentError(
            "Extractor returned an unsupported result type: "
            f"{type(result).__name__}"
        )

    if not isinstance(converted, dict):
        raise ChecklistEnrichmentError("Extractor result is not a dictionary")
    text = converted.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ChecklistEnrichmentError("Extractor returned empty text")
    return converted


def _error_outcome(error: BaseException) -> dict[str, Any]:
    return {
        "retrieval_status": "error",
        "content": None,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _retrieve_once(
    *,
    pdf_path: Path,
    extractor: Extractor,
    target: str,
    password: str | None,
    join_wrapped_lines: bool,
) -> dict[str, Any]:
    try:
        result = extractor(
            pdf_path,
            target,
            password=password,
            join_wrapped_lines=join_wrapped_lines,
        )
        return {
            "retrieval_status": "ok",
            "content": _result_as_dict(result),
        }
    except Exception as error:
        return _error_outcome(error)


def enrich_checklist_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    arp4754_pdf: str | Path,
    std_pdf: str | Path,
    do297_pdf: str | Path,
    arp4754_password: str | None = None,
    std_password: str | None = None,
    do297_password: str | None = None,
    join_wrapped_lines: bool = True,
) -> dict[str, int]:
    """Enrich each JSONL record with its reference retrieval results."""
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output JSONL paths must be different")

    document_config: dict[str, tuple[Path, Extractor, str | None]] = {
        "ARP4754": (
            Path(arp4754_pdf),
            extract_arp4754_reference,
            arp4754_password,
        ),
        "ENG-STD-011": (Path(std_pdf), extract_std_reference, std_password),
        "DO-297": (Path(do297_pdf), extract_do297_reference, do297_password),
    }
    for document, (pdf_path, _extractor, _password) in document_config.items():
        if not pdf_path.is_file():
            raise FileNotFoundError(f"{document} PDF not found: {pdf_path}")

    stats = {
        "jsonl_records": 0,
        "reference_occurrences": 0,
        "references_ok": 0,
        "retrieval_errors": 0,
        "unparsed_reference_parts": 0,
        "records_with_reference_issues": 0,
        "unique_retrievals": 0,
        "cache_hits": 0,
    }
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ChecklistEnrichmentError(
                    f"Invalid JSON on {input_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ChecklistEnrichmentError(
                    f"Expected a JSON object on {input_path}:{line_number}"
                )

            stats["jsonl_records"] += 1
            record_has_issue = False

            unparsed = record.get("reference_unparsed")
            if unparsed is None:
                unparsed = []
                if record.get("reference_raw"):
                    unparsed = [
                        "The input was generated without reference_unparsed; "
                        "rerun the updated extract_checklist.py"
                    ]
            elif not isinstance(unparsed, list):
                unparsed = [f"Invalid reference_unparsed value: {unparsed!r}"]

            parse_issues = []
            for fragment in unparsed:
                parse_issues.append(
                    {
                        "fragment": fragment,
                        "status": "parse_error",
                        "error_type": "UnsupportedReferenceFormat",
                        "error": "This part of Reference could not be parsed.",
                    }
                )
            record["reference_unparsed"] = unparsed
            record["reference_parse_issues"] = parse_issues
            stats["unparsed_reference_parts"] += len(parse_issues)
            if parse_issues:
                record_has_issue = True

            references = record.get("references")
            if not isinstance(references, list):
                references = []
                record.setdefault("enrichment_errors", []).append(
                    {
                        "error_type": "InvalidReferencesField",
                        "error": "The references field is not a list.",
                    }
                )
                record_has_issue = True

            enriched_references: list[dict[str, Any]] = []
            for reference in references:
                stats["reference_occurrences"] += 1
                if not isinstance(reference, dict):
                    outcome = _error_outcome(
                        ChecklistEnrichmentError("Reference is not a JSON object")
                    )
                    enriched_references.append(
                        {"original_reference": reference, **outcome}
                    )
                    stats["retrieval_errors"] += 1
                    record_has_issue = True
                    continue

                enriched_reference = dict(reference)
                try:
                    document = _document_key(reference.get("document"))
                    target = reference.get("target")
                    if not isinstance(target, str) or not target.strip():
                        raise ChecklistEnrichmentError(
                            "Reference has no non-empty target"
                        )
                    target = target.strip()
                    cache_key = (document, target)
                    if cache_key in cache:
                        stats["cache_hits"] += 1
                        outcome = cache[cache_key]
                    else:
                        pdf_path, extractor, password = document_config[document]
                        stats["unique_retrievals"] += 1
                        outcome = _retrieve_once(
                            pdf_path=pdf_path,
                            extractor=extractor,
                            target=target,
                            password=password,
                            join_wrapped_lines=join_wrapped_lines,
                        )
                        cache[cache_key] = outcome
                except Exception as error:
                    outcome = _error_outcome(error)

                enriched_reference.update(outcome)
                enriched_references.append(enriched_reference)
                if outcome["retrieval_status"] == "ok":
                    stats["references_ok"] += 1
                else:
                    stats["retrieval_errors"] += 1
                    record_has_issue = True

            record["references"] = enriched_references
            if record_has_issue:
                stats["records_with_reference_issues"] += 1
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve every checklist reference and write enriched checklist JSONL."
        )
    )
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: <input_stem>_enriched.jsonl)",
    )
    parser.add_argument("--arp4754-pdf", type=Path, required=True)
    parser.add_argument("--std-pdf", type=Path, required=True)
    parser.add_argument("--do297-pdf", type=Path, required=True)
    parser.add_argument("--arp4754-password")
    parser.add_argument("--std-password")
    parser.add_argument("--do297-password", default="912hyx4")
    parser.add_argument(
        "--join-wrapped-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="join visual PDF line wraps (default: enabled)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = args.output or args.input_jsonl.with_name(
        f"{args.input_jsonl.stem}_enriched.jsonl"
    )
    stats = enrich_checklist_jsonl(
        args.input_jsonl,
        output_path,
        arp4754_pdf=args.arp4754_pdf,
        std_pdf=args.std_pdf,
        do297_pdf=args.do297_pdf,
        arp4754_password=args.arp4754_password,
        std_password=args.std_password,
        do297_password=args.do297_password,
        join_wrapped_lines=args.join_wrapped_lines,
    )
    print(json.dumps({"output": str(output_path), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())