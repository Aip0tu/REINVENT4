#!/usr/bin/env python
"""Extract strict FLAME hits from a REINVENT scoring CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_EM_COLUMN = "FLAME Emission (raw)"
DEFAULT_PLQY_COLUMN = "FLAME PLQY (raw)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a REINVENT scoring CSV and keep only molecules with "
            "emission and PLQY above the requested FLAME thresholds."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Scoring CSV produced by REINVENT scoring mode.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default="flame_hits_em1000_plqy04.csv",
        help="Output CSV containing only the matching rows.",
    )
    parser.add_argument(
        "--em-column",
        default=DEFAULT_EM_COLUMN,
        help="Column containing the raw FLAME emission values.",
    )
    parser.add_argument(
        "--plqy-column",
        default=DEFAULT_PLQY_COLUMN,
        help="Column containing the raw FLAME PLQY values.",
    )
    parser.add_argument(
        "--em-min",
        type=float,
        default=1000.0,
        help="Keep rows with emission strictly larger than this value.",
    )
    parser.add_argument(
        "--plqy-min",
        type=float,
        default=0.4,
        help="Keep rows with PLQY strictly larger than this value.",
    )
    return parser.parse_args()


def read_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    value = value.strip()

    if not value:
        return None

    return float(value)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    with input_csv.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise RuntimeError(f"No header found in {input_csv}")

        missing = [
            column
            for column in (args.em_column, args.plqy_column)
            if column not in reader.fieldnames
        ]
        if missing:
            raise RuntimeError(
                f"Missing required column(s) in {input_csv}: {', '.join(missing)}"
            )

        hits: list[dict[str, str]] = []

        for row in reader:
            emission = read_float(row, args.em_column)
            plqy = read_float(row, args.plqy_column)

            if emission is None or plqy is None:
                continue

            if emission > args.em_min and plqy > args.plqy_min:
                hits.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(hits)

    print(
        f"Filtered {len(hits)} hit(s) from {input_csv} "
        f"into {output_csv} using emission > {args.em_min} and PLQY > {args.plqy_min}."
    )


if __name__ == "__main__":
    main()
