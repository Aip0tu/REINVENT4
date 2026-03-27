#!/usr/bin/env python
"""Screen molecules with FLAME_plugin using emission and PLQY thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reinvent_plugins.FLAME_plugin import flam_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLAME emission and PLQY prediction on a SMILES file and keep "
            "molecules whose raw values exceed the requested thresholds."
        )
    )
    parser.add_argument(
        "input_file",
        help="Input .csv or .smi file.",
    )
    parser.add_argument(
        "--smiles-column",
        default="SMILES",
        help="SMILES column to read when the input is CSV.",
    )
    parser.add_argument(
        "--solvent",
        default="O",
        help="Solvent token passed to FLAME together with each SMILES.",
    )
    parser.add_argument(
        "--em-min",
        type=float,
        default=1000.0,
        help="Strict emission threshold in nm.",
    )
    parser.add_argument(
        "--plqy-min",
        type=float,
        default=0.4,
        help="Strict PLQY threshold.",
    )
    parser.add_argument(
        "--output-all",
        default="flame_screen_em1000_plqy04.csv",
        help="Output CSV with all predictions.",
    )
    parser.add_argument(
        "--output-hits",
        default="flame_hits_em1000_plqy04.csv",
        help="Output CSV containing only matching molecules.",
    )
    return parser.parse_args()


def read_input(path: Path, smiles_column: str) -> tuple[list[str], list[dict[str, str]], list[str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                raise RuntimeError(f"No header found in {path}")
            if smiles_column not in reader.fieldnames:
                raise RuntimeError(f"SMILES column '{smiles_column}' not found in {path}")
            rows = list(reader)
            smiles = [row[smiles_column].strip() for row in rows]
            return smiles, rows, list(reader.fieldnames)

    if path.suffix.lower() == ".smi":
        rows = []
        smiles = []
        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                smile = parts[0]
                smiles.append(smile)
                rows.append({"SMILES": smile, "Comment": " ".join(parts[1:])})
        return smiles, rows, ["SMILES", "Comment"]

    raise RuntimeError(f"Unsupported input format for {path}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    all_output = Path(args.output_all)
    hits_output = Path(args.output_hits)

    smiles, original_rows, original_fields = read_input(input_file, args.smiles_column)
    paired_inputs = [[smile, args.solvent] for smile in smiles]

    print(f"[1/2] Predicting FLAME emission for {len(paired_inputs)} molecule(s)...")
    emission = flam_predict("emi", paired_inputs)
    print(f"[2/2] Predicting FLAME PLQY for {len(paired_inputs)} molecule(s)...")
    plqy = flam_predict("plqy", paired_inputs)

    result_fields = original_fields + [
        "FLAME_Emission_nm",
        "FLAME_PLQY",
        "meets_emission_threshold",
        "meets_plqy_threshold",
        "is_hit",
    ]

    all_rows: list[dict[str, str]] = []
    hit_rows: list[dict[str, str]] = []

    for row, em_value, plqy_value in zip(original_rows, emission, plqy):
        em_score = float(em_value)
        plqy_score = float(plqy_value)
        em_hit = em_score > args.em_min
        plqy_hit = plqy_score > args.plqy_min
        is_hit = em_hit and plqy_hit

        output_row = dict(row)
        output_row["FLAME_Emission_nm"] = f"{em_score:.6f}"
        output_row["FLAME_PLQY"] = f"{plqy_score:.6f}"
        output_row["meets_emission_threshold"] = str(em_hit)
        output_row["meets_plqy_threshold"] = str(plqy_hit)
        output_row["is_hit"] = str(is_hit)

        all_rows.append(output_row)
        if is_hit:
            hit_rows.append(output_row)

    write_csv(all_output, result_fields, all_rows)
    write_csv(hits_output, result_fields, hit_rows)

    print(
        f"Processed {len(all_rows)} molecules from {input_file}. "
        f"Found {len(hit_rows)} hit(s) with emission > {args.em_min} and PLQY > {args.plqy_min}."
    )
    print(f"All results: {all_output}")
    print(f"Hits only: {hits_output}")


if __name__ == "__main__":
    main()
