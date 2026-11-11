"""Write a Markdown summary of missing enrichment fields in three datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    label: str
    path: Path


DEFAULT_DATASETS = (
    Dataset("Bench", ROOT / "data_extraction" / "bench_road_enrichment.csv"),
    Dataset("Plaque", ROOT / "data_extraction" / "open-plaques-United-Kingdom-2025-12-14-footway.csv"),
    Dataset("Monument", ROOT / "data_extraction" / "monuments_enriched.csv"),
)

FIELD_ALIASES = {
    "distance": ("distance_m", "distance_to_road", "dist"),
    "city": ("city",),
    "highway": ("highway",),
}


def is_missing(value: str | None) -> bool:
    """Treat blank and common serialized null values as missing."""

    return value is None or value.strip().lower() in {"", "nan", "none", "null"}


def find_column(headers: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {header.strip().lower(): header for header in headers}
    return next((normalized[name] for name in aliases if name in normalized), None)


def inspect_dataset(dataset: Dataset) -> tuple[int, dict[str, tuple[int, str | None]]]:
    if not dataset.path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset.path}")

    with dataset.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header row: {dataset.path}")
        columns = {name: find_column(reader.fieldnames, aliases) for name, aliases in FIELD_ALIASES.items()}
        missing = {name: 0 for name in FIELD_ALIASES}
        rows = 0
        for row in reader:
            rows += 1
            for field_name, column in columns.items():
                if column is None or is_missing(row.get(column)):
                    missing[field_name] += 1

    return rows, {name: (count, columns[name]) for name, count in missing.items()}


def format_missing(count: int, total: int, column: str | None) -> str:
    percent = (100.0 * count / total) if total else 0.0
    rendered = f"{count:,} ({percent:.1f}%)"
    return rendered if column is not None else f"{rendered} (column absent)"


def build_report(datasets: Iterable[Dataset]) -> str:
    lines = [
        "# Missing enrichment values",
        "",
        "Missing cells include blanks and serialized `nan`, `none`, or `null` values. Percentages use each dataset's data-row count as the denominator.",
        "",
        "| Dataset | Rows | Distance missing (rows, %) | City missing (rows, %) | Highway missing (rows, %) |",
        "|---|---:|---:|---:|---:|",
    ]

    for dataset in datasets:
        total, stats = inspect_dataset(dataset)
        values = [
            format_missing(count, total, column)
            for count, column in (stats[field] for field in FIELD_ALIASES)
        ]
        lines.append(
            f"| {dataset.label} (`{dataset.path.name}`) | {total:,} | "
            + " | ".join(values)
            + " |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data_extraction" / "enrichment_missing_summary.md",
        help="Markdown report output path",
    )
    parser.add_argument("--bench", type=Path, default=DEFAULT_DATASETS[0].path, help="Bench enrichment CSV")
    parser.add_argument("--plaque", type=Path, default=DEFAULT_DATASETS[1].path, help="Plaque enrichment CSV")
    parser.add_argument("--monument", type=Path, default=DEFAULT_DATASETS[2].path, help="Monument enrichment CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = (
        Dataset("Bench", args.bench),
        Dataset("Plaque", args.plaque),
        Dataset("Monument", args.monument),
    )
    report = build_report(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
