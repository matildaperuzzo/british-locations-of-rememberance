"""Add local built-up-area hectares to the enriched datasets."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AREAS = Path(__file__).resolve().with_name("os_open_built_up_areas.gpkg")
LAYER_NAME = "os_open_built_up_areas"
BUFFER_M = 100.0


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    latitude_column: str
    longitude_column: str


def normalize_name(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return " ".join(normalized.split())


def is_missing(value: str | None) -> bool:
    return value is None or value.strip().casefold() in {"", "nan", "none", "null"}


def valid_coordinate(lat_text: str | None, lon_text: str | None) -> tuple[float, float] | None:
    try:
        lat, lon = float(lat_text), float(lon_text)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill areahectares in the bench, plaque, and monument CSVs from the "
            "OS Open Built Up Areas areahectares field."
        )
    )
    parser.add_argument("--bench", type=Path, default=ROOT / "data_extraction" / "bench_road_enrichment.csv")
    parser.add_argument(
        "--plaque",
        type=Path,
        default=ROOT / "data_extraction" / "open-plaques-United-Kingdom-2025-12-14-footway.csv",
    )
    parser.add_argument("--monument", type=Path, default=ROOT / "data_extraction" / "monuments_enriched.csv")
    parser.add_argument("--areas", type=Path, default=DEFAULT_AREAS, help="OS built-up-area GeoPackage")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def write_csv_atomically(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    specs = (
        DatasetSpec("Bench", args.bench, "lat", "lon"),
        DatasetSpec("Plaque", args.plaque, "latitude", "longitude"),
        DatasetSpec("Monument", args.monument, "@lat", "@lon"),
    )
    for spec in specs:
        if not spec.path.is_file():
            logging.error("Dataset not found: %s", spec.path)
            return 1
    if not args.areas.is_file():
        logging.error("Built-up-area GeoPackage not found: %s", args.areas)
        return 1

    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point
    except ModuleNotFoundError as err:
        logging.error("City-size lookup needs GeoPandas, pyproj, and Shapely: %s", err)
        return 1

    datasets: list[tuple[DatasetSpec, list[dict], list[str]]] = []
    for spec in specs:
        with spec.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                logging.error("Dataset has no header row: %s", spec.path)
                return 1
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
        if "areahectares" not in fieldnames:
            fieldnames.append("areahectares")
        datasets.append((spec, rows, fieldnames))

    logging.info("Loading built-up-area polygons from %s", args.areas)
    areas = gpd.read_file(
        args.areas,
        layer=LAYER_NAME,
        columns=["name1_text", "areahectares", "geometry_area_m"],
    )
    logging.info("Loaded %s built-up-area polygons", len(areas))
    if areas.crs is None:
        logging.error("Built-up-area layer has no coordinate reference system")
        return 1

    transformer = Transformer.from_crs("EPSG:4326", areas.crs, always_xy=True)
    spatial_index = areas.sindex

    def area_for_city(city: str, lat: float, lon: float) -> float | None:
        x, y = transformer.transform(lon, lat)
        point = Point(x, y)
        nearby = spatial_index.query(point.buffer(BUFFER_M), predicate="intersects")
        if len(nearby) == 0:
            return None
        candidates = areas.iloc[nearby].copy()
        candidates["_distance"] = candidates.geometry.distance(point)
        candidates = candidates[candidates["_distance"] <= BUFFER_M]
        candidates = candidates[
            candidates["name1_text"].map(lambda name: normalize_name(str(name))) == normalize_name(city)
        ]
        if candidates.empty:
            return None
        candidates = candidates.sort_values(["_distance", "geometry_area_m"])
        value = candidates.iloc[0]["areahectares"]
        return None if value is None else float(value)

    for spec, rows, fieldnames in datasets:
        logging.info("Processing %s dataset (%s rows): %s", spec.label, len(rows), spec.path)
        distinct_cities: set[str] = set()
        filled = 0
        missing_city = 0
        invalid_coordinates = 0
        for row_index, row in enumerate(rows, start=1):
            if row_index % 5000 == 0:
                logging.info("%s: checked %s / %s rows", spec.label, row_index, len(rows))
            city = row.get("city")
            if is_missing(city):
                missing_city += 1
                continue
            distinct_cities.add(normalize_name(city))
            if not is_missing(row.get("areahectares")):
                continue
            coordinates = valid_coordinate(row.get(spec.latitude_column), row.get(spec.longitude_column))
            if coordinates is None:
                invalid_coordinates += 1
                continue
            size = area_for_city(city, *coordinates)
            if size is not None:
                row["areahectares"] = size
                filled += 1

        write_csv_atomically(spec.path, rows, fieldnames)
        logging.info(
            "%s: %s rows, %s distinct city names, filled %s areahectares values; %s missing city names; %s invalid coordinates",
            spec.label,
            len(rows),
            len(distinct_cities),
            filled,
            missing_city,
            invalid_coordinates,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
