"""Batch plaque enrichment that adds a highway column using clustered Overpass queries.

The script reads the Open Plaques CSV, finds the nearest OSM highway for each
plaque location, and writes the nearest OSM highway type (e.g. footway,
residential, cycleway). Empty values indicate no nearby highway geometry.

Progress is flushed after each cluster, so reruns can resume from the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
import numpy as np
from identify_settlement import DEFAULT_DATASET, find_built_up_area, load_built_up_areas

from enrichment_common import (
    GeoPoint,
    build_overpass_bbox_query,
    cluster_points,
    fetch_overpass_json,
    is_in_uk,
    load_uk_geometry,
    nearest_road_for_point,
    read_existing_csv,
    setup_logging,
    write_csv,
)


def normalize_column_name(name: str) -> str:
    """Normalize CSV header names for robust matching."""

    return name.replace("\ufeff", "").strip().lower()


def parse_args() -> argparse.Namespace:
    """Define CLI arguments and parse command line input."""

    parser = argparse.ArgumentParser(
        description=(
            "Batch-enrich Open Plaques rows with a highway column using "
            "clustered bbox Overpass requests."
        )
    )

    parser.add_argument(
        "--input",
        default="data_extraction/open-plaques-United-Kingdom-2025-12-14.csv",
        help="Path to input Open Plaques CSV",
    )
    parser.add_argument(
        "--output",
        default="data_extraction/open-plaques-United-Kingdom-2025-12-14-footway.csv",
        help="Path to output CSV with added highway column",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Input CSV column used as stable row identifier",
    )
    parser.add_argument(
        "--lat-column",
        default="latitude",
        help="Input CSV latitude column",
    )
    parser.add_argument(
        "--lon-column",
        default="longitude",
        help="Input CSV longitude column",
    )
    parser.add_argument(
        "--rerun",
        choices=("distance", "city", "all"),
        default="all",
        help=(
            "Fields to fill when missing: distance (distance_m and highway), "
            "city, or all. Existing populated values are preserved."
        ),
    )

    parser.add_argument(
        "--cluster-cell-m",
        type=float,
        default=300.0,
        help="Grid cell size in meters used to form point clusters",
    )
    parser.add_argument(
        "--bbox-buffer-m",
        type=float,
        default=120.0,
        help="Extra meter buffer added around each cluster bbox",
    )

    parser.add_argument(
        "--overpass-endpoint",
        default="https://overpass-api.de/api/interpreter",
        help="Overpass interpreter endpoint URL",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=25,
        help="Overpass query timeout hint in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for HTTP 429/5xx errors",
    )
    parser.add_argument(
        "--retry-wait-s",
        type=int,
        default=30,
        help="Base wait time in seconds before retrying failed requests",
    )

    parser.add_argument(
        "--skip-uk-filter",
        action="store_true",
        help="If set, do not filter points by UK boundary",
    )
    parser.add_argument(
        "--shapefile-cache-dir",
        default="data_extraction/shapefiles",
        help="Directory used to cache UK boundary shapefile files",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    """Parse float from CSV cell, returning None for blanks/invalid values."""

    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def value_is_missing(value: object) -> bool:
    """Treat blank cells and serialized null values as missing."""

    return value is None or str(value).strip().lower() in {"", "nan", "none", "null"}


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        return 1

    uk_geometry = None
    if not args.skip_uk_filter:
        logging.info("Loading UK boundary geometry for point filtering")
        uk_geometry = load_uk_geometry(Path(args.shapefile_cache_dir))

    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        input_fieldnames = reader.fieldnames or []
        source_rows = list(reader)

    normalized_to_original = {normalize_column_name(name): name for name in input_fieldnames}
    requested_columns = {
        "id": normalize_column_name(args.id_column),
        "lat": normalize_column_name(args.lat_column),
        "lon": normalize_column_name(args.lon_column),
    }

    missing_columns = [
        original
        for original in (args.id_column, args.lat_column, args.lon_column)
        if normalize_column_name(original) not in normalized_to_original
    ]
    if missing_columns:
        logging.error("Input CSV is missing required columns: %s", ", ".join(sorted(missing_columns)))
        return 1

    id_column = normalized_to_original[requested_columns["id"]]
    lat_column = normalized_to_original[requested_columns["lat"]]
    lon_column = normalized_to_original[requested_columns["lon"]]

    output_fieldnames = list(input_fieldnames)
    if "highway" not in output_fieldnames:
        output_fieldnames.append("highway")
    if "distance_m" not in output_fieldnames:
        output_fieldnames.append("distance_m")
    if "city" not in output_fieldnames:
        output_fieldnames.append("city")
    if "road_status" not in output_fieldnames:
        output_fieldnames.append("road_status")

    existing_rows = read_existing_csv(output_path, key_column=id_column)
    if existing_rows:
        logging.info("Loaded %s existing rows from %s", len(existing_rows), output_path)

    rows: List[dict] = []
    kept_points: List[GeoPoint] = []
    row_by_feature_id: Dict[str, dict] = {}

    for idx, source_row in enumerate(source_rows, start=1):
        row_id = (source_row.get(id_column) or "").strip() or str(idx)
        lat_value = parse_float(source_row.get(lat_column))
        lon_value = parse_float(source_row.get(lon_column))

        row = dict(source_row)
        row.setdefault("highway", np.nan)
        row.setdefault("distance_m", np.nan)
        row.setdefault("city", np.nan)
        row.setdefault("road_status", "")

        existing_row = existing_rows.get(row_id)
        if existing_row is not None:
            row["highway"] = existing_row.get("highway", np.nan)
            row["distance_m"] = existing_row.get("distance_m", np.nan)
            row["city"] = existing_row.get("city", np.nan)
            row["road_status"] = existing_row.get("road_status", "")

        has_coordinates = lat_value is not None and lon_value is not None
        in_uk = False
        if has_coordinates:
            in_uk = True if uk_geometry is None else is_in_uk(float(lat_value), float(lon_value), uk_geometry)

        rows.append(row)
        row_by_feature_id[row_id] = row

        if has_coordinates and in_uk:
            kept_points.append(
                GeoPoint(
                    feature_id=row_id,
                    lat=float(lat_value),
                    lon=float(lon_value),
                )
            )
        elif not row.get("highway"):
            row["highway"] = np.nan

    logging.info("Eligible plaques with UK coordinates: %s / %s", len(kept_points), len(rows))

    # Fill only missing city values locally before deciding which road rows need work.
    city_lookup_needed = args.rerun in {"city", "all"} and any(
        value_is_missing(row_by_feature_id[point.feature_id].get("city")) for point in kept_points
    )
    if city_lookup_needed:
        built_up_areas = load_built_up_areas(DEFAULT_DATASET)
        for point in kept_points:
            row = row_by_feature_id[point.feature_id]
            if value_is_missing(row.get("city")):
                row["city"] = find_built_up_area(point.lat, point.lon, built_up_areas, buffer_m=100.0)["name"]

    if args.rerun == "city":
        write_csv(output_path, rows, output_fieldnames)
        logging.info("City rerun complete. CSV written to %s", output_path)
        return 0

    if not kept_points:
        write_csv(output_path, rows, output_fieldnames)
        logging.info("No processable points found. CSV written to %s", output_path)
        return 0

    clusters = cluster_points(
        points=kept_points,
        cell_m=max(1.0, float(args.cluster_cell_m)),
        bbox_buffer_m=max(0.0, float(args.bbox_buffer_m)),
    )
    logging.info("Built %s clusters from valid plaque points", len(clusters))

    for c_idx, cluster in enumerate(clusters, start=1):
        pending_points = [
            kept_points[kept_idx]
            for kept_idx in cluster.point_indices
            if (
                value_is_missing(row_by_feature_id[kept_points[kept_idx].feature_id].get("distance_m"))
                or (
                    args.rerun == "all"
                    and value_is_missing(row_by_feature_id[kept_points[kept_idx].feature_id].get("highway"))
                )
            )
        ]
        if not pending_points:
            logging.info(
                "Skipping cluster %s/%s (id=%s) because all rows already exist in CSV",
                c_idx,
                len(clusters),
                cluster.cluster_id,
            )
            continue

        south, west, north, east = cluster.bbox
        logging.info(
            "Cluster %s/%s (id=%s): %s plaques, bbox=(%.6f, %.6f, %.6f, %.6f)",
            c_idx,
            len(clusters),
            cluster.cluster_id,
            len(cluster.point_indices),
            south,
            west,
            north,
            east,
        )

        query = build_overpass_bbox_query(
            south=south,
            west=west,
            north=north,
            east=east,
            timeout_s=args.timeout_s,
        )

        try:
            payload = fetch_overpass_json(
                query=query,
                endpoint=args.overpass_endpoint,
                timeout_s=args.timeout_s,
                max_retries=max(0, int(args.max_retries)),
                retry_wait_s=max(1, int(args.retry_wait_s)),
            )
        except HTTPError as err:
            # Keep the batch running when Overpass remains temporarily unavailable after retries.
            if err.code in {429, 500, 502, 503, 504}:
                logging.warning(
                    "Skipping cluster id=%s due to Overpass HTTP %s after retries.",
                    cluster.cluster_id,
                    err.code,
                )
                write_csv(output_path, rows, output_fieldnames)
                logging.info(
                    "Flushed progress to %s after skipping cluster %s",
                    output_path,
                    cluster.cluster_id,
                )
                continue
            raise
        except (URLError, TimeoutError) as err:
            logging.warning(
                "Skipping cluster id=%s due to network timeout/error: %s",
                cluster.cluster_id,
                err,
            )
            write_csv(output_path, rows, output_fieldnames)
            logging.info(
                "Flushed progress to %s after skipping cluster %s",
                output_path,
                cluster.cluster_id,
            )
            continue

        for kept_idx in cluster.point_indices:
            point = kept_points[kept_idx]
            road = nearest_road_for_point(point.lat, point.lon, payload)
            row = row_by_feature_id[point.feature_id]

            distance_missing = value_is_missing(row.get("distance_m"))
            highway_missing = value_is_missing(row.get("highway"))
            if not distance_missing and not (args.rerun == "all" and highway_missing):
                continue

            if road.distance_m is None:
                if distance_missing:
                    row["distance_m"] = np.nan
                    row["road_status"] = "not_found"
                if highway_missing:
                    row["highway"] = np.nan
            else:
                if distance_missing:
                    row["distance_m"] = road.distance_m
                    row["road_status"] = "found"
                if highway_missing:
                    row["highway"] = road.highway

        write_csv(output_path, rows, output_fieldnames)
        logging.info("Flushed progress to %s after cluster %s", output_path, cluster.cluster_id)

    write_csv(output_path, rows, output_fieldnames)
    logging.info("CSV written to %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        logging.error("Interrupted by user")
        raise SystemExit(130) from exc
