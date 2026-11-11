"""Add nearest OpenStreetMap road details to the monument export."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from identify_settlement import DEFAULT_DATASET, city_is_missing, find_built_up_area, load_built_up_areas

from enrichment_common import (
    GeoPoint,
    build_overpass_bbox_query,
    cluster_is_complete,
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
    """Normalize a source header so BOMs and whitespace do not cause mismatches."""

    # Overpass exports can include a UTF-8 BOM in their first header.
    return name.replace("\ufeff", "").strip().lower()


def parse_args() -> argparse.Namespace:
    """Define command-line options for the monument enrichment run."""

    # Keep the source and enriched files separate so the raw download remains recoverable.
    parser = argparse.ArgumentParser(
        description="Add nearest OSM highway type and distance to a monument TSV export."
    )
    parser.add_argument("--input", default="data_extraction/monuments.csv")
    parser.add_argument("--output", default="data_extraction/monuments_enriched.csv")
    parser.add_argument(
        "--city-only",
        action="store_true",
        help="Fill missing city values from the local built-up-area dataset, then exit without Overpass requests",
    )
    parser.add_argument("--cluster-cell-m", type=float, default=300.0)
    parser.add_argument("--bbox-buffer-m", type=float, default=120.0)
    parser.add_argument("--overpass-endpoint", default="https://overpass-api.de/api/interpreter")
    parser.add_argument("--timeout-s", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-wait-s", type=int, default=30)
    parser.add_argument("--skip-uk-filter", action="store_true")
    parser.add_argument("--shapefile-cache-dir", default="data_extraction/shapefiles")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    """Convert a coordinate cell to a float, returning None for invalid values."""

    # Empty or malformed coordinates cannot be sent to the geographic lookup.
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def main() -> int:
    """Read monuments, enrich eligible points, and write a tab-separated result."""

    # Parse options and configure progress logging before doing network work.
    args = parse_args()
    setup_logging(args.log_level)
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Fail clearly when the source download is missing.
    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        return 1

    # Load the optional UK boundary once, rather than once per monument.
    uk_geometry = None
    if not args.skip_uk_filter:
        uk_geometry = load_uk_geometry(Path(args.shapefile_cache_dir))

    # Overpass CSV exports are tab-separated even when the file has a .csv suffix.
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        input_fieldnames = reader.fieldnames or []
        source_rows = list(reader)

    # Match the Overpass headers while retaining their original spelling in output.
    normalized = {normalize_column_name(name): name for name in input_fieldnames}
    required = {"id": "@id", "lat": "@lat", "lon": "@lon"}
    missing = [name for name in required.values() if normalize_column_name(name) not in normalized]
    if missing:
        logging.error("Input file is missing required columns: %s", ", ".join(missing))
        return 1

    id_column = normalized[normalize_column_name(required["id"])]
    lat_column = normalized[normalize_column_name(required["lat"])]
    lon_column = normalized[normalize_column_name(required["lon"])]
    output_fields = list(input_fieldnames)
    # Add the requested enrichment columns once, preserving all source columns.
    for column in ("highway", "dist"):
        if column not in output_fields:
            output_fields.append(column)
    if "city" not in output_fields:
        output_fields.append("city")
    if "road_status" not in output_fields:
        output_fields.append("road_status")

    # Load prior progress so interrupted runs can resume without repeating completed clusters.
    existing_rows = read_existing_csv(output_path, key_column=id_column)
    rows: List[dict] = []
    kept_points: List[GeoPoint] = []
    row_by_id: Dict[str, dict] = {}

    for index, source_row in enumerate(source_rows, start=1):
        # Fall back to the source row number if an OSM id is absent.
        feature_id = (source_row.get(id_column) or "").strip() or str(index)
        lat = parse_float(source_row.get(lat_column))
        lon = parse_float(source_row.get(lon_column))
        row = dict(source_row)
        row.setdefault("highway", "")
        row.setdefault("dist", "")
        row.setdefault("city", "")
        row.setdefault("road_status", "")

        # Restore completed values from a previous output file when available.
        previous = existing_rows.get(feature_id)
        if previous:
            row["highway"] = previous.get("highway", "")
            row["dist"] = previous.get("dist", "")
            row["city"] = previous.get("city", "")
            row["road_status"] = previous.get("road_status", "")

        rows.append(row)
        row_by_id[feature_id] = row
        if lat is not None and lon is not None:
            in_uk = args.skip_uk_filter or is_in_uk(lat, lon, uk_geometry)
            if in_uk:
                kept_points.append(GeoPoint(feature_id=feature_id, lat=lat, lon=lon))

    logging.info("Eligible monuments with UK coordinates: %s / %s", len(kept_points), len(rows))

    # Fill missing city values locally before deciding which road rows need work.
    city_lookup_needed = any(
        city_is_missing(row_by_id[point.feature_id].get("city"))
        for point in kept_points
    )
    if city_lookup_needed:
        built_up_areas = load_built_up_areas(DEFAULT_DATASET)
        for point in kept_points:
            row = row_by_id[point.feature_id]
            if city_is_missing(row.get("city")):
                try:
                    row["city"] = find_built_up_area(
                        point.lat,
                        point.lon,
                        built_up_areas,
                        buffer_m=100.0,
                    )["name"]
                except ValueError as error:
                    logging.warning(
                        "Skipping city lookup for monument %s: %s",
                        point.feature_id,
                        error,
                    )
    if args.city_only:
        write_csv(output_path, rows, output_fields)
        logging.info("City-only enrichment complete. CSV written to %s", output_path)
        return 0
    if not kept_points:
        write_csv(output_path, rows, output_fields)
        return 0

    # Cluster nearby monuments so each Overpass request covers many points.
    clusters = cluster_points(kept_points, max(1.0, args.cluster_cell_m), max(0.0, args.bbox_buffer_m))
    for cluster in clusters:
        # A distance value or explicit no-road status marks road distance enrichment as complete.
        if cluster_is_complete(cluster, kept_points, row_by_id, {"dist": set(), "road_status": {"not_found"}}):
            continue
        south, west, north, east = cluster.bbox
        query = build_overpass_bbox_query(south, west, north, east, args.timeout_s)
        try:
            payload = fetch_overpass_json(query, args.overpass_endpoint, args.timeout_s, args.max_retries, args.retry_wait_s)
        except (HTTPError, URLError, TimeoutError) as error:
            # Flush completed work before moving on when an Overpass request fails.
            logging.warning("Skipping cluster %s: %s", cluster.cluster_id, error)
            write_csv(output_path, rows, output_fields)
            continue

        for point_index in cluster.point_indices:
            point = kept_points[point_index]
            road = nearest_road_for_point(point.lat, point.lon, payload)
            row = row_by_id[point.feature_id]

            # Preserve the prior road classification when this row only needs its distance.
            if not city_is_missing(row.get("dist")):
                continue
            highway_was_populated = not city_is_missing(row.get("highway"))
            if road.distance_m is None:
                row["dist"] = ""
                row["road_status"] = "not_found"
                if not highway_was_populated:
                    row["highway"] = ""
            else:
                row["dist"] = road.distance_m
                row["road_status"] = "found"
                if not highway_was_populated:
                    row["highway"] = road.highway or ""

        # Flush after every cluster to make long UK-wide runs restartable.
        write_csv(output_path, rows, output_fields)
        logging.info("Completed cluster %s/%s", cluster.cluster_id, len(clusters))

    # Write the final complete snapshot using the original tabular column order.
    write_csv(output_path, rows, output_fields)
    logging.info("CSV written to %s", output_path)
    return 0


if __name__ == "__main__":
    # Convert Ctrl+C into the conventional shell exit code.
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        logging.error("Interrupted by user")
        raise SystemExit(130) from exc
