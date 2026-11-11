"""Batch bench-to-road enrichment using clustered bbox Overpass queries.

Pipeline overview:
1. Load bench points from a GeoJSON FeatureCollection.
2. Keep only points inside the UK boundary (optional, enabled by default).
3. Cluster benches spatially (grid-cell connected components).
4. Query Overpass once per cluster bbox for all highway ways.
5. Query Overpass once per cluster bbox for settlement features (city/town/village/hamlet).
6. Compute nearest road segment per bench locally.
7. Determine settlement name/type for each bench.
8. Write results to CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
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
    project_local_meters,
    read_existing_csv,
    setup_logging,
    write_csv,
)


@dataclass(frozen=True)
class SettlementResult:
    """Settlement lookup output for one bench."""

    settlement_name: str | None
    settlement_type: str | None


def parse_args() -> argparse.Namespace:
    """Define CLI arguments and parse command line input."""

    parser = argparse.ArgumentParser(
        description=(
            "Batch-enrich bench points with nearest OSM road details using "
            "clustered bbox Overpass requests."
        )
    )

    parser.add_argument(
        "--input",
        default="data_extraction/data.json",
        help="Path to input GeoJSON FeatureCollection (default: data_extraction/data.json)",
    )
    parser.add_argument(
        "--output",
        default="data_extraction/bench_road_enrichment.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--city-only",
        action="store_true",
        help="Fill missing city values from the local built-up-area dataset, then exit without Overpass requests",
    )

    parser.add_argument(
        "--cluster-cell-m",
        type=float,
        default=300.0,
        help="Grid cell size in meters used to form bench clusters",
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
        "--settlement-fallback-distance-m",
        type=float,
        default=0.0,
        help=(
            "Optional max distance in meters to nearest settlement center when no containing "
            "settlement polygon is found (0 disables fallback)"
        ),
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


def load_geojson_points(input_path: Path) -> List[GeoPoint]:
    """Load bench points from a GeoJSON FeatureCollection."""

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", [])
    points: List[GeoPoint] = []

    for idx, feature in enumerate(features):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Point":
            continue

        coordinates = geometry.get("coordinates", [])
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue

        lon = float(coordinates[0])
        lat = float(coordinates[1])
        feature_id = str(int(feature.get("id", idx + 1)))
        points.append(GeoPoint(feature_id=feature_id, lon=lon, lat=lat))

    return points


def build_overpass_settlement_query(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout_s: int,
) -> str:
    """Construct an Overpass QL query for settlement features in a bbox."""

    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        "(\n"
        f"  way[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        f"  rel[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        f"  node[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        ");\n"
        "out tags geom center;"
    )


def extract_settlement_candidates(overpass_payload: dict):
    """Extract settlement polygons and center points from Overpass payload."""

    allowed = {"city", "town", "village", "hamlet"}

    try:
        from shapely.geometry import Polygon
    except ModuleNotFoundError:
        Polygon = None  # type: ignore[assignment]

    polygons = []
    centers = []

    for element in overpass_payload.get("elements", []):
        tags = element.get("tags", {})
        place_type = tags.get("place")
        settlement_name = tags.get("name")

        if place_type not in allowed or not settlement_name:
            continue

        geometry = element.get("geometry", [])
        if Polygon is not None and len(geometry) >= 4:
            first = geometry[0]
            last = geometry[-1]
            if first.get("lat") == last.get("lat") and first.get("lon") == last.get("lon"):
                coords = [(p["lon"], p["lat"]) for p in geometry if "lat" in p and "lon" in p]
                if len(coords) >= 4:
                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if not polygon.is_empty:
                        polygons.append(
                            {
                                "name": settlement_name,
                                "type": place_type,
                                "polygon": polygon,
                                "area": float(polygon.area),
                            }
                        )

        if element.get("type") == "node" and "lat" in element and "lon" in element:
            centers.append(
                {
                    "name": settlement_name,
                    "type": place_type,
                    "lat": float(element["lat"]),
                    "lon": float(element["lon"]),
                }
            )
        elif "center" in element and "lat" in element["center"] and "lon" in element["center"]:
            centers.append(
                {
                    "name": settlement_name,
                    "type": place_type,
                    "lat": float(element["center"]["lat"]),
                    "lon": float(element["center"]["lon"]),
                }
            )

    return polygons, centers


def settlement_for_point(
    lat: float,
    lon: float,
    settlement_polygons: Sequence[dict],
    settlement_centers: Sequence[dict],
    fallback_distance_m: float,
) -> SettlementResult:
    """Determine settlement name/type for one point."""

    try:
        from shapely.geometry import Point
    except ModuleNotFoundError:
        Point = None  # type: ignore[assignment]

    if Point is not None:
        point = Point(float(lon), float(lat))
        containing = [poly for poly in settlement_polygons if poly["polygon"].covers(point)]
        if containing:
            best = min(containing, key=lambda poly: poly["area"])
            return SettlementResult(
                settlement_name=best["name"],
                settlement_type=best["type"],
            )

    if fallback_distance_m > 0 and settlement_centers:
        nearest = None
        nearest_distance = None
        for center in settlement_centers:
            dx, dy = project_local_meters(center["lat"], center["lon"], lat, lon)
            distance_m = math.sqrt((dx * dx) + (dy * dy))
            if nearest_distance is None or distance_m < nearest_distance:
                nearest_distance = distance_m
                nearest = center

        if nearest is not None and nearest_distance is not None and nearest_distance <= fallback_distance_m:
            return SettlementResult(
                settlement_name=nearest["name"],
                settlement_type=nearest["type"],
            )

    return SettlementResult(settlement_name=None, settlement_type=None)


def bench_fieldnames() -> list[str]:
    """Return stable output columns for bench enrichment CSV."""

    return [
        "feature_id",
        "lat",
        "lon",
        "is_in_uk",
        "cluster_id",
        "distance_m",
        "highway",
        "name",
        "osm_id",
        "settlement_name",
        "settlement_type",
        "city",
        "road_status",
        "settlement_status",
    ]


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        return 1

    logging.info("Loading bench points from %s", input_path)
    all_points = load_geojson_points(input_path)
    logging.info("Loaded %s point features", len(all_points))

    if not all_points:
        logging.error("No valid point features found in input file")
        return 1

    uk_geometry = None
    if not args.skip_uk_filter:
        logging.info("Loading UK boundary geometry for point filtering")
        uk_geometry = load_uk_geometry(Path(args.shapefile_cache_dir))

    existing_rows = read_existing_csv(output_path, key_column="feature_id")
    if existing_rows:
        logging.info("Loaded %s existing rows from %s", len(existing_rows), output_path)

    rows: List[dict] = []
    kept_points: List[GeoPoint] = []
    row_by_feature_id: Dict[str, dict] = {}

    for point in all_points:
        in_uk = True
        if uk_geometry is not None:
            in_uk = is_in_uk(point.lat, point.lon, uk_geometry)

        row = {
            "feature_id": point.feature_id,
            "lat": point.lat,
            "lon": point.lon,
            "is_in_uk": in_uk,
            "cluster_id": None,
            "distance_m": None,
            "highway": None,
            "name": None,
            "osm_id": None,
            "settlement_name": None,
            "settlement_type": None,
            "city": None,
            "road_status": None,
            "settlement_status": None,
        }

        existing_row = existing_rows.get(point.feature_id)
        if existing_row is not None:
            row["cluster_id"] = existing_row.get("cluster_id") or None
            row["distance_m"] = existing_row.get("distance_m") or None
            row["highway"] = existing_row.get("highway") or None
            row["name"] = existing_row.get("name") or None
            row["osm_id"] = existing_row.get("osm_id") or None
            row["settlement_name"] = existing_row.get("settlement_name") or None
            row["settlement_type"] = existing_row.get("settlement_type") or None
            row["city"] = existing_row.get("city") or None
            row["road_status"] = existing_row.get("road_status") or None
            row["settlement_status"] = existing_row.get("settlement_status") or None

        rows.append(row)
        row_by_feature_id[point.feature_id] = row

        if in_uk:
            kept_points.append(point)

    logging.info("Points inside UK: %s / %s", len(kept_points), len(all_points))

    # Fill missing city values from the local built-up polygons before road resumption checks.
    city_lookup_needed = any(
        city_is_missing(row_by_feature_id[point.feature_id].get("city"))
        for point in kept_points
    )
    if city_lookup_needed:
        built_up_areas = load_built_up_areas(DEFAULT_DATASET)
        for point in kept_points:
            row = row_by_feature_id[point.feature_id]
            if city_is_missing(row.get("city")):
                try:
                    row["city"] = find_built_up_area(
                        point.lat,
                        point.lon,
                        built_up_areas,
                        buffer_m=100.0,
                    )["name"]
                except ValueError as err:
                    logging.warning(
                        "Skipping city lookup for bench feature %s: %s",
                        point.feature_id,
                        err,
                    )

    if args.city_only:
        write_csv(output_path, rows, bench_fieldnames())
        logging.info("City-only enrichment complete. CSV written to %s", output_path)
        return 0

    if not kept_points:
        logging.warning("No UK points found. Writing output with empty enrichment fields.")
        write_csv(output_path, rows, bench_fieldnames())
        logging.info("CSV written to %s", output_path)
        return 0

    clusters = cluster_points(
        points=kept_points,
        cell_m=max(1.0, float(args.cluster_cell_m)),
        bbox_buffer_m=max(0.0, float(args.bbox_buffer_m)),
    )
    logging.info("Built %s clusters from UK points", len(clusters))

    # A distance value or explicit no-road status marks road distance enrichment as complete.
    required_values = {"distance_m": set(), "road_status": {"not_found"}}

    for c_idx, cluster in enumerate(clusters, start=1):
        if cluster_is_complete(cluster, kept_points, row_by_feature_id, required_values):
            logging.info(
                "Skipping cluster %s/%s (id=%s) because all rows already exist in CSV",
                c_idx,
                len(clusters),
                cluster.cluster_id,
            )
            continue

        south, west, north, east = cluster.bbox
        logging.info(
            "Cluster %s/%s (id=%s): %s benches, bbox=(%.6f, %.6f, %.6f, %.6f)",
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

        # Skip settlement requests when existing rows only need distance backfill.
        needs_full_rows = any(
            row_by_feature_id[kept_points[point_index].feature_id].get("road_status") not in {"found", "not_found"}
            for point_index in cluster.point_indices
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
                write_csv(output_path, rows, bench_fieldnames())
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
            write_csv(output_path, rows, bench_fieldnames())
            logging.info(
                "Flushed progress to %s after skipping cluster %s",
                output_path,
                cluster.cluster_id,
            )
            continue

        settlement_polygons, settlement_centers = [], []
        if needs_full_rows:
            settlement_query = build_overpass_settlement_query(
                south=south,
                west=west,
                north=north,
                east=east,
                timeout_s=args.timeout_s,
            )
            try:
                settlement_payload = fetch_overpass_json(
                    query=settlement_query,
                    endpoint=args.overpass_endpoint,
                    timeout_s=args.timeout_s,
                    max_retries=max(0, int(args.max_retries)),
                    retry_wait_s=max(1, int(args.retry_wait_s)),
                )
                settlement_polygons, settlement_centers = extract_settlement_candidates(settlement_payload)
            except (HTTPError, URLError, TimeoutError) as err:
                logging.warning("Settlement lookup failed for cluster %s: %s", cluster.cluster_id, err)
                write_csv(output_path, rows, bench_fieldnames())
                continue

        for kept_idx in cluster.point_indices:
            point = kept_points[kept_idx]
            road = nearest_road_for_point(point.lat, point.lon, payload)

            row = row_by_feature_id[point.feature_id]
            previous_status = row.get("road_status")
            if previous_status == "not_found" or not city_is_missing(row.get("distance_m")):
                continue
            row_was_populated = previous_status == "found"
            row["cluster_id"] = cluster.cluster_id
            row["distance_m"] = road.distance_m
            row["road_status"] = "found" if road.distance_m is not None else "not_found"

            # Existing road rows receive only the missing distance, preserving other columns.
            if row_was_populated:
                continue

            row["highway"] = road.highway
            row["name"] = road.name
            row["osm_id"] = road.osm_id

            settlement = settlement_for_point(
                lat=point.lat,
                lon=point.lon,
                settlement_polygons=settlement_polygons,
                settlement_centers=settlement_centers,
                fallback_distance_m=max(0.0, float(args.settlement_fallback_distance_m)),
            )

            row["settlement_name"] = settlement.settlement_name
            row["settlement_type"] = settlement.settlement_type
            row["settlement_status"] = (
                "found" if settlement.settlement_name and settlement.settlement_type else "not_found"
            )

        write_csv(output_path, rows, bench_fieldnames())
        logging.info("Flushed progress to %s after cluster %s", output_path, cluster.cluster_id)

    write_csv(output_path, rows, bench_fieldnames())
    logging.info("CSV written to %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        logging.error("Interrupted by user")
        raise SystemExit(130) from exc
