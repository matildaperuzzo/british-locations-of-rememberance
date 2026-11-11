"""Identify the built-up area containing a latitude/longitude point."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().with_name("os_open_built_up_areas.gpkg")
BUILT_UP_LAYER = "os_open_built_up_areas"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the local built-up-area lookup."""

    parser = argparse.ArgumentParser(
        description="Find the OS Open Built Up Area containing a coordinate."
    )
    parser.add_argument("latitude", type=float, help="Point latitude in decimal degrees")
    parser.add_argument("longitude", type=float, help="Point longitude in decimal degrees")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the OS Open Built Up Areas GeoPackage",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=100.0,
        help="Distance in meters to buffer built-up polygons for edge locations",
    )
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", type=Path, help="Optional path for a JSON result")
    return parser.parse_args()


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Reject coordinates outside valid WGS84 latitude and longitude ranges."""

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180")


def city_is_missing(value) -> bool:
    """Return whether a CSV city cell is blank or a serialized null value."""

    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none"}


def load_built_up_areas(dataset_path: Path):
    """Read the named built-up-area polygons from the local GeoPackage."""

    try:
        import geopandas as gpd
    except ModuleNotFoundError as error:
        raise RuntimeError("GeoPandas is required. Install it with: pip install geopandas") from error

    if not dataset_path.exists():
        raise FileNotFoundError(f"Built-up-area GeoPackage not found: {dataset_path}")
    return gpd.read_file(
        dataset_path,
        layer=BUILT_UP_LAYER,
        columns=["gsscode", "name1_text", "areahectares", "geometry_area_m"],
    )


def find_built_up_area(latitude: float, longitude: float, built_up_areas, buffer_m: float = 100.0) -> dict:
    """Find a containing built-up polygon or the closest polygon within the buffer."""

    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.ops import transform

    validate_coordinates(latitude, longitude)
    if built_up_areas.crs is None:
        raise ValueError("The built-up-area layer has no coordinate reference system")

    # Transform WGS84 coordinates into the GeoPackage's projected coordinate system.
    transformer = Transformer.from_crs("EPSG:4326", built_up_areas.crs, always_xy=True)
    point = transform(transformer.transform, Point(float(longitude), float(latitude)))

    # Search nearby polygon candidates, including any that fall within the edge buffer.
    buffer_m = max(0.0, float(buffer_m))
    search_geometry = point.buffer(buffer_m) if buffer_m else point
    candidate_indices = built_up_areas.sindex.query(search_geometry, predicate="intersects")
    matches = built_up_areas.iloc[candidate_indices]
    matches = matches.assign(distance_to_area_m=matches.geometry.distance(point))
    matches = matches[matches.distance_to_area_m <= buffer_m]
    if matches.empty:
        return {
            "name": "no settlement",
            "gsscode": None,
            "areahectares": None,
            "match": "outside_built_up_area",
        }

    # Prefer an interior match, then the closest polygon, then the smaller overlapping area.
    match = matches.sort_values(["distance_to_area_m", "geometry_area_m"]).iloc[0]
    is_inside = bool(match.geometry.covers(point))
    return {
        "name": match["name1_text"],
        "gsscode": match["gsscode"],
        "areahectares": float(match["areahectares"]),
        "match": "inside_built_up_area" if is_inside else "within_buffer",
        "distance_to_area_m": round(float(match["distance_to_area_m"]), 2),
    }


def main() -> int:
    """Classify the coordinate against the local built-up-area polygons."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    try:
        validate_coordinates(args.latitude, args.longitude)
        built_up_areas = load_built_up_areas(args.dataset)
        settlement = find_built_up_area(args.latitude, args.longitude, built_up_areas, args.buffer_m)
    except (ValueError, OSError, RuntimeError) as error:
        logging.error("Built-up-area lookup failed: %s", error)
        return 1

    output = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "settlement": settlement,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
