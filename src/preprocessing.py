"""
preprocessing.py
-----------------
Cleans, merges, and prepares all raw data into analysis-ready datasets:
  1. Merge Census ACS data with tract shapefiles (GeoDataFrame)
  2. Parse GTFS stops into GeoDataFrames with agency labels
  3. Parse GTFS stop_times for service frequency analysis
  4. Spatial join: count stops per tract, calculate coverage metrics
  5. Export processed datasets for QGIS, modeling, and dashboard

Usage:
    python src/preprocessing.py
"""

import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"

# Bay Area county FIPS codes
BAY_AREA_FIPS = ["001", "013", "041", "055", "075", "081", "085", "095", "097"]
STATE_FIPS = "06"


# ---------------------------------------------------------------------------
# 1. Census + Shapefile Merge
# ---------------------------------------------------------------------------

def load_and_merge_census_tracts() -> gpd.GeoDataFrame:
    """
    Load Census ACS CSV and TIGER/Line shapefile, merge on GEOID,
    filter to Bay Area counties only.
    """
    print("  Loading Census ACS data...")
    census_path = DATA_RAW / "census" / "bay_area_acs_tracts.csv"
    if not census_path.exists():
        raise FileNotFoundError(
            f"Census data not found at {census_path}. Run data_collection.py first."
        )
    census_df = pd.read_csv(census_path, dtype={"GEOID": str})

    print("  Loading TIGER/Line tract boundaries...")
    shp_path = DATA_RAW / "shapefiles" / "tl_2022_06_tract.shp"
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found at {shp_path}. Run data_collection.py first."
        )
    tracts_gdf = gpd.read_file(shp_path)

    # Filter to Bay Area counties
    tracts_gdf = tracts_gdf[
        tracts_gdf["COUNTYFP"].isin(BAY_AREA_FIPS)
    ].copy()

    print(f"  Filtered to {len(tracts_gdf)} Bay Area census tracts")

    # Merge census attributes onto geometry
    merged = tracts_gdf.merge(
        census_df,
        left_on="GEOID",
        right_on="GEOID",
        how="left",
        suffixes=("_shp", "_acs"),
    )

    # Calculate area in sq km (project to CA Albers for accurate area)
    merged_projected = merged.to_crs(epsg=3310)  # CA Albers
    merged["area_sq_km"] = merged_projected.geometry.area / 1e6

    # Calculate derived demographic metrics
    merged["transit_mode_share"] = (
        merged["public_transit_commuters"] / merged["total_commuters"]
    ).replace([np.inf, -np.inf], np.nan)

    merged["zero_vehicle_pct"] = (
        merged["zero_vehicle_households"] / merged["total_households_vehicles"]
    ).replace([np.inf, -np.inf], np.nan)

    merged["long_commute_pct"] = (
        (merged["commute_60_to_89_min"].fillna(0) + merged["commute_90_plus_min"].fillna(0))
        / merged["total_commuters"]
    ).replace([np.inf, -np.inf], np.nan)

    merged["pop_density"] = (
        merged["total_population"] / merged["area_sq_km"]
    ).replace([np.inf, -np.inf], np.nan)

    # Ensure CRS is WGS84 for mapping
    merged = merged.to_crs(epsg=4326)

    return merged


# ---------------------------------------------------------------------------
# 2. GTFS Stop Parsing
# ---------------------------------------------------------------------------

def load_gtfs_stops(agency: str) -> gpd.GeoDataFrame:
    """Load stops.txt from a GTFS feed into a GeoDataFrame."""
    stops_path = DATA_RAW / "gtfs" / agency / "stops.txt"
    if not stops_path.exists():
        print(f"    ⚠️  No stops.txt found for {agency}, skipping")
        return gpd.GeoDataFrame()

    stops = pd.read_csv(stops_path)

    # Some feeds have parent stations — filter to actual stops
    if "location_type" in stops.columns:
        stops = stops[stops["location_type"].isin([0, np.nan, ""])].copy()

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(stops["stop_lon"], stops["stop_lat"])
    ]
    gdf = gpd.GeoDataFrame(stops, geometry=geometry, crs="EPSG:4326")
    gdf["agency"] = agency
    return gdf


def load_all_stops() -> gpd.GeoDataFrame:
    """Load and combine stops from all available GTFS feeds."""
    agencies = ["bart", "actransit", "sfmta"]
    frames = []
    for agency in agencies:
        print(f"  Loading {agency} stops...")
        gdf = load_gtfs_stops(agency)
        if len(gdf) > 0:
            print(f"    ✓ {len(gdf)} stops")
            frames.append(gdf)

    if not frames:
        raise FileNotFoundError("No GTFS stops found. Run data_collection.py first.")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Service Frequency Analysis
# ---------------------------------------------------------------------------

def calculate_stop_frequency(agency: str) -> pd.DataFrame:
    """
    Parse stop_times.txt to calculate average weekday departures per stop.
    Returns a DataFrame with stop_id and avg_daily_departures.
    """
    stop_times_path = DATA_RAW / "gtfs" / agency / "stop_times.txt"
    trips_path = DATA_RAW / "gtfs" / agency / "trips.txt"
    calendar_path = DATA_RAW / "gtfs" / agency / "calendar.txt"

    if not stop_times_path.exists():
        return pd.DataFrame(columns=["stop_id", "avg_daily_departures", "agency"])

    stop_times = pd.read_csv(stop_times_path, dtype={"stop_id": str, "trip_id": str})
    trips = pd.read_csv(trips_path, dtype={"trip_id": str, "service_id": str})

    # Try to filter to weekday services
    weekday_service_ids = None
    if calendar_path.exists():
        calendar = pd.read_csv(calendar_path, dtype={"service_id": str})
        weekday_services = calendar[
            (calendar["monday"] == 1)
            | (calendar["tuesday"] == 1)
            | (calendar["wednesday"] == 1)
            | (calendar["thursday"] == 1)
            | (calendar["friday"] == 1)
        ]
        weekday_service_ids = set(weekday_services["service_id"])

    # Join stop_times with trips to get service_id
    merged = stop_times.merge(trips[["trip_id", "service_id"]], on="trip_id", how="left")

    # Filter to weekday trips if we have calendar data
    if weekday_service_ids:
        merged = merged[merged["service_id"].isin(weekday_service_ids)]

    # Count departures per stop
    freq = (
        merged.groupby("stop_id")
        .size()
        .reset_index(name="total_weekday_departures")
    )

    # Average over weekdays (5 days)
    freq["avg_daily_departures"] = freq["total_weekday_departures"] / 5
    freq["agency"] = agency

    return freq[["stop_id", "avg_daily_departures", "agency"]]


def calculate_all_frequencies() -> pd.DataFrame:
    """Calculate service frequency for all agencies."""
    agencies = ["bart", "actransit", "sfmta"]
    frames = []
    for agency in agencies:
        print(f"  Calculating frequency for {agency}...")
        freq = calculate_stop_frequency(agency)
        if len(freq) > 0:
            print(f"    ✓ {len(freq)} stops with frequency data")
            frames.append(freq)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Spatial Join & Coverage Metrics
# ---------------------------------------------------------------------------

def spatial_join_stops_to_tracts(
    tracts_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatial join: count transit stops within each census tract,
    and calculate buffer-based coverage metrics.
    """
    print("  Performing spatial join (stops → tracts)...")

    # Project to CA Albers (meters) for accurate distance/buffer calculations
    tracts_proj = tracts_gdf.to_crs(epsg=3310)
    stops_proj = stops_gdf.to_crs(epsg=3310)

    # --- Stop count per tract ---
    joined = gpd.sjoin(stops_proj, tracts_proj, how="inner", predicate="within")
    stop_counts = (
        joined.groupby("GEOID")
        .agg(
            stop_count=("stop_id", "count"),
            agencies_serving=("agency", "nunique"),
        )
        .reset_index()
    )

    # --- Buffer coverage (800m = ~10-min walk) ---
    print("  Calculating 800m buffer coverage per tract...")
    stop_buffers = stops_proj.copy()
    stop_buffers["geometry"] = stop_buffers.geometry.buffer(800)
    combined_buffer = stop_buffers.union_all()

    # Calculate % of each tract area covered by stop buffers
    coverage = []
    for idx, row in tracts_proj.iterrows():
        tract_area = row.geometry.area
        if tract_area == 0:
            coverage.append({"GEOID": row["GEOID"], "pct_area_covered_800m": 0})
            continue
        intersection = row.geometry.intersection(combined_buffer)
        pct = (intersection.area / tract_area) * 100
        coverage.append({"GEOID": row["GEOID"], "pct_area_covered_800m": min(pct, 100)})

    coverage_df = pd.DataFrame(coverage)

    # --- Nearest BART station distance ---
    print("  Calculating distance to nearest BART station...")
    bart_stops = stops_proj[stops_proj["agency"] == "bart"]

    if len(bart_stops) > 0:
        from shapely.ops import nearest_points

        distances = []
        for idx, tract in tracts_proj.iterrows():
            centroid = tract.geometry.centroid
            min_dist = bart_stops.geometry.distance(centroid).min()
            distances.append({
                "GEOID": tract["GEOID"],
                "dist_to_nearest_bart_m": min_dist,
            })
        dist_df = pd.DataFrame(distances)
    else:
        dist_df = pd.DataFrame(columns=["GEOID", "dist_to_nearest_bart_m"])

    # --- Merge everything back onto tracts ---
    result = tracts_gdf.merge(stop_counts, on="GEOID", how="left")
    result = result.merge(coverage_df, on="GEOID", how="left")
    if len(dist_df) > 0:
        result = result.merge(dist_df, on="GEOID", how="left")

    # Fill NaN (tracts with no stops)
    result["stop_count"] = result["stop_count"].fillna(0).astype(int)
    result["agencies_serving"] = result["agencies_serving"].fillna(0).astype(int)
    result["pct_area_covered_800m"] = result["pct_area_covered_800m"].fillna(0)

    # Stop density
    result["stop_density_per_sq_km"] = (
        result["stop_count"] / result["area_sq_km"]
    ).replace([np.inf, -np.inf], 0)

    return result


# ---------------------------------------------------------------------------
# 5. Export
# ---------------------------------------------------------------------------

def export_processed_data(
    tracts_gdf: gpd.GeoDataFrame,
    stops_gdf: gpd.GeoDataFrame,
    freq_df: pd.DataFrame,
):
    """Save all processed datasets."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Tract features (main analysis table)
    tract_csv = tracts_gdf.drop(columns=["geometry"]).copy()
    tract_csv.to_csv(DATA_PROCESSED / "tract_features.csv", index=False)
    print(f"  ✓ tract_features.csv ({len(tract_csv)} rows)")

    # Tract features as GeoJSON (for QGIS and Folium)
    tracts_gdf.to_file(DATA_PROCESSED / "tract_features.geojson", driver="GeoJSON")
    print(f"  ✓ tract_features.geojson")

    # All stops with frequency
    if len(freq_df) > 0:
        stops_with_freq = stops_gdf.merge(
            freq_df[["stop_id", "avg_daily_departures"]],
            on="stop_id",
            how="left",
        )
    else:
        stops_with_freq = stops_gdf.copy()
        stops_with_freq["avg_daily_departures"] = np.nan

    stops_with_freq.to_file(DATA_PROCESSED / "all_stops.geojson", driver="GeoJSON")
    print(f"  ✓ all_stops.geojson ({len(stops_with_freq)} stops)")

    # Flat CSV for Power BI / Tableau
    dashboard_df = tract_csv.copy()
    dashboard_cols = [
        "GEOID", "county_name", "total_population", "pop_density",
        "median_household_income", "total_commuters", "public_transit_commuters",
        "transit_mode_share", "zero_vehicle_pct", "long_commute_pct",
        "stop_count", "stop_density_per_sq_km", "pct_area_covered_800m",
        "agencies_serving",
    ]
    # Only keep columns that exist
    dashboard_cols = [c for c in dashboard_cols if c in dashboard_df.columns]
    if "dist_to_nearest_bart_m" in dashboard_df.columns:
        dashboard_cols.append("dist_to_nearest_bart_m")

    dashboard_df[dashboard_cols].to_csv(
        DATA_PROCESSED / "dashboard_export.csv", index=False
    )
    print(f"  ✓ dashboard_export.csv (ready for Power BI / Tableau)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Bay Area Transit Equity — Preprocessing")
    print("=" * 60)

    print("\n📊 Step 1: Merging Census data with tract boundaries...")
    tracts_gdf = load_and_merge_census_tracts()

    print("\n🚏 Step 2: Loading GTFS stops...")
    stops_gdf = load_all_stops()

    print("\n⏱️  Step 3: Calculating service frequency...")
    freq_df = calculate_all_frequencies()

    print("\n🔗 Step 4: Spatial joins & coverage metrics...")
    tracts_gdf = spatial_join_stops_to_tracts(tracts_gdf, stops_gdf)

    # Merge frequency stats at tract level
    if len(freq_df) > 0:
        stops_with_freq = stops_gdf.merge(
            freq_df[["stop_id", "avg_daily_departures"]], on="stop_id", how="left"
        )
        freq_by_tract = gpd.sjoin(
            stops_with_freq.to_crs(epsg=3310),
            tracts_gdf[["GEOID", "geometry"]].to_crs(epsg=3310),
            how="inner",
            predicate="within",
        )
        avg_freq = (
            freq_by_tract.groupby("GEOID")["avg_daily_departures"]
            .mean()
            .reset_index()
            .rename(columns={"avg_daily_departures": "avg_stop_departures_per_day"})
        )
        tracts_gdf = tracts_gdf.merge(avg_freq, on="GEOID", how="left")
        tracts_gdf["avg_stop_departures_per_day"] = tracts_gdf[
            "avg_stop_departures_per_day"
        ].fillna(0)

    print("\n💾 Step 5: Exporting processed datasets...")
    export_processed_data(tracts_gdf, stops_gdf, freq_df)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"  Tracts: {len(tracts_gdf)}")
    print(f"  Stops:  {len(stops_gdf)}")
    print(f"  Output: {DATA_PROCESSED}")
    print("\nNext steps:")
    print("  1. Open data/processed/tract_features.geojson in QGIS")
    print("  2. Run notebooks/02_eda_and_mapping.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
