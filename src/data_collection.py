"""
data_collection.py
-------------------
Fetches all raw data for the Bay Area Transit Equity project:
  1. ACS 5-Year demographic/commute data at the census-tract level
  2. GTFS static feeds for BART, AC Transit, and SFMTA
  3. Census tract boundary shapefiles (TIGER/Line)

Usage:
    python src/data_collection.py              # run everything
    python src/data_collection.py --census     # census data only
    python src/data_collection.py --gtfs       # GTFS feeds only
    python src/data_collection.py --shapefiles # shapefiles only
"""

import os
import sys
import zipfile
import argparse
import io
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# Bay Area 9-county FIPS codes (California FIPS = 06)
BAY_AREA_COUNTIES = {
    "001": "Alameda",
    "013": "Contra Costa",
    "041": "Marin",
    "055": "Napa",
    "075": "San Francisco",
    "081": "San Mateo",
    "085": "Santa Clara",
    "095": "Solano",
    "097": "Sonoma",
}

STATE_FIPS = "06"  # California

# ACS 5-Year variables to pull
# Commuting, income, vehicle access, population, race/ethnicity
ACS_VARIABLES = {
    # Population
    "B01003_001E": "total_population",
    # Commuting - means of transportation
    "B08301_001E": "total_commuters",
    "B08301_010E": "public_transit_commuters",
    "B08301_003E": "drove_alone",
    "B08301_004E": "carpooled",
    "B08301_019E": "worked_from_home",
    # Commuting - travel time
    "B08303_001E": "travel_time_total",
    "B08303_012E": "commute_60_to_89_min",
    "B08303_013E": "commute_90_plus_min",
    # Income
    "B19013_001E": "median_household_income",
    # Vehicle access
    "B08201_001E": "total_households_vehicles",
    "B08201_002E": "zero_vehicle_households",
    # Race / ethnicity (for equity analysis)
    "B03002_001E": "total_pop_race",
    "B03002_003E": "white_non_hispanic",
    "B03002_004E": "black",
    "B03002_006E": "asian",
    "B03002_012E": "hispanic_latino",
}

ACS_YEAR = 2022  # Most recent stable 5-year ACS

# GTFS feed URLs (primary sources — direct from agencies)
GTFS_FEEDS = {
    "bart": "https://www.bart.gov/dev/schedules/google_transit.zip",
    "actransit": "https://api.actransit.org/transit/gtfs/download",
    "sfmta": "https://gtfs.sfmta.com/transitdata/google_transit.zip",
}

# 511.org fallback — requires a free 511 API token (https://511.org/open-data/token)
# Operator IDs: BA=BART, AC=AC Transit, SF=SFMTA
# URL pattern: http://api.511.org/transit/datafeeds?api_key={TOKEN}&operator_id={ID}
GTFS_511_OPERATOR_IDS = {
    "bart": "BA",
    "actransit": "AC",
    "sfmta": "SF",
}

# TIGER/Line shapefile for CA census tracts
TIGER_TRACTS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_06_tract.zip"
)


# ---------------------------------------------------------------------------
# Census ACS Data Collection
# ---------------------------------------------------------------------------

def fetch_census_data() -> pd.DataFrame:
    """
    Pull ACS 5-year estimates at tract level for all 9 Bay Area counties.
    Uses the Census Bureau REST API directly (no library dependency issues).
    """
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  CENSUS_API_KEY not set!")
        print("   1. Get a free key at: https://api.census.gov/data/key_signup.html")
        print("   2. Copy .env.example to .env and paste your key")
        print("   3. Re-run this script\n")
        sys.exit(1)

    variable_codes = list(ACS_VARIABLES.keys())
    cols = ",".join(["NAME"] + variable_codes)

    base_url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

    all_frames = []

    for county_fips, county_name in BAY_AREA_COUNTIES.items():
        print(f"  Fetching tracts for {county_name} County ({county_fips})...")
        url = (
            f"{base_url}?get={cols}"
            f"&for=tract:*"
            f"&in=state:{STATE_FIPS}%20county:{county_fips}"
            f"&key={api_key}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df["county_name"] = county_name
        all_frames.append(df)

    result = pd.concat(all_frames, ignore_index=True)

    # Build GEOID (state + county + tract) for joining with shapefiles
    result["GEOID"] = result["state"] + result["county"] + result["tract"]

    # Rename coded columns to readable names
    result.rename(columns=ACS_VARIABLES, inplace=True)

    # Convert numeric columns
    numeric_cols = list(ACS_VARIABLES.values())
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def save_census_data(df: pd.DataFrame) -> Path:
    """Save census data to CSV."""
    outdir = DATA_RAW / "census"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "bay_area_acs_tracts.csv"
    df.to_csv(outpath, index=False)
    return outpath


# ---------------------------------------------------------------------------
# GTFS Feed Downloads
# ---------------------------------------------------------------------------

def download_gtfs_feeds(agencies: list[str] | None = None):
    """
    Download and extract GTFS static feeds for Bay Area transit agencies.
    Pass a list of agency keys to limit downloads, or None for all.
    """
    targets = {k: v for k, v in GTFS_FEEDS.items() if agencies is None or k in agencies}

    for agency, url in targets.items():
        dest = DATA_RAW / "gtfs" / agency
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded
        if (dest / "stops.txt").exists():
            print(f"  {agency}: already downloaded, skipping (delete folder to re-download)")
            continue

        print(f"  Downloading {agency} GTFS from {url}...")
        try:
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(dest)

            # Verify essential files exist
            essential = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
            found = [f for f in essential if (dest / f).exists()]
            print(f"    ✓ Extracted {len(found)}/{len(essential)} essential files")

        except requests.exceptions.RequestException as e:
            print(f"    ✗ Failed to download {agency}: {e}")
            print(f"      You can manually download from: {url}")
            print(f"      Extract contents to: {dest}")
        except zipfile.BadZipFile:
            print(f"    ✗ Downloaded file for {agency} is not a valid ZIP")
            print(f"      Try downloading manually from: {url}")


# ---------------------------------------------------------------------------
# Shapefile Downloads
# ---------------------------------------------------------------------------

def download_shapefiles():
    """Download TIGER/Line census tract boundaries for California."""
    dest = DATA_RAW / "shapefiles"
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "tl_2022_06_tract.zip"
    shp_path = dest / "tl_2022_06_tract.shp"

    if shp_path.exists():
        print("  Census tract shapefile already downloaded, skipping")
        return

    print(f"  Downloading CA census tract boundaries...")
    try:
        resp = requests.get(TIGER_TRACTS_URL, timeout=120)
        resp.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)

        print(f"    ✓ Extracted to {dest}")

    except requests.exceptions.RequestException as e:
        print(f"    ✗ Failed: {e}")
        print(f"      Download manually from: {TIGER_TRACTS_URL}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect raw data for transit equity project")
    parser.add_argument("--census", action="store_true", help="Fetch Census ACS data only")
    parser.add_argument("--gtfs", action="store_true", help="Download GTFS feeds only")
    parser.add_argument("--shapefiles", action="store_true", help="Download shapefiles only")
    args = parser.parse_args()

    # If no flags specified, run everything
    run_all = not (args.census or args.gtfs or args.shapefiles)

    print("=" * 60)
    print("Bay Area Transit Equity — Data Collection")
    print("=" * 60)

    if run_all or args.census:
        print("\n📊 Fetching Census ACS Data...")
        df = fetch_census_data()
        path = save_census_data(df)
        print(f"   ✓ Saved {len(df)} tracts across {df['county_name'].nunique()} counties")
        print(f"   → {path}\n")

    if run_all or args.gtfs:
        print("🚌 Downloading GTFS Feeds...")
        download_gtfs_feeds()
        print()

    if run_all or args.shapefiles:
        print("🗺️  Downloading Census Tract Boundaries...")
        download_shapefiles()
        print()

    print("=" * 60)
    print("Done! Next step: run src/preprocessing.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
