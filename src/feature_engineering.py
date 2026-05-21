"""
feature_engineering.py  (v3)
------------------------------
Key changes from v2:
- Removed county dummies (they dominated predictions and caused CV failure)
- Added tract centroid lat/lon as continuous spatial features
- Added interaction terms between transit access and demographics
- Target remains: long_commute_pct (Census ACS, independent of transit features)

Usage:
    python src/feature_engineering.py
"""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

TARGET = "commute_burden_score"
GROUP_COL = "county_name"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ------------------------------------------------------------------
    # Target: long_commute_pct — pure Census measurement, independent
    # of our transit features (no circularity)
    # ------------------------------------------------------------------
    out[TARGET] = (
        out["long_commute_pct"]
        .fillna(out["long_commute_pct"].median())
        .clip(0, 1)
        .round(4)
    )

    # ------------------------------------------------------------------
    # Spatial features — continuous lat/lon instead of county dummies
    # Captures geographic gradient without hardcoding county identity
    # ------------------------------------------------------------------
    if "INTPTLAT" in out.columns and "INTPTLON" in out.columns:
        out["centroid_lat"] = pd.to_numeric(out["INTPTLAT"], errors="coerce")
        out["centroid_lon"] = pd.to_numeric(out["INTPTLON"], errors="coerce")
    elif "INTPTLAT10" in out.columns:
        out["centroid_lat"] = pd.to_numeric(out["INTPTLAT10"], errors="coerce")
        out["centroid_lon"] = pd.to_numeric(out["INTPTLON10"], errors="coerce")
    else:
        # Fall back: approximate from county medians — will be filled below
        out["centroid_lat"] = np.nan
        out["centroid_lon"] = np.nan

    # ------------------------------------------------------------------
    # Transit infrastructure features
    # ------------------------------------------------------------------
    out["log_dist_to_bart_m"] = np.log1p(
        out["dist_to_nearest_bart_m"].fillna(out["dist_to_nearest_bart_m"].median())
    )

    coverage_p30 = out["pct_area_covered_800m"].quantile(0.30)
    freq_p30     = out["avg_stop_departures_per_day"].quantile(0.30)
    out["is_transit_desert"] = (
        (out["pct_area_covered_800m"]       <= coverage_p30) &
        (out["avg_stop_departures_per_day"]  <= freq_p30)
    ).astype(int)

    out["transit_access_index"] = (
        out["pct_area_covered_800m"].fillna(0)          / 100 * 0.40
        + out["stop_density_per_sq_km"].fillna(0).clip(upper=10) / 10  * 0.30
        + out["avg_stop_departures_per_day"].fillna(0).clip(upper=200) / 200 * 0.30
    )

    # ------------------------------------------------------------------
    # Demographic features
    # ------------------------------------------------------------------
    out["log_median_income"] = np.log1p(
        out["median_household_income"].fillna(out["median_household_income"].median())
    )

    out["income_quintile"] = pd.qcut(
        out["median_household_income"].fillna(out["median_household_income"].median()),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    out["transit_need_gap"] = (
        out["zero_vehicle_pct"].fillna(0)
        * (1 - out["pct_area_covered_800m"].fillna(0) / 100)
    )

    out["car_dependency_index"] = (
        out["zero_vehicle_pct"].fillna(0) * 0.5
        + (1 - out["transit_access_index"]) * 0.5
    )

    # ------------------------------------------------------------------
    # Interaction terms
    # ------------------------------------------------------------------
    # Poor transit + low income = compounded burden
    out["low_income_poor_transit"] = (
        (1 - out["transit_access_index"])
        * (1 - out["log_median_income"] / out["log_median_income"].max())
    )

    # Zero-vehicle households far from BART = highest need
    out["zero_veh_x_bart_dist"] = (
        out["zero_vehicle_pct"].fillna(0)
        * out["log_dist_to_bart_m"]
    )

    # ------------------------------------------------------------------
    # Equity need score (for budget engine — not used in ML)
    # ------------------------------------------------------------------
    income_rank = (
        1 - out["median_household_income"]
              .fillna(out["median_household_income"].median())
              .rank(pct=True)
    )
    out["equity_need_score"] = (
        out[TARGET] * 0.50
        + income_rank * 0.30
        + (1 - out["transit_access_index"]) * 0.20
    ).round(4)

    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return numeric ML feature columns — no county dummies, no target leakage."""
    candidates = [
        # Transit
        "stop_density_per_sq_km",
        "pct_area_covered_800m",
        "agencies_serving",
        "avg_stop_departures_per_day",
        "log_dist_to_bart_m",
        "transit_access_index",
        "is_transit_desert",
        # Demographics
        "pop_density",
        "log_median_income",
        "zero_vehicle_pct",
        "transit_mode_share",
        "income_quintile",
        # Interactions
        "car_dependency_index",
        "transit_need_gap",
        "low_income_poor_transit",
        "zero_veh_x_bart_dist",
        # Spatial (continuous — not county dummies)
        "centroid_lat",
        "centroid_lon",
    ]
    return [c for c in candidates if c in df.columns]


def main():
    print("=" * 60)
    print("Bay Area Transit Equity — Feature Engineering v3")
    print("=" * 60)

    path = DATA_PROCESSED / "tract_features.csv"
    if not path.exists():
        raise FileNotFoundError("tract_features.csv not found.")

    print(f"\n📂 Loading tract features...")
    df = pd.read_csv(path, dtype={"GEOID": str})
    print(f"   {len(df)} tracts, {len(df.columns)} raw columns")

    # Show available columns for debugging
    lat_cols = [c for c in df.columns if "LAT" in c.upper() or "LON" in c.upper()]
    print(f"   Spatial columns found: {lat_cols if lat_cols else 'none — will skip lat/lon features'}")

    print(f"\n🔧 Engineering features...")
    df_feat = build_features(df)
    feature_cols = get_feature_columns(df_feat)

    target_vals = df_feat[TARGET].dropna()
    print(f"   ✓ Target: long_commute_pct")
    print(f"   ✓ Range:  {target_vals.min():.3f} – {target_vals.max():.3f}  |  mean: {target_vals.mean():.3f}")
    print(f"   ✓ {len(feature_cols)} features (no county dummies)")
    print(f"   ✓ Transit deserts: {df_feat['is_transit_desert'].sum()} tracts")

    # Build output dataframe
    keep_cols = (
        ["GEOID", "county_name", "total_population"]
        + feature_cols
        + [TARGET, "equity_need_score", "transit_access_index", "is_transit_desert"]
    )
    keep_cols = [c for c in keep_cols if c in df_feat.columns]
    ml_df = df_feat[keep_cols].copy()

    # Fill NAs — numeric only
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(ml_df[c])]
    for col in numeric_features:
        null_count = ml_df[col].isna().sum()
        if null_count > 0:
            ml_df[col] = ml_df[col].fillna(ml_df[col].median())

    ml_df = ml_df.dropna(subset=[TARGET])

    ml_df.to_csv(DATA_PROCESSED / "ml_features.csv", index=False)
    print(f"\n💾 ml_features.csv — {len(ml_df)} rows, {len(feature_cols)} features")

    equity_cols = ["GEOID", "county_name", "total_population",
                   "equity_need_score", TARGET, "is_transit_desert",
                   "transit_access_index", "median_household_income"]
    equity_cols = [c for c in equity_cols if c in df_feat.columns]
    df_feat[equity_cols].to_csv(DATA_PROCESSED / "equity_scores.csv", index=False)
    print(f"   equity_scores.csv — {len(df_feat)} rows")

    print("\n" + "=" * 60)
    print("Next step: python src/modeling.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
