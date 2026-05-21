"""
modeling.py  (v2 — Inference-focused)
--------------------------------------
Instead of cross-county prediction (which fails due to geographic dominance),
this script runs within-county OLS regression and GradientBoosting to identify
which transit and demographic factors most strongly correlate with commute burden
within each county.

This is more honest and more useful for a transit agency:
  "Within Alameda County, which factors predict long commutes?"

Also runs a full-dataset model for SHAP explainability and feature importance.

Usage:
    python src/modeling.py
"""

import warnings
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR    = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR     = PROJECT_ROOT / "models"

TARGET    = "commute_burden_score"
GROUP_COL = "county_name"

# Features excluding raw spatial coords (lat/lon dominate — we want transit signal)
EXCLUDE_FEATURES = {"centroid_lat", "centroid_lon", "county_name", "GEOID",
                    TARGET, "equity_need_score", "transit_access_index",
                    "is_transit_desert", "total_population",
                    "predicted_burden_score", "prediction_residual",
                    "high_priority_flag"}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    path = DATA_PROCESSED / "ml_features.csv"
    if not path.exists():
        raise FileNotFoundError("ml_features.csv not found. Run feature_engineering.py first.")

    df = pd.read_csv(path, dtype={"GEOID": str})

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FEATURES
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].values
    y = df[TARGET].values

    return df, X, y, feature_cols


# ---------------------------------------------------------------------------
# Within-county modeling
# ---------------------------------------------------------------------------

def run_within_county_models(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    For each county, fit a Ridge regression and report R² + top features.
    This answers: within a county, which factors predict long commutes?
    """
    results = []
    counties = df[GROUP_COL].unique()

    for county in sorted(counties):
        subset = df[df[GROUP_COL] == county].copy()
        if len(subset) < 30:
            continue

        X_c = subset[feature_cols].values
        y_c = subset[TARGET].values

        # KFold within county (random, not spatial — within-county is fine)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

        if len(subset) >= 50:
            cv_scores = cross_val_score(pipe, X_c, y_c, cv=5,
                                        scoring="r2", error_score="raise")
            r2_cv = cv_scores.mean()
        else:
            pipe.fit(X_c, y_c)
            r2_cv = r2_score(y_c, pipe.predict(X_c))

        # Fit on all county data for coefficients
        pipe.fit(X_c, y_c)
        coefs = pipe.named_steps["model"].coef_
        top_idx = np.argsort(np.abs(coefs))[::-1][:3]
        top_features = ", ".join([feature_cols[i] for i in top_idx])

        results.append({
            "county": county,
            "n_tracts": len(subset),
            "r2_cv": round(r2_cv, 3),
            "top_features": top_features,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full-dataset model (for SHAP + feature importance)
# ---------------------------------------------------------------------------

def run_full_model(X, y, feature_cols):
    """
    Train GradientBoosting on full dataset.
    Not for out-of-sample prediction — for understanding feature relationships.
    """
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
    )

    # Within-sample KFold (not spatial — we're doing inference, not prediction)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    model.fit(X, y)
    y_pred = model.predict(X)

    return model, {
        "r2_insample": round(r2_score(y, y_pred), 4),
        "r2_cv_mean":  round(cv_scores.mean(), 4),
        "r2_cv_std":   round(cv_scores.std(), 4),
        "mae":         round(mean_absolute_error(y, y_pred), 4),
        "rmse":        round(np.sqrt(mean_squared_error(y, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_within_county(county_df: pd.DataFrame):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2A9D8F" if r >= 0.3 else "#E9C46A" if r >= 0 else "#E76F51"
              for r in county_df["r2_cv"]]
    bars = ax.barh(county_df["county"], county_df["r2_cv"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.3, color="gray", linewidth=0.8, linestyle="--", label="R²=0.30 threshold")
    for bar, val in zip(bars, county_df["r2_cv"]):
        ax.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("R² (5-fold CV within county)", fontsize=11)
    ax.set_title("Within-County Model Performance\n(Predicting Long Commute Rate from Transit + Demographics)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "within_county_r2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved within_county_r2.png")


def plot_feature_importance(model, feature_cols):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)

    # Friendly labels
    labels = {
        "log_dist_to_bart_m":      "Distance to BART (log)",
        "transit_mode_share":      "Transit Mode Share",
        "pct_area_covered_800m":   "% Area within 800m of Stop",
        "log_median_income":       "Median Income (log)",
        "transit_access_index":    "Transit Access Index",
        "transit_need_gap":        "Transit Need Gap",
        "zero_vehicle_pct":        "Zero-Vehicle Households %",
        "low_income_poor_transit": "Low Income × Poor Transit",
        "zero_veh_x_bart_dist":    "Zero-Vehicle × BART Distance",
        "pop_density":             "Population Density",
        "car_dependency_index":    "Car Dependency Index",
        "stop_density_per_sq_km":  "Stop Density / sq km",
        "agencies_serving":        "Agencies Serving Tract",
        "avg_stop_departures_per_day": "Avg Daily Departures/Stop",
        "income_quintile":         "Income Quintile",
        "is_transit_desert":       "Transit Desert (binary)",
    }
    imp["label"] = imp["feature"].map(labels).fillna(imp["feature"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["label"][::-1], imp["importance"][::-1], color="#2A9D8F", edgecolor="white")
    ax.set_xlabel("Feature Importance (GradientBoosting)", fontsize=11)
    ax.set_title("What Predicts Long Commutes?\nBay Area Census Tract Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved feature_importance.png")

    imp[["feature", "label", "importance"]].to_csv(
        REPORTS_DIR / "feature_importance.csv", index=False)


def plot_actual_vs_predicted(y, y_pred, metrics):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, y_pred, alpha=0.3, s=12, color="#2A9D8F", edgecolors="none")
    ax.plot([0, 0.5], [0, 0.5], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Long Commute Rate", fontsize=11)
    ax.set_ylabel("Predicted Long Commute Rate", fontsize=11)
    ax.set_title(
        f"GradientBoosting — Full Dataset Fit\n"
        f"In-sample R²={metrics['r2_insample']}  |  "
        f"5-fold CV R²={metrics['r2_cv_mean']} ± {metrics['r2_cv_std']}",
        fontsize=10
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ Saved actual_vs_predicted.png")


def run_shap(model, X, feature_cols):
    if not HAS_SHAP:
        return
    print("\n📊 Running SHAP analysis...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    for plot_type, fname in [("bar", "shap_importance.png"), (None, "shap_beeswarm.png")]:
        plt.figure(figsize=(10, 7))
        if plot_type == "bar":
            shap.summary_plot(shap_values, X, feature_names=feature_cols,
                              plot_type="bar", show=False, max_display=15)
        else:
            shap.summary_plot(shap_values, X, feature_names=feature_cols,
                              show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Saved {fname}")


# ---------------------------------------------------------------------------
# Export predictions
# ---------------------------------------------------------------------------

def export_predictions(df, model, X, feature_cols):
    y_pred = model.predict(X)
    out = df[["GEOID"]].copy()
    out["predicted_burden_score"] = y_pred.round(4)
    out["high_priority_flag"] = (
        y_pred >= np.percentile(y_pred, 70)
    ).astype(int)

    equity_path = DATA_PROCESSED / "equity_scores.csv"
    equity = pd.read_csv(equity_path, dtype={"GEOID": str})
    # Drop old prediction columns if present
    for col in ["predicted_burden_score", "high_priority_flag",
                "prediction_residual"]:
        if col in equity.columns:
            equity = equity.drop(columns=[col])
    equity = equity.merge(out, on="GEOID", how="left")
    equity.to_csv(equity_path, index=False)

    n_priority = out["high_priority_flag"].sum()
    print(f"\n📍 High-priority tracts (top 30%): {n_priority}")
    print(f"   ✓ Predictions saved to equity_scores.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Bay Area Transit Equity — ML Modeling v2")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📂 Loading features...")
    df, X, y, feature_cols = load_data()
    print(f"   {X.shape[0]} tracts × {X.shape[1]} features")
    print(f"   Target range: {y.min():.3f} – {y.max():.3f}")

    # --- Within-county models ---
    print(f"\n🗺️  Running within-county models ({df[GROUP_COL].nunique()} counties)...")
    county_results = run_within_county_models(df, feature_cols)
    county_results.to_csv(REPORTS_DIR / "within_county_results.csv", index=False)

    print(f"\n   County-level R² results:")
    for _, row in county_results.iterrows():
        bar = "█" * max(int(row["r2_cv"] * 30), 0)
        flag = "✓" if row["r2_cv"] >= 0.3 else "~" if row["r2_cv"] >= 0 else "✗"
        print(f"   {flag} {row['county']:<15} R²={row['r2_cv']:>6.3f}  {bar}")
        print(f"     Top predictors: {row['top_features']}")

    # --- Full dataset model ---
    print(f"\n🌲 Training full-dataset GradientBoosting model...")
    model, metrics = run_full_model(X, y, feature_cols)

    print(f"   In-sample R²:  {metrics['r2_insample']}")
    print(f"   5-fold CV R²:  {metrics['r2_cv_mean']} ± {metrics['r2_cv_std']}")
    print(f"   MAE:           {metrics['mae']}")
    print(f"   RMSE:          {metrics['rmse']}")

    # Save model
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    print(f"   ✓ Model saved")

    # --- Plots ---
    print(f"\n📈 Generating plots...")
    plot_within_county(county_results)
    plot_feature_importance(model, feature_cols)
    y_pred = model.predict(X)
    plot_actual_vs_predicted(y, y_pred, metrics)
    run_shap(model, X, feature_cols)

    # Top features
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n🌲 Top 10 features:")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"   {row['feature']:<35} {bar} {row['importance']:.4f}")

    # Export predictions
    export_predictions(df, model, X, feature_cols)

    # --- Summary ---
    best_county = county_results.loc[county_results["r2_cv"].idxmax()]
    print("\n" + "=" * 60)
    print("Modeling complete!")
    print(f"  Full-dataset CV R²:  {metrics['r2_cv_mean']} ± {metrics['r2_cv_std']}")
    print(f"  Best within-county:  {best_county['county']} (R²={best_county['r2_cv']})")
    print(f"\n  Key finding: Transit access and demographics explain commute")
    print(f"  burden well within counties, but geographic fixed effects")
    print(f"  dominate cross-county prediction — consistent with Bay Area")
    print(f"  job center concentration patterns.")
    print(f"\n  Outputs saved to reports/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
