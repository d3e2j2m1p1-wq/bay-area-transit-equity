# 🚍 Bay Area Transit Equity Planning Platform

**A geospatial ML platform identifying underserved transit communities across the San Francisco Bay Area, with budget scenario modeling, community engagement tracking, and an interactive Power BI dashboard.**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.x-green)
![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14-brightgreen)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## The Problem

The San Francisco Bay Area's 27 transit agencies serve 7.7 million residents across 9 counties — but access is deeply uneven. Low-income communities and communities of color frequently face longer commutes, fewer route options, and greater dependence on infrequent bus service, while better-resourced neighborhoods benefit from BART access and high-frequency corridors.

**This platform answers three questions a transit agency actually needs to act on:**
1. Which communities are most underserved, and by how much?
2. What would it cost to close the gap?
3. What are residents in those communities actually saying?

---

## What This Project Does

| Phase | What It Produces |
|-------|-----------------|
| **Data Pipeline** | Ingests GTFS feeds (BART, AC Transit, SFMTA), Census ACS demographics, and TIGER/Line shapefiles via automated Python scripts |
| **Geospatial Analysis** | Calculates stop density, 800m walk-coverage, service frequency, and commute burden per census tract using GeoPandas and QGIS |
| **ML Modeling** | Predicts commute burden scores using GradientBoosting with spatial cross-validation and SHAP explainability *(in progress)* |
| **Budget Engine** | Models cost and population impact of 5 investment scenarios using NTD cost benchmarks |
| **CRM Tracker** | Tracks stakeholder engagement and community feedback geographically linked to equity data |
| **Power BI Dashboard** | Four-page interactive dashboard: Equity Overview → Route Analysis → Budget Planner → Community Engagement *(in progress)* |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Census tracts analyzed | **1,772** across 9 Bay Area counties |
| Transit stops mapped | **3,351** (BART: 105, AC Transit: 4,704 freq. stops, SFMTA: 3,246) |
| ACS variables collected | **16** demographic and commute indicators per tract |
| Budget scenarios modeled | **5** ($5M – $30M investment levels) |
| CRM records generated | **26** stakeholders, **100** feedback records, **40** outreach logs |

### Budget Scenario Results

| Scenario | Budget | Tracts Improved | Population Reached | Cost/Resident |
|----------|--------|----------------|--------------------|---------------|
| Minimum Viable | $5M | 18 | 75,639 | $65.44 |
| Population Priority | $5M | 18 | 151,567 | $32.66 |
| Moderate Expansion | $15M | 54 | 232,878 | $63.77 |
| Full Equity | $30M | 109 | 458,419 | $65.39 |
| Conservative Estimate | $15M | 36 | 152,207 | $96.97 |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Collection | Python, Census REST API, GTFS static feeds, `requests` |
| Geospatial | QGIS 3.x, GeoPandas, Shapely, Folium |
| Data Processing | Pandas, NumPy, `python-dotenv` |
| ML *(in progress)* | Scikit-learn, XGBoost, SHAP |
| Dashboard *(in progress)* | Power BI |
| Data Sources | US Census ACS 5-Year (2022), BART/AC Transit/SFMTA GTFS, TIGER/Line shapefiles |

---

## Project Structure

```
bay-area-transit-equity/
│
├── src/
│   ├── data_collection.py      # Census API + GTFS + shapefile downloads
│   ├── preprocessing.py        # spatial joins, stop density, buffer coverage
│   ├── budget_engine.py        # investment scenario calculator (5 scenarios)
│   ├── generate_crm_data.py    # synthetic stakeholder & feedback data
│   ├── feature_engineering.py  # ML feature construction (in progress)
│   └── modeling.py             # GradientBoosting + SHAP (in progress)
│
├── data/
│   ├── raw/
│   │   ├── gtfs/{bart,actransit,sfmta}/  # GTFS static feeds
│   │   ├── census/                        # ACS 5-year tract data
│   │   └── shapefiles/                    # TIGER/Line CA tract boundaries
│   ├── processed/
│   │   ├── tract_features.csv             # 1,772 tracts, 30+ features
│   │   ├── tract_features.geojson         # spatial version for QGIS/Folium
│   │   ├── all_stops.geojson              # 3,351 stops with agency + frequency
│   │   └── dashboard_export.csv           # flattened for Power BI
│   ├── crm/
│   │   ├── stakeholders.csv               # 26 agency contacts & community orgs
│   │   ├── community_feedback.csv         # 100 feedback records by tract
│   │   ├── outreach_log.csv               # 40 engagement activities
│   │   └── feedback_categories.csv        # 10 issue types with priority weights
│   └── budget/
│       ├── cost_assumptions.csv           # NTD cost benchmarks
│       └── scenarios.csv                  # 5 investment scenario outputs
│
├── notebooks/                   # Jupyter analysis notebooks (in progress)
├── qgis/                        # QGIS project file + exported map images
├── dashboard/                   # Power BI .pbix file + screenshots
└── maps/                        # Interactive Folium HTML maps
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/d3e2j2m1p1-wq/bay-area-transit-equity.git
cd bay-area-transit-equity

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your Census API key (free: https://api.census.gov/data/key_signup.html)
cp .env.example .env
# Edit .env and paste your key

# 4. Download GTFS feeds manually and extract into data/raw/gtfs/{bart,actransit,sfmta}/
#    BART:      https://www.bart.gov/dev/schedules/google_transit.zip
#    AC Transit: https://opendata.actransit.org/dataset/general-transit-feed-specification-gtfs
#    SFMTA:     https://www.sfmta.com/reports/gtfs-transit-data

# 5. Run the pipeline
python src/data_collection.py --census      # pulls ACS data for 9 counties
python src/data_collection.py --shapefiles  # downloads tract boundaries
python src/preprocessing.py                 # spatial joins + coverage metrics
python src/generate_crm_data.py             # creates CRM data
python src/budget_engine.py                 # runs 5 budget scenarios
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| US Census Bureau | ACS 5-Year 2022, tract-level demographics & commute | Free API key |
| BART | GTFS static feed — 105 stations | No registration |
| AC Transit | GTFS static feed — Aug 2025 service schedule | Open Data Portal |
| SFMTA | GTFS static feed — 3,246 stops, 70 routes | License agreement |
| TIGER/Line | 2022 California census tract boundaries | No registration |

---

## Geospatial Metrics Calculated Per Tract

- `stop_count` — total transit stops within tract boundaries
- `stop_density_per_sq_km` — stops per square kilometer
- `pct_area_covered_800m` — % of tract within 800m walk of any stop
- `agencies_serving` — number of distinct agencies serving the tract
- `avg_stop_departures_per_day` — average weekday departures per stop
- `dist_to_nearest_bart_m` — distance in meters to nearest BART station
- `transit_mode_share` — share of commuters using public transit
- `zero_vehicle_pct` — share of households with no vehicle
- `long_commute_pct` — share of commuters with 60+ minute commutes
- `pop_density` — population per square kilometer

---

## Maps

> *QGIS map exports will appear here once the visualization phase is complete.*

---

## Dashboard

> *Power BI dashboard screenshots will appear here once complete.*

**Planned pages:**
1. **Equity Overview** — choropleth map + KPI cards (underserved population, coverage %, commute burden)
2. **Route & Stop Analysis** — stop-level frequency map by agency
3. **Budget Scenario Planner** — what-if investment modeling with sliders
4. **Community Engagement Tracker** — feedback heatmap + stakeholder activity log

---

## Author

**Miguel Davila** — M.S. Business Analytics, CSU East Bay (May 2026)

[LinkedIn](https://linkedin.com/in/miguel-md-davila) · [GitHub](https://github.com/d3e2j2m1p1-wq)

---

## License

MIT
