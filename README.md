# 🚍 Bay Area Transit Equity Planning Platform

A geospatial ML platform that identifies underserved communities across the San Francisco Bay Area's transit network, models budget scenarios for equitable resource allocation, and provides an operational Power BI dashboard for transit agency planners.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![QGIS](https://img.shields.io/badge/QGIS-3.x-green)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## The Problem

The Bay Area's 27 transit agencies serve 7.7 million residents across 9 counties, but access is uneven. Low-income communities and communities of color often face longer commutes, fewer transit options, and greater reliance on infrequent bus service — while better-resourced neighborhoods enjoy BART access and high-frequency routes.

**This platform answers:** Which communities are most underserved, what would it cost to close the gap, and how should a transit agency prioritize investments?

## What This Project Does

1. **Data Pipeline** — Ingests GTFS transit feeds (BART, AC Transit, SFMTA), Census ACS demographic data, and TIGER/Line shapefiles via automated Python scripts
2. **Geospatial Analysis** — Calculates transit stop density, 800m walk-coverage, service frequency, and commute burden metrics per census tract using GeoPandas and QGIS
3. **ML Modeling** — Predicts commute burden scores using GradientBoosting with spatial cross-validation and SHAP explainability
4. **Budget Scenario Engine** — Models the cost and impact of different investment strategies using NTD cost benchmarks
5. **CRM / Community Tracker** — Tracks stakeholder engagement and community feedback linked to geographic equity data
6. **Power BI Dashboard** — Four-page interactive dashboard: Equity Overview → Route Analysis → Budget Planner → Community Engagement

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Collection | Python, Census API, GTFS, requests |
| Geospatial | QGIS, GeoPandas, Shapely, Folium |
| ML | Scikit-learn, XGBoost, SHAP |
| Dashboard | Power BI (or Tableau) |
| Data Storage | CSV, GeoJSON, PostGIS (optional) |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/bay-area-transit-equity.git
cd bay-area-transit-equity

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your Census API key (free: https://api.census.gov/data/key_signup.html)

# Run the pipeline
python src/data_collection.py     # fetch all raw data
python src/preprocessing.py       # clean, merge, spatial joins
python src/generate_crm_data.py   # create synthetic CRM data
python src/budget_engine.py       # run budget scenarios
```

## Project Structure

```
bay-area-transit-equity/
├── src/
│   ├── data_collection.py        # Census API + GTFS + shapefile downloads
│   ├── preprocessing.py          # spatial joins, coverage metrics
│   ├── feature_engineering.py    # ML feature construction
│   ├── modeling.py               # train/evaluate ML models
│   ├── budget_engine.py          # investment scenario calculator
│   ├── generate_crm_data.py      # synthetic CRM data generator
│   └── visualization.py          # Folium maps, charts
├── data/
│   ├── raw/                      # GTFS feeds, Census CSVs, shapefiles
│   ├── processed/                # analysis-ready datasets
│   ├── crm/                      # stakeholder & feedback data
│   └── budget/                   # cost assumptions & scenarios
├── notebooks/                    # Jupyter analysis notebooks
├── dashboard/                    # Power BI file + screenshots
├── qgis/                         # QGIS project + map exports
└── maps/                         # interactive Folium HTML maps
```

## Key Findings

> _To be completed after analysis_

## Author

**Miguel** — M.S. Business Analytics, CSU East Bay (2026)

## License

MIT
