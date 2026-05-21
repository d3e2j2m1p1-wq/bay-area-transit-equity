"""
generate_crm_data.py
---------------------
Generates realistic synthetic CRM data for the transit equity dashboard:
  - stakeholders.csv         (30 community orgs, council members, agencies)
  - community_feedback.csv   (100 complaints/requests tied to real tracts)
  - feedback_categories.csv  (10 issue types with priority weights)
  - outreach_log.csv         (40 outreach activities)

Feedback is weighted toward underserved tracts (if equity scores exist)
to create realistic correlation patterns.

Usage:
    python src/generate_crm_data.py
"""

import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CRM = PROJECT_ROOT / "data" / "crm"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Real Bay Area tract GEOIDs to anchor data geographically
# ---------------------------------------------------------------------------

# These are real census tract GEOIDs in Alameda & Contra Costa counties
# representing a mix of underserved and well-served areas
SAMPLE_TRACTS = {
    # East Oakland — lower income, transit-dependent
    "06001401000": {"area": "East Oakland", "county": "Alameda", "need": "high"},
    "06001401100": {"area": "East Oakland", "county": "Alameda", "need": "high"},
    "06001401200": {"area": "East Oakland", "county": "Alameda", "need": "high"},
    "06001401600": {"area": "Fruitvale", "county": "Alameda", "need": "high"},
    "06001401700": {"area": "Fruitvale", "county": "Alameda", "need": "high"},
    # West Oakland
    "06001401500": {"area": "West Oakland", "county": "Alameda", "need": "high"},
    "06001983200": {"area": "West Oakland", "county": "Alameda", "need": "medium"},
    # Deep East Oakland
    "06001402000": {"area": "Elmhurst", "county": "Alameda", "need": "high"},
    "06001402100": {"area": "Seminary", "county": "Alameda", "need": "high"},
    # San Leandro / Hayward
    "06001432000": {"area": "San Leandro", "county": "Alameda", "need": "medium"},
    "06001433000": {"area": "Hayward", "county": "Alameda", "need": "medium"},
    "06001434000": {"area": "South Hayward", "county": "Alameda", "need": "high"},
    # Fremont / Union City
    "06001441000": {"area": "Fremont", "county": "Alameda", "need": "medium"},
    "06001442000": {"area": "Union City", "county": "Alameda", "need": "medium"},
    # Contra Costa — Richmond / San Pablo
    "06013360000": {"area": "Richmond", "county": "Contra Costa", "need": "high"},
    "06013361000": {"area": "San Pablo", "county": "Contra Costa", "need": "high"},
    "06013362000": {"area": "North Richmond", "county": "Contra Costa", "need": "high"},
    # Contra Costa — Concord / Pittsburg / Antioch
    "06013313000": {"area": "Concord", "county": "Contra Costa", "need": "medium"},
    "06013314000": {"area": "Pittsburg", "county": "Contra Costa", "need": "high"},
    "06013315000": {"area": "Antioch", "county": "Contra Costa", "need": "high"},
    # San Francisco — Tenderloin / SOMA / Bayview
    "06075012400": {"area": "Tenderloin", "county": "San Francisco", "need": "high"},
    "06075012600": {"area": "SOMA", "county": "San Francisco", "need": "medium"},
    "06075023100": {"area": "Bayview", "county": "San Francisco", "need": "high"},
    # Well-served areas (for contrast)
    "06001420100": {"area": "Downtown Oakland", "county": "Alameda", "need": "low"},
    "06075010100": {"area": "Financial District", "county": "San Francisco", "need": "low"},
    "06075017600": {"area": "Mission District", "county": "San Francisco", "need": "low"},
}


# ---------------------------------------------------------------------------
# 1. Feedback Categories
# ---------------------------------------------------------------------------

def generate_feedback_categories() -> pd.DataFrame:
    return pd.DataFrame([
        {"category_id": "overcrowding", "category_label": "Overcrowding", "priority_weight": 2.5},
        {"category_id": "frequency", "category_label": "Service Frequency", "priority_weight": 3.0},
        {"category_id": "safety", "category_label": "Safety & Security", "priority_weight": 2.5},
        {"category_id": "accessibility", "category_label": "ADA Accessibility", "priority_weight": 2.0},
        {"category_id": "route_coverage", "category_label": "Route Coverage Gap", "priority_weight": 3.0},
        {"category_id": "cleanliness", "category_label": "Cleanliness", "priority_weight": 1.0},
        {"category_id": "fare", "category_label": "Fare Issues", "priority_weight": 1.5},
        {"category_id": "schedule_reliability", "category_label": "Schedule Reliability", "priority_weight": 2.5},
        {"category_id": "new_stop_request", "category_label": "New Stop Request", "priority_weight": 2.0},
        {"category_id": "other", "category_label": "Other", "priority_weight": 1.0},
    ])


# ---------------------------------------------------------------------------
# 2. Stakeholders
# ---------------------------------------------------------------------------

def generate_stakeholders() -> pd.DataFrame:
    stakeholders = [
        # Community organizations
        {"name": "East Oakland Neighbors Coalition", "type": "community_org", "county": "Alameda", "primary_tract_geoid": "06001401000", "priority_level": "high"},
        {"name": "Fruitvale Unity Council", "type": "community_org", "county": "Alameda", "primary_tract_geoid": "06001401600", "priority_level": "high"},
        {"name": "West Oakland Environmental Indicators Project", "type": "community_org", "county": "Alameda", "primary_tract_geoid": "06001401500", "priority_level": "high"},
        {"name": "Richmond Community Foundation", "type": "community_org", "county": "Contra Costa", "primary_tract_geoid": "06013360000", "priority_level": "high"},
        {"name": "Hayward Area Recreation District", "type": "community_org", "county": "Alameda", "primary_tract_geoid": "06001433000", "priority_level": "medium"},
        {"name": "Pittsburg Community Action Council", "type": "community_org", "county": "Contra Costa", "primary_tract_geoid": "06013314000", "priority_level": "high"},
        {"name": "SF Tenderloin Community Benefit District", "type": "community_org", "county": "San Francisco", "primary_tract_geoid": "06075012400", "priority_level": "high"},
        {"name": "Bayview Hunters Point Community Advocates", "type": "community_org", "county": "San Francisco", "primary_tract_geoid": "06075023100", "priority_level": "high"},
        # Advocacy groups
        {"name": "TransForm (Regional Transit Advocacy)", "type": "advocacy_group", "county": "Alameda", "primary_tract_geoid": "06001420100", "priority_level": "high"},
        {"name": "Bike East Bay", "type": "advocacy_group", "county": "Alameda", "primary_tract_geoid": "06001420100", "priority_level": "medium"},
        {"name": "SPUR (SF Bay Area Planning & Urban Research)", "type": "advocacy_group", "county": "San Francisco", "primary_tract_geoid": "06075010100", "priority_level": "medium"},
        {"name": "Antioch Transit Riders Alliance", "type": "advocacy_group", "county": "Contra Costa", "primary_tract_geoid": "06013315000", "priority_level": "high"},
        # City council contacts
        {"name": "Oakland City Council — District 5", "type": "city_council", "county": "Alameda", "primary_tract_geoid": "06001401000", "priority_level": "high"},
        {"name": "Oakland City Council — District 7", "type": "city_council", "county": "Alameda", "primary_tract_geoid": "06001402000", "priority_level": "high"},
        {"name": "Richmond City Council", "type": "city_council", "county": "Contra Costa", "primary_tract_geoid": "06013360000", "priority_level": "medium"},
        {"name": "Hayward City Council — Transportation Cmte", "type": "city_council", "county": "Alameda", "primary_tract_geoid": "06001433000", "priority_level": "medium"},
        {"name": "Pittsburg City Council", "type": "city_council", "county": "Contra Costa", "primary_tract_geoid": "06013314000", "priority_level": "medium"},
        {"name": "Antioch City Council", "type": "city_council", "county": "Contra Costa", "primary_tract_geoid": "06013315000", "priority_level": "medium"},
        # Transit agencies
        {"name": "AC Transit — Planning Dept", "type": "transit_agency", "county": "Alameda", "primary_tract_geoid": "06001420100", "priority_level": "high"},
        {"name": "BART — Community Affairs", "type": "transit_agency", "county": "Alameda", "primary_tract_geoid": "06001420100", "priority_level": "high"},
        {"name": "SFMTA — Equity Office", "type": "transit_agency", "county": "San Francisco", "primary_tract_geoid": "06075010100", "priority_level": "high"},
        {"name": "Tri Delta Transit (East CCC)", "type": "transit_agency", "county": "Contra Costa", "primary_tract_geoid": "06013315000", "priority_level": "medium"},
        # Individual residents (sample)
        {"name": "Maria Gonzalez", "type": "resident", "county": "Alameda", "primary_tract_geoid": "06001401700", "priority_level": "low"},
        {"name": "James Washington", "type": "resident", "county": "Alameda", "primary_tract_geoid": "06001402100", "priority_level": "low"},
        {"name": "Li Wei Chen", "type": "resident", "county": "San Francisco", "primary_tract_geoid": "06075012400", "priority_level": "low"},
        {"name": "Angela Reyes", "type": "resident", "county": "Contra Costa", "primary_tract_geoid": "06013361000", "priority_level": "low"},
    ]

    df = pd.DataFrame(stakeholders)
    df["stakeholder_id"] = [f"STK-{i+1:03d}" for i in range(len(df))]
    df["contact_email"] = df["name"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(" ", ".", regex=False).str[:30] + "@example.org"

    # Random last contact dates (within past 6 months)
    base_date = datetime(2026, 4, 1)
    df["last_contact_date"] = [
        (base_date - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d")
        for _ in range(len(df))
    ]
    df["notes"] = ""

    return df[["stakeholder_id", "name", "type", "county", "primary_tract_geoid",
               "contact_email", "priority_level", "last_contact_date", "notes"]]


# ---------------------------------------------------------------------------
# 3. Community Feedback
# ---------------------------------------------------------------------------

FEEDBACK_TEMPLATES = {
    "overcrowding": [
        "Bus is consistently packed during morning rush, riders left at stop",
        "Standing room only every day on the {route} line",
        "Cannot board with stroller due to overcrowding on weekday mornings",
    ],
    "frequency": [
        "Bus only comes every 45 minutes, need more frequent service",
        "Wait times too long after 7 PM, cuts off evening commuters",
        "Weekend service on {route} reduced to hourly — not usable",
    ],
    "safety": [
        "Poor lighting at bus stop, feels unsafe after dark",
        "No shelter at stop on busy road, pedestrians at risk",
        "Multiple incidents reported at this station in past month",
    ],
    "route_coverage": [
        "No direct bus route connecting our neighborhood to BART",
        "Nearest bus stop is a 25-minute walk from residential area",
        "No transit service to the new medical center on {route} corridor",
    ],
    "schedule_reliability": [
        "Bus regularly arrives 10-15 minutes late",
        "Real-time predictions don't match actual arrivals",
        "Route {route} frequently skips stops without explanation",
    ],
    "new_stop_request": [
        "Requesting new stop near senior housing on Main St",
        "Large apartment complex 1 mile from nearest stop, need coverage",
        "New school campus has no transit access",
    ],
    "accessibility": [
        "Wheelchair ramp at stop is broken, reported twice already",
        "Audible announcements not working on {route} vehicles",
        "Bus kneel feature not functioning at this stop",
    ],
    "fare": [
        "Transfer between BART and AC Transit too expensive for daily commuters",
        "Clipper card machines frequently broken at this station",
        "Need low-income fare program awareness at this location",
    ],
    "cleanliness": [
        "Bus shelter covered in graffiti, bench broken",
        "Trash overflow at stop, no regular maintenance",
    ],
    "other": [
        "Requesting better signage for route changes",
        "Bus stop sign is missing, confuses new riders",
    ],
}

SAMPLE_ROUTES = ["51A", "57", "99", "1R", "72M", "18", "NL", "40", "97", "52"]


def generate_feedback(n: int = 100) -> pd.DataFrame:
    """Generate n synthetic feedback records."""
    categories = list(FEEDBACK_TEMPLATES.keys())
    # Weight toward high-priority categories
    cat_weights = [0.15, 0.20, 0.10, 0.15, 0.15, 0.08, 0.05, 0.05, 0.04, 0.03]

    # Weight tract selection toward high-need areas
    tract_ids = list(SAMPLE_TRACTS.keys())
    tract_weights = [
        3.0 if SAMPLE_TRACTS[t]["need"] == "high"
        else 1.5 if SAMPLE_TRACTS[t]["need"] == "medium"
        else 0.5
        for t in tract_ids
    ]
    tract_weights = np.array(tract_weights) / sum(tract_weights)

    agencies = ["AC Transit", "BART", "SFMTA"]
    agency_weights = [0.50, 0.30, 0.20]
    sentiments = ["negative", "neutral", "positive"]
    statuses = ["open", "in_progress", "resolved", "closed"]
    status_weights = [0.35, 0.25, 0.25, 0.15]

    records = []
    base_date = datetime(2026, 4, 15)

    for i in range(n):
        cat = np.random.choice(categories, p=cat_weights)
        tract = np.random.choice(tract_ids, p=tract_weights)
        route = random.choice(SAMPLE_ROUTES)
        template = random.choice(FEEDBACK_TEMPLATES[cat]).replace("{route}", route)

        # Sentiment: most feedback in high-need areas is negative
        need = SAMPLE_TRACTS[tract]["need"]
        if need == "high":
            sent = np.random.choice(sentiments, p=[0.70, 0.20, 0.10])
        elif need == "medium":
            sent = np.random.choice(sentiments, p=[0.50, 0.30, 0.20])
        else:
            sent = np.random.choice(sentiments, p=[0.20, 0.30, 0.50])

        records.append({
            "feedback_id": f"FB-{i+1:04d}",
            "date_submitted": (base_date - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "tract_geoid": tract,
            "category": cat,
            "route_id": route if cat != "new_stop_request" else "",
            "stop_id": f"{random.randint(50000, 59999)}" if random.random() > 0.3 else "",
            "agency": np.random.choice(agencies, p=agency_weights),
            "sentiment": sent,
            "description": template,
            "status": np.random.choice(statuses, p=status_weights),
            "assigned_to": f"STK-{random.randint(1, 26):03d}" if random.random() > 0.5 else "",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Outreach Log
# ---------------------------------------------------------------------------

def generate_outreach_log(stakeholders_df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Generate synthetic outreach records tied to stakeholders and tracts."""
    types = ["town_hall", "survey", "email", "phone", "site_visit"]
    type_weights = [0.20, 0.15, 0.30, 0.20, 0.15]

    summaries = {
        "town_hall": [
            "Residents voiced concerns about {area} bus frequency and safety",
            "Community meeting on proposed route changes in {area}",
            "Public hearing on transit equity findings for {area} tracts",
        ],
        "survey": [
            "Distributed rider satisfaction surveys at {area} stops",
            "Online survey on transit priorities collected 120 responses from {area}",
        ],
        "email": [
            "Shared equity analysis findings with {stakeholder}",
            "Follow-up on feedback items from {area} residents",
            "Monthly update to {stakeholder} on service improvements",
        ],
        "phone": [
            "Call with {stakeholder} regarding route {route} concerns",
            "Check-in with {stakeholder} on outreach coordination",
        ],
        "site_visit": [
            "Walked {area} corridor to assess stop conditions",
            "Inspected proposed new stop locations in {area}",
            "Documented accessibility issues at {area} stations",
        ],
    }

    base_date = datetime(2026, 4, 15)
    records = []

    for i in range(n):
        stk = stakeholders_df.iloc[random.randint(0, len(stakeholders_df) - 1)]
        otype = np.random.choice(types, p=type_weights)
        tract = stk["primary_tract_geoid"]
        area = SAMPLE_TRACTS.get(tract, {}).get("area", "the area")

        template = random.choice(summaries[otype])
        summary = template.replace("{area}", area).replace(
            "{stakeholder}", stk["name"]
        ).replace("{route}", random.choice(SAMPLE_ROUTES))

        records.append({
            "outreach_id": f"OUT-{i+1:03d}",
            "date": (base_date - timedelta(days=random.randint(0, 270))).strftime("%Y-%m-%d"),
            "stakeholder_id": stk["stakeholder_id"],
            "tract_geoid": tract,
            "type": otype,
            "attendees": random.randint(1, 85) if otype == "town_hall" else random.randint(1, 5),
            "summary": summary,
            "follow_up_needed": random.random() > 0.4,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Bay Area Transit Equity — CRM Data Generator")
    print("=" * 60)

    DATA_CRM.mkdir(parents=True, exist_ok=True)

    print("\n📋 Generating feedback categories...")
    categories = generate_feedback_categories()
    categories.to_csv(DATA_CRM / "feedback_categories.csv", index=False)
    print(f"   ✓ {len(categories)} categories")

    print("\n👥 Generating stakeholders...")
    stakeholders = generate_stakeholders()
    stakeholders.to_csv(DATA_CRM / "stakeholders.csv", index=False)
    print(f"   ✓ {len(stakeholders)} stakeholders")

    print("\n💬 Generating community feedback...")
    feedback = generate_feedback(100)
    feedback.to_csv(DATA_CRM / "community_feedback.csv", index=False)
    print(f"   ✓ {len(feedback)} feedback records")

    # Summary stats
    print(f"     Sentiment breakdown: {feedback['sentiment'].value_counts().to_dict()}")
    print(f"     Top categories: {feedback['category'].value_counts().head(3).to_dict()}")

    print("\n📞 Generating outreach log...")
    outreach = generate_outreach_log(stakeholders, 40)
    outreach.to_csv(DATA_CRM / "outreach_log.csv", index=False)
    print(f"   ✓ {len(outreach)} outreach records")

    print("\n" + "=" * 60)
    print(f"All CRM data saved to: {DATA_CRM}")
    print("=" * 60)


if __name__ == "__main__":
    main()
