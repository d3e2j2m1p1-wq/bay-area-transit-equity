"""
budget_engine.py
-----------------
Budget scenario calculator for transit equity resource allocation.

Given a set of underserved census tracts and cost assumptions,
models the impact of different investment levels and strategies.

Usage:
    python src/budget_engine.py                    # run default scenarios
    python src/budget_engine.py --budget 10000000  # custom single scenario
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_BUDGET = PROJECT_ROOT / "data" / "budget"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Cost Assumptions
# ---------------------------------------------------------------------------

def generate_cost_assumptions() -> pd.DataFrame:
    """
    Transit infrastructure cost benchmarks.
    Sources: National Transit Database (NTD) 2023, published agency reports.
    """
    return pd.DataFrame([
        {
            "cost_item": "Bus operating cost (per revenue-mile)",
            "unit": "per_revenue_mile",
            "low_estimate": 12,
            "mid_estimate": 18,
            "high_estimate": 25,
            "source": "NTD 2023",
        },
        {
            "cost_item": "Bus operating cost (per revenue-hour)",
            "unit": "per_revenue_hour",
            "low_estimate": 150,
            "mid_estimate": 200,
            "high_estimate": 275,
            "source": "NTD 2023",
        },
        {
            "cost_item": "New bus stop (basic)",
            "unit": "per_stop",
            "low_estimate": 5_000,
            "mid_estimate": 15_000,
            "high_estimate": 30_000,
            "source": "Agency reports",
        },
        {
            "cost_item": "New bus stop (shelter + ADA)",
            "unit": "per_stop_shelter",
            "low_estimate": 25_000,
            "mid_estimate": 50_000,
            "high_estimate": 80_000,
            "source": "Agency reports",
        },
        {
            "cost_item": "New bus route (annual operating)",
            "unit": "per_route_mile_year",
            "low_estimate": 200_000,
            "mid_estimate": 350_000,
            "high_estimate": 500_000,
            "source": "Derived from NTD",
        },
        {
            "cost_item": "Route extension",
            "unit": "per_mile_capital",
            "low_estimate": 300_000,
            "mid_estimate": 500_000,
            "high_estimate": 750_000,
            "source": "Derived from NTD",
        },
    ])


# ---------------------------------------------------------------------------
# Scenario Runner
# ---------------------------------------------------------------------------

class BudgetScenario:
    """
    Calculate the impact of a transit investment scenario.

    Takes a ranked list of underserved tracts and cost assumptions,
    then greedily allocates budget to tracts in priority order.
    """

    def __init__(self, costs_df: pd.DataFrame, underserved_tracts_df: pd.DataFrame):
        """
        Parameters:
            costs_df: DataFrame from generate_cost_assumptions()
            underserved_tracts_df: DataFrame with at minimum:
                - GEOID
                - total_population
                - equity_need_score (higher = more underserved)
        """
        self.costs = costs_df
        self.tracts = underserved_tracts_df.copy()

    def _get_cost(self, item_substring: str, tier: str) -> float:
        """Look up a cost estimate by item name substring and tier."""
        col = f"{tier}_estimate"
        match = self.costs[self.costs["cost_item"].str.contains(item_substring, case=False, regex=False)]
        if match.empty:
            raise ValueError(f"No cost item matching '{item_substring}'")
        return float(match.iloc[0][col])

    def run(
        self,
        budget: float,
        cost_tier: str = "mid",
        strategy: str = "highest_need_first",
        stops_per_tract: int = 2,
        route_miles_per_tract: float = 0.5,
    ) -> dict:
        """
        Allocate budget across underserved tracts.

        Parameters:
            budget: Total dollars available
            cost_tier: 'low', 'mid', or 'high'
            strategy: 'highest_need_first' or 'most_population_first'
            stops_per_tract: Assumed new stops needed per tract
            route_miles_per_tract: Assumed new route-miles per tract

        Returns:
            dict with summary metrics and per-tract detail DataFrame
        """
        cost_per_stop = self._get_cost("shelter + ADA", cost_tier)
        cost_per_route_mile = self._get_cost("annual operating", cost_tier)

        # Sort by chosen strategy
        if strategy == "highest_need_first":
            ranked = self.tracts.sort_values("equity_need_score", ascending=False)
        elif strategy == "most_population_first":
            ranked = self.tracts.sort_values("total_population", ascending=False)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        remaining = budget
        improved = []
        total_new_stops = 0
        total_route_miles = 0

        for _, tract in ranked.iterrows():
            tract_cost = (
                stops_per_tract * cost_per_stop
                + route_miles_per_tract * cost_per_route_mile
            )

            if remaining >= tract_cost:
                remaining -= tract_cost
                total_new_stops += stops_per_tract
                total_route_miles += route_miles_per_tract
                improved.append({
                    "GEOID": tract["GEOID"],
                    "total_population": tract["total_population"],
                    "equity_need_score": tract["equity_need_score"],
                    "cost_allocated": tract_cost,
                })
            else:
                break

        detail = pd.DataFrame(improved)
        pop_covered = int(detail["total_population"].sum()) if len(detail) > 0 else 0
        budget_used = budget - remaining

        return {
            "tracts_improved": len(detail),
            "population_newly_covered": pop_covered,
            "total_new_stops": total_new_stops,
            "total_route_miles_added": total_route_miles,
            "budget_used": budget_used,
            "budget_remaining": remaining,
            "budget_utilization_pct": round((budget_used / budget) * 100, 1) if budget > 0 else 0,
            "cost_per_resident": round(budget_used / max(pop_covered, 1), 2),
            "avg_equity_score_improved": (
                round(detail["equity_need_score"].mean(), 3) if len(detail) > 0 else 0
            ),
            "details": detail,
        }


# ---------------------------------------------------------------------------
# Generate Default Scenarios
# ---------------------------------------------------------------------------

SCENARIO_CONFIGS = [
    {
        "scenario_id": "S1",
        "scenario_name": "Minimum Viable",
        "total_budget": 5_000_000,
        "strategy": "highest_need_first",
        "cost_tier": "mid",
        "description": "Maximum equity impact per dollar — focuses on worst-served tracts",
    },
    {
        "scenario_id": "S2",
        "scenario_name": "Population Priority",
        "total_budget": 5_000_000,
        "strategy": "most_population_first",
        "cost_tier": "mid",
        "description": "Maximum residents reached — may skip highest-need but low-pop areas",
    },
    {
        "scenario_id": "S3",
        "scenario_name": "Moderate Expansion",
        "total_budget": 15_000_000,
        "strategy": "highest_need_first",
        "cost_tier": "mid",
        "description": "Meaningful regional improvement with moderate investment",
    },
    {
        "scenario_id": "S4",
        "scenario_name": "Full Equity",
        "total_budget": 30_000_000,
        "strategy": "highest_need_first",
        "cost_tier": "mid",
        "description": "What would it take to eliminate all transit deserts?",
    },
    {
        "scenario_id": "S5",
        "scenario_name": "Conservative Estimate",
        "total_budget": 15_000_000,
        "strategy": "highest_need_first",
        "cost_tier": "high",
        "description": "Same as S3 but using high-end cost estimates",
    },
]


def generate_synthetic_underserved_tracts() -> pd.DataFrame:
    """
    If processed tract data doesn't exist yet, generate synthetic
    underserved tract data for testing the budget engine.
    """
    np.random.seed(42)
    n = 150  # ~10% of Bay Area tracts flagged as underserved

    geoids = [f"0600{np.random.choice(['1','13','75'])}{np.random.randint(300000,500000)}"
              for _ in range(n)]

    return pd.DataFrame({
        "GEOID": geoids,
        "total_population": np.random.randint(1000, 12000, size=n),
        "equity_need_score": np.random.uniform(0.4, 1.0, size=n).round(3),
    })


def run_all_scenarios() -> pd.DataFrame:
    """Run all predefined scenarios and return a comparison table."""
    costs = generate_cost_assumptions()

    # Try to load real processed data, fall back to synthetic
    tract_path = DATA_PROCESSED / "tract_features.csv"
    if tract_path.exists():
        tracts = pd.read_csv(tract_path, dtype={"GEOID": str})
        # Build equity need score from available metrics
        if "equity_need_score" not in tracts.columns:
            # Composite: high long commute + low coverage + low income = high need
            tracts["equity_need_score"] = (
                tracts["long_commute_pct"].fillna(0).rank(pct=True) * 0.3
                + (1 - tracts["pct_area_covered_800m"].fillna(0) / 100).rank(pct=True) * 0.4
                + (1 - tracts["median_household_income"].fillna(tracts["median_household_income"].median())
                   .rank(pct=True)) * 0.3
            )
        underserved = tracts[tracts["equity_need_score"] > tracts["equity_need_score"].quantile(0.7)]
    else:
        print("  ⚠️  No processed tract data found, using synthetic data for demo")
        underserved = generate_synthetic_underserved_tracts()

    engine = BudgetScenario(costs, underserved)

    results = []
    for config in SCENARIO_CONFIGS:
        outcome = engine.run(
            budget=config["total_budget"],
            cost_tier=config["cost_tier"],
            strategy=config["strategy"],
        )
        results.append({
            "scenario_id": config["scenario_id"],
            "scenario_name": config["scenario_name"],
            "total_budget": config["total_budget"],
            "strategy": config["strategy"],
            "cost_tier": config["cost_tier"],
            "description": config["description"],
            "tracts_improved": outcome["tracts_improved"],
            "population_newly_covered": outcome["population_newly_covered"],
            "total_new_stops": outcome["total_new_stops"],
            "total_route_miles_added": outcome["total_route_miles_added"],
            "budget_used": outcome["budget_used"],
            "budget_utilization_pct": outcome["budget_utilization_pct"],
            "cost_per_resident": outcome["cost_per_resident"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run budget scenarios")
    parser.add_argument("--budget", type=float, help="Custom budget amount for single scenario")
    args = parser.parse_args()

    DATA_BUDGET.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bay Area Transit Equity — Budget Scenario Engine")
    print("=" * 60)

    # Save cost assumptions
    costs = generate_cost_assumptions()
    costs.to_csv(DATA_BUDGET / "cost_assumptions.csv", index=False)
    print(f"\n💰 Cost assumptions saved ({len(costs)} line items)")

    if args.budget:
        # Single custom scenario
        print(f"\n🔧 Running custom scenario with ${args.budget:,.0f} budget...")
        underserved = generate_synthetic_underserved_tracts()
        engine = BudgetScenario(costs, underserved)
        result = engine.run(budget=args.budget)
        print(f"   Tracts improved:    {result['tracts_improved']}")
        print(f"   Population covered: {result['population_newly_covered']:,}")
        print(f"   Cost per resident:  ${result['cost_per_resident']:,.2f}")
        print(f"   Budget used:        ${result['budget_used']:,.0f} ({result['budget_utilization_pct']}%)")
    else:
        # Run all default scenarios
        print("\n📊 Running all scenarios...")
        scenarios = run_all_scenarios()
        scenarios.to_csv(DATA_BUDGET / "scenarios.csv", index=False)
        print(f"\n   ✓ {len(scenarios)} scenarios saved to data/budget/scenarios.csv\n")

        # Print comparison table
        display_cols = [
            "scenario_name", "total_budget", "tracts_improved",
            "population_newly_covered", "cost_per_resident", "budget_utilization_pct"
        ]
        summary = scenarios[display_cols].copy()
        summary["total_budget"] = summary["total_budget"].apply(lambda x: f"${x:,.0f}")
        summary["population_newly_covered"] = summary["population_newly_covered"].apply(lambda x: f"{x:,}")
        summary["cost_per_resident"] = summary["cost_per_resident"].apply(lambda x: f"${x:,.2f}")
        summary["budget_utilization_pct"] = summary["budget_utilization_pct"].apply(lambda x: f"{x}%")

        print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
