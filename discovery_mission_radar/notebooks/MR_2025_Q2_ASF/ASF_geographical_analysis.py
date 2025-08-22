#!/usr/bin/env python3
"""
ASF Geographical Analysis Script

This script analyzes geographical investment patterns across multiple ASF categories:
- Heat pumps
- Geothermal heating  
- District heating
- Heat storage

For each category, it produces:
1. Regional investment breakdowns
2. Top European countries comparison (including UK)
3. Charts saved as PNG files
4. CSV data exports

Usage:
    python ASF_geographical_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

from discovery_utils.getters import crunchbase
from discovery_mission_radar import PROJECT_DIR
from discovery_utils.utils import google

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CATEGORIES = {
    "heat_pumps": "Heat pumps",
    "geothermal_energy": "Geothermal energy", 
    "district_heating": "District heating",
    "heat_storage": "Heat storage"
}

# Chart colors
EARLY_STAGE_COLOR = "#2ea590"
GROWTH_STAGE_COLOR = "blue"

# Output directories
OUTPUT_BASE = PROJECT_DIR / "outputs" / "ASF"
CHARTS_DIR = OUTPUT_BASE / "charts"
CSV_DIR = OUTPUT_BASE / "csv_tables"

# Ensure output directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# --- Load reviewer table from Google Sheets ---
sheet_id = "1m9_tKyJDaSy2vDWxYVP_9HlfBbGysQUV-xrb1FW3vok"
sheet_name = "crunchbase_check_v2"
cb_checks_df = google.access_google_sheet(sheet_id, sheet_name)

def get_european_countries() -> List[str]:
    """Get list of European countries from REGION_TO_COUNTRIES."""
    return crunchbase.REGION_TO_COUNTRIES.get("Europe", [])


def load_and_process_data(category: str) -> Optional[pd.DataFrame]:
    """
    Load and process data for a given category, filtering out rejected org_ids.
    """
    # Construct file path
    file_path = OUTPUT_BASE / category / "crunchbase" / "csv" / f"{CATEGORIES[category]}_funding_rounds.csv"
    
    if not file_path.exists():
        logger.warning(f"Data file not found: {file_path}")
        return None
    
    try:
        # Load and process data
        rounds = (
            pd.read_csv(file_path, low_memory=False)
            .assign(
                region=lambda d: d["country_code"].map(crunchbase.country_to_region()),
                stage=lambda d: d["investment_type"].map(crunchbase.investment_type_to_stage()),
                year=lambda d: pd.to_datetime(d["announced_on"]).dt.year,
            )
        )
        
        # Filter to relevant years
        rounds = rounds.query("2020 <= year <= 2024")
        
        # --- Remove rejected org_ids for this category ---
        reviewer_theme = CATEGORIES[category]
        rejected_ids = set(cb_checks_df[
            (cb_checks_df['theme'] == reviewer_theme) &
            (cb_checks_df['reviewer (karlis)'].str.lower() == 'no')
        ]['id'])
        if rejected_ids:
            rounds = rounds[~rounds['org_id'].isin(rejected_ids)]
            logger.info(f"Removed {len(rejected_ids)} rejected org_ids for {category}")
        
        logger.info(f"Loaded {len(rounds)} funding rounds for {category} after filtering")
        return rounds
        
    except Exception as e:
        logger.error(f"Error loading data for {category}: {e}")
        return None


def create_country_stage_data(rounds: pd.DataFrame) -> pd.DataFrame:
    """Create country-stage aggregated data."""
    return (
        rounds
        .query("stage in ['early_stage', 'growth_stage']")
        .groupby(["country_code", "stage"], as_index=False)
        .agg(gbp=("raised_amount_gbp", "sum"))
        .assign(gbp_m=lambda d: d["gbp"] / 1_000_000)
    )


def create_region_stage_data(rounds: pd.DataFrame) -> pd.DataFrame:
    """Create region-stage aggregated data."""
    return (
        rounds
        .query("stage in ['early_stage', 'growth_stage']")
        .groupby(['region', 'stage'], as_index=False)
        .agg(gbp=('raised_amount_gbp', 'sum'))
        .assign(gbp_m=lambda d: d['gbp'] / 1_000_000)
    )


def plot_regional_investment(region_stage: pd.DataFrame, category: str) -> None:
    """Create and save regional investment chart."""
    region_pivot = region_stage.pivot(index='region', columns='stage', values='gbp_m').fillna(0).reset_index()
    
    # Ensure both stages exist
    if 'early_stage' not in region_pivot.columns:
        region_pivot['early_stage'] = 0
    if 'growth_stage' not in region_pivot.columns:
        region_pivot['growth_stage'] = 0
    
    region_sorted = region_pivot.sort_values('early_stage', ascending=False)
    
    regions = region_sorted['region']
    early_amt = region_sorted['early_stage']
    growth_amt = region_sorted['growth_stage']
    
    plt.figure(figsize=(10, 6))
    plt.bar(regions, early_amt, label='Early-stage', color=EARLY_STAGE_COLOR)
    plt.bar(regions, growth_amt, bottom=early_amt, label='Growth-stage', color=GROWTH_STAGE_COLOR)
    plt.ylabel("Investment (£ billions)")
    plt.title(f"Investment in {CATEGORIES[category]} (2020–2024) per region")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    filename = f"{category}_regional_investment.png"
    plt.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def plot_regional_counts(rounds: pd.DataFrame, category: str) -> None:
    """Create and save regional funding round counts chart."""
    region_stage_counts = (
        rounds
        .query("stage in ['early_stage', 'growth_stage']")
        .groupby(['region', 'stage'])
        .size()
        .reset_index(name='count')
        .pivot(index='region', columns='stage', values='count')
        .fillna(0)
        .reset_index()
    )
    
    # Ensure both stages exist
    if 'early_stage' not in region_stage_counts.columns:
        region_stage_counts['early_stage'] = 0
    if 'growth_stage' not in region_stage_counts.columns:
        region_stage_counts['growth_stage'] = 0
    
    region_counts_sorted = region_stage_counts.assign(
        total=lambda d: d['early_stage'] + d['growth_stage']
    ).sort_values('total', ascending=False)
    
    regions = region_counts_sorted['region']
    early_counts = region_counts_sorted['early_stage']
    growth_counts = region_counts_sorted['growth_stage']
    
    plt.figure(figsize=(10, 6))
    plt.bar(regions, early_counts, label='Early-stage', color=EARLY_STAGE_COLOR)
    plt.bar(regions, growth_counts, bottom=early_counts, label='Growth-stage', color=GROWTH_STAGE_COLOR)
    plt.ylabel("Count of funding rounds")
    plt.title(f"Number of funding rounds (2020–2024) per region\n{CATEGORIES[category]}")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    filename = f"{category}_regional_counts.png"
    plt.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def get_top_european_countries(country_stage: pd.DataFrame, top_n: int = 10) -> List[str]:
    """Get top European countries by total investment."""
    european_countries = get_european_countries()
    
    # Filter to European countries and calculate totals
    european_data = (
        country_stage
        .query("country_code in @european_countries")
        .groupby("country_code")
        .agg(total_gbp_m=("gbp_m", "sum"))
        .sort_values("total_gbp_m", ascending=False)
        .head(top_n)
    )
    
    top_countries = european_data.index.tolist()
    
    # Always include UK if not already in top countries
    if "GBR" not in top_countries:
        top_countries.append("GBR")
    
    return top_countries


def plot_european_comparison(country_stage: pd.DataFrame, category: str) -> None:
    """Create and save European countries comparison chart."""
    top_countries = get_top_european_countries(country_stage)
    
    # Prepare data for plotting
    european_data = (
        country_stage
        .query("country_code in @top_countries")
        .pivot(index="country_code", columns="stage", values="gbp_m")
        .fillna(0)
        .reset_index()
        .rename(columns={"country_code": "country"})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_data.columns:
        european_data['early_stage'] = 0
    if 'growth_stage' not in european_data.columns:
        european_data['growth_stage'] = 0
    
    # Sort by early_stage descending
    european_sorted = european_data.sort_values("early_stage", ascending=False)
    
    countries = european_sorted["country"]
    early_stage = european_sorted["early_stage"]
    growth_stage = european_sorted["growth_stage"]
    
    plt.figure(figsize=(12, 6))
    plt.bar(countries, early_stage, label='Early-stage', color=EARLY_STAGE_COLOR)
    plt.bar(countries, growth_stage, bottom=early_stage, label='Growth-stage', color=GROWTH_STAGE_COLOR)
    plt.ylabel("Investment (£ billions)")
    plt.title(f"Investment in {CATEGORIES[category]} (2020–2024)\nTop European countries")
    plt.xticks(countries, countries)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    filename = f"{category}_european_comparison.png"
    plt.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def plot_european_counts(rounds: pd.DataFrame, country_stage: pd.DataFrame, category: str) -> None:
    """Create and save European countries funding round counts chart."""
    top_countries = get_top_european_countries(country_stage)
    
    european_counts = (
        rounds
        .query("country_code in @top_countries and stage in ['early_stage', 'growth_stage']")
        .groupby(['country_code', 'stage'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={'country_code': 'country'})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_counts.columns:
        european_counts['early_stage'] = 0
    if 'growth_stage' not in european_counts.columns:
        european_counts['growth_stage'] = 0
    
    # Sort by total descending
    european_counts['total'] = european_counts['early_stage'] + european_counts['growth_stage']
    european_counts_sorted = european_counts.sort_values('total', ascending=False)
    
    countries = european_counts_sorted['country']
    early_counts = european_counts_sorted['early_stage']
    growth_counts = european_counts_sorted['growth_stage']
    
    plt.figure(figsize=(12, 6))
    plt.bar(countries, early_counts, label='Early-stage', color=EARLY_STAGE_COLOR)
    plt.bar(countries, growth_counts, bottom=early_counts, label='Growth-stage', color=GROWTH_STAGE_COLOR)
    plt.ylabel("Count of funding rounds")
    plt.title(f"Number of funding rounds (2020–2024)\nTop European countries - {CATEGORIES[category]}")
    plt.xticks(countries, countries)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    filename = f"{category}_european_counts.png"
    plt.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def print_top_european_data(country_stage: pd.DataFrame, rounds: pd.DataFrame, category: str) -> None:
    """Print data tables for top European countries."""
    top_countries = get_top_european_countries(country_stage)
    
    # Investment data
    european_investment = (
        country_stage
        .query("country_code in @top_countries")
        .pivot(index="country_code", columns="stage", values="gbp_m")
        .fillna(0)
        .reset_index()
        .rename(columns={"country_code": "country"})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_investment.columns:
        european_investment['early_stage'] = 0
    if 'growth_stage' not in european_investment.columns:
        european_investment['growth_stage'] = 0
    
    european_investment = european_investment.sort_values("early_stage", ascending=False)
    
    # Counts data
    european_counts = (
        rounds
        .query("country_code in @top_countries and stage in ['early_stage', 'growth_stage']")
        .groupby(['country_code', 'stage'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={'country_code': 'country'})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_counts.columns:
        european_counts['early_stage'] = 0
    if 'growth_stage' not in european_counts.columns:
        european_counts['growth_stage'] = 0
    
    european_counts['total'] = european_counts['early_stage'] + european_counts['growth_stage']
    european_counts_sorted = european_counts.sort_values('total', ascending=False)
    
    print(f"\n=== Top European Countries - {CATEGORIES[category]} ===")
    print("\nInvestment (£ billions):")
    print(european_investment.to_string(index=False))
    print(f"\nNumber of funding rounds:")
    print(european_counts_sorted.to_string(index=False))


def export_csv_data(region_stage: pd.DataFrame, country_stage: pd.DataFrame, 
                   rounds: pd.DataFrame, category: str) -> None:
    """Export data to CSV files."""
    
    # Regional investment data
    region_investment = (
        region_stage
        .pivot(index='region', columns='stage', values='gbp_m')
        .fillna(0)
        .reset_index()
    )
    
    # Ensure both stages exist
    if 'early_stage' not in region_investment.columns:
        region_investment['early_stage'] = 0
    if 'growth_stage' not in region_investment.columns:
        region_investment['growth_stage'] = 0
    
    region_investment.to_csv(CSV_DIR / f"{category}_regional_investment.csv", index=False)
    
    # Regional counts data
    region_counts = (
        rounds
        .query("stage in ['early_stage', 'growth_stage']")
        .groupby(['region', 'stage'])
        .size()
        .reset_index(name='count')
        .pivot(index='region', columns='stage', values='count')
        .fillna(0)
        .reset_index()
    )
    
    # Ensure both stages exist
    if 'early_stage' not in region_counts.columns:
        region_counts['early_stage'] = 0
    if 'growth_stage' not in region_counts.columns:
        region_counts['growth_stage'] = 0
    
    region_counts.to_csv(CSV_DIR / f"{category}_regional_counts.csv", index=False)
    
    # European countries data
    top_countries = get_top_european_countries(country_stage)
    european_investment = (
        country_stage
        .query("country_code in @top_countries")
        .pivot(index="country_code", columns="stage", values="gbp_m")
        .fillna(0)
        .reset_index()
        .rename(columns={"country_code": "country"})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_investment.columns:
        european_investment['early_stage'] = 0
    if 'growth_stage' not in european_investment.columns:
        european_investment['growth_stage'] = 0
    
    european_investment.to_csv(CSV_DIR / f"{category}_european_investment.csv", index=False)
    
    # European counts data
    european_counts = (
        rounds
        .query("country_code in @top_countries and stage in ['early_stage', 'growth_stage']")
        .groupby(['country_code', 'stage'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={'country_code': 'country'})
    )
    
    # Ensure both stages exist
    if 'early_stage' not in european_counts.columns:
        european_counts['early_stage'] = 0
    if 'growth_stage' not in european_counts.columns:
        european_counts['growth_stage'] = 0
    
    european_counts.to_csv(CSV_DIR / f"{category}_european_counts.csv", index=False)
    
    logger.info(f"Exported CSV data for {category}")


def analyze_category(category: str) -> None:
    """Analyze a single category and generate all outputs."""
    logger.info(f"Analyzing category: {category}")
    
    # Load data
    rounds = load_and_process_data(category)
    if rounds is None:
        logger.warning(f"Skipping {category} - no data available")
        return
    
    # Create aggregated data
    country_stage = create_country_stage_data(rounds)
    region_stage = create_region_stage_data(rounds)
    
    # Generate charts
    plot_regional_investment(region_stage, category)
    plot_regional_counts(rounds, category)
    plot_european_comparison(country_stage, category)
    plot_european_counts(rounds, country_stage, category)
    
    # Export CSV data
    export_csv_data(region_stage, country_stage, rounds, category)
    
    # Print top European countries data tables
    print_top_european_data(country_stage, rounds, category)
    
    # Print summary statistics
    total_investment = region_stage['gbp_m'].sum()
    total_rounds = len(rounds.query("stage in ['early_stage', 'growth_stage']"))
    logger.info(f"{category}: £{total_investment:.1f}M total investment, {total_rounds} funding rounds")


def main():
    """Main function to run the analysis for all categories."""
    logger.info("Starting ASF geographical analysis")
    
    # Set matplotlib style for better charts
    plt.style.use('default')
    
    # Process each category
    for category in CATEGORIES:
        try:
            analyze_category(category)
        except Exception as e:
            logger.error(f"Error processing {category}: {e}")
            continue
    
    logger.info("Analysis complete!")
    logger.info(f"Charts saved to: {CHARTS_DIR}")
    logger.info(f"CSV data saved to: {CSV_DIR}")


if __name__ == "__main__":
    main() 