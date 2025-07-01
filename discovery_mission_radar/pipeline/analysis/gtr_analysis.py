"""
GTR Analysis

Mirrors notebook process exactly: receive topic data → produce charts + csv files.
Simple, stateless functions for analysis.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from discovery_utils.getters import gtr
from discovery_utils.utils import analysis_gtr, analysis, charts
import altair as alt
import logging
import datetime
from ..config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

def produce_gtr_stats(topic_data: Dict[str, Any], output_dir: Path, gtr_getter=None) -> Dict[str, Any]:
    """
    Process GTR project IDs through analysis functions to produce charts and CSVs.
    
    Mirrors the GTR notebook process:
    Get relevant project IDs → Analysis functions → Charts + CSVs
    
    Args:
        topic_data: Output from GtrDataSource.get_data() containing IDs and config
        output_dir: Directory to save charts and CSV files
        gtr_getter: Pre-initialized GtrGetter (optional)
        
    Returns:
        Dictionary with file paths and statistics
    """
    # Get configuration
    config = get_pipeline_config()
    
    matching_ids = topic_data['ids']
    category_name = topic_data['config']['search_recipe']['category_name']
    
    logger.info(f"Producing GTR stats for {category_name} with {len(matching_ids)} projects")
    
    if gtr_getter is None:
        GTR = gtr.GtrGetter(data_version=config.get_gtr_data_version())
    else:
        GTR = gtr_getter
    
    matchings_projects_df = GTR.projects_enriched.query("id in @matching_ids")
    
    ts_df = (
        analysis_gtr.get_timeseries(
            matchings_projects_df, 
            period='year', 
            min_year=config.yearly_start_year, 
            max_year=config.yearly_end_year, 
            description_column="abstractText"
        )
        .assign(amount=lambda df: df.amount / 1_000_000)  # Convert to millions
    )
    
    growth_magnitude_df = (
        analysis.magnitude_growth(ts_df, 
                                year_start=config.growth_base_year, 
                                year_end=config.growth_comparison_year)
        .assign(theme=category_name)
        .reset_index()
        .rename(columns={'index': 'variable'})
    )
    
    quarterly_start_year = int(config.quarterly_start_quarter.split('-')[0])
    quarterly_end_year = int(config.quarterly_end_quarter.split('-')[0])
    
    ts_quarterly_df = (
        analysis_gtr.get_timeseries(
            matchings_projects_df,
            period='quarter',
            min_year=quarterly_start_year,
            max_year=quarterly_end_year,
            description_column="abstractText"
        )
        .assign(amount=lambda df: df.amount / 1_000_000)
        .assign(quarter=lambda df: df.time_period.apply(date_to_quarter))
        .query("quarter <= @config.current_quarter")
    )
    
    growth_magnitude_quarterly_df = _calculate_quarterly_growth(ts_quarterly_df, category_name, config)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    charts_dir = output_dir / "charts"
    csv_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)
    
    csv_files = _save_csv_files({
        'matching_projects': matchings_projects_df,
        'ts_yearly': ts_df,
        'ts_quarterly': ts_quarterly_df,
        'growth_magnitude': growth_magnitude_df,
        'growth_magnitude_quarterly': growth_magnitude_quarterly_df
    }, csv_dir, category_name)
    
    chart_files = _generate_charts({
        'ts_df': ts_df,
        'ts_quarterly_df': ts_quarterly_df
    }, charts_dir, category_name)
    
    return {
        'csv_files': csv_files,
        'chart_files': chart_files,
        'stats': {
            'projects_analysed': len(matchings_projects_df),
            'total_amount_gbp': ts_df['amount'].sum() if not ts_df.empty else 0,
            'avg_project_amount': matchings_projects_df['amount'].mean() if not matchings_projects_df.empty else 0
        }
    }

def _generate_charts(data: Dict[str, pd.DataFrame], output_dir: Path, category_name: str) -> List[str]:
    """Generate all charts"""
    chart_files = []
    scale_factor = 2
    
    current_theme = alt.themes.active
    alt.themes.enable('default')
    
    try:
        if not data['ts_df'].empty:
            fig = charts.ts_bar(
                data['ts_df'],
                variable='n_projects',
                variable_title="Number of projects",
                category_column="_category",
            )
            fig = charts.configure_plots(fig, chart_title=f"Number of projects for {category_name}")
            chart_file = output_dir / f"gtr_{category_name}_n_projects.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
            
            fig = charts.ts_bar(
                data['ts_df'],
                variable='amount',
                variable_title="Amount, £ millions",
                category_column="_category",
            )
            fig = charts.configure_plots(fig, chart_title="")
            chart_file = output_dir / f"gtr_{category_name}_amount.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
        
        if not data['ts_quarterly_df'].empty:
            fig = charts.ts_bar(
                data['ts_quarterly_df'],
                variable='n_projects',
                variable_title="Number of projects",
                category_column="_category",
                time_column="quarter"
            )
            fig = charts.configure_plots(fig, chart_title=f"Number of projects for {category_name}")
            chart_file = output_dir / f"gtr_{category_name}_quarterly_n_projects.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
            
            fig = charts.ts_bar(
                data['ts_quarterly_df'],
                variable='amount',
                variable_title="Amount, £ millions",
                category_column="_category",
                time_column="quarter"
            )
            fig = charts.configure_plots(fig, chart_title=f"Amount for {category_name}")
            chart_file = output_dir / f"gtr_{category_name}_quarterly_amount.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
    
    finally:
        alt.themes.enable(current_theme)
    
    logger.info(f"Generated {len(chart_files)} GTR charts for {category_name}")
    return chart_files

def date_to_quarter(date: datetime.datetime) -> str:
    """Convert datetime to quarter string"""
    return f"{date.year}-Q{date.quarter}"

def _save_csv_files(dataframes: Dict[str, pd.DataFrame], output_dir: Path, category_name: str) -> List[str]:
    """Save all dataframes as CSV files"""
    csv_files = []
    for name, df in dataframes.items():
        if not df.empty:
            filename = output_dir / f"gtr_{category_name}_{name}.csv"
            df.to_csv(filename, index=False)
            csv_files.append(str(filename))
            logger.info(f"Saved GTR {name}: {len(df)} rows to {filename}")
    return csv_files

def _calculate_quarterly_growth(ts_quarterly_df, category_name, config):
    """Calculate quarterly growth analysis"""
    if ts_quarterly_df.empty:
        return pd.DataFrame()
    
    current_quarter = config.current_quarter
    
    previous_four_quarters = ts_quarterly_df.query("quarter < @current_quarter").sort_values("quarter").tail(4).quarter.tolist()
    
    if len(previous_four_quarters) < 4:
        return pd.DataFrame()
    
    previous_four_quarters_mean_df = (
        ts_quarterly_df
        .query("quarter in @previous_four_quarters")
        .assign(_col="previous_four_quarters")
        .groupby("_col")
        .agg(
            amount=("amount", "mean"),
            n_projects=("n_projects", "mean"),
        )
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    
    present_quarter_df = (
        ts_quarterly_df.query("quarter == @current_quarter")
        .assign(_col="magnitude")
        .groupby("_col")
        .agg(
            amount=("amount", "mean"),
            n_projects=("n_projects", "mean"),
        )
        .T.reset_index().rename(columns={"index": "variable"})
    )
    
    growth_magnitude_quarterly_df = (
        previous_four_quarters_mean_df
        .merge(present_quarter_df, on="variable", how="left")
        .assign(growth=lambda df: (df.magnitude - df.previous_four_quarters) / df.previous_four_quarters * 100)
        .assign(theme=category_name)
    )
    
    return growth_magnitude_quarterly_df 