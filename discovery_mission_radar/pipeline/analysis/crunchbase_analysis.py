"""
Crunchbase analysis - mirrors kk_03_mission_radar_cb.py produce_stats() exactly
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from discovery_utils.getters import crunchbase
from discovery_utils.utils import analysis_crunchbase, analysis, charts
from discovery_utils.utils.io import remap_dict
import logging
import datetime
from ..config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

investment_stages = {
    'early_stage': ['pre_seed', 'seed', 'angel', 'series_a', 'series_b', 'convertible_note',
                   'equity_crowdfunding', 'product_crowdfunding', 'non_equity_assistance', 'initial_coin_offering'],
    'growth_stage': ['series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j'],
    'late_stage': ['private_equity', 'post_ipo_equity', 'post_ipo_debt', 'post_ipo_secondary', 'secondary_market'],
    'other': ['corporate_round', 'debt_financing', 'grant', 'series_unknown', 'undisclosed']
}
investment_type_to_stage = remap_dict(investment_stages)

def produce_cb_stats(topic_data: Dict[str, Any], output_dir: Path, cb_getter=None) -> Dict[str, Any]:
    """
    Exactly mirrors the notebook produce_stats() function.
    
    Args:
        topic_data: Output from get_cb_data() containing IDs and config
        output_dir: Where to save CSVs and charts
        cb_getter: Pre-initialized CrunchbaseGetter (optional)
        
    Returns:
        {
            'csv_files': List[str],     # Paths to generated CSV files
            'chart_files': List[str],   # Paths to generated charts
            'stats': Dict               # Summary statistics
        }
    """
    # Get configuration
    config = get_pipeline_config()
    
    matching_ids = topic_data['ids']
    category_name = topic_data['config']['search_recipe']['category_name']
    
    logger.info(f"Producing Crunchbase stats for {category_name} with {len(matching_ids)} companies")
    
    if cb_getter is None:
        CB = crunchbase.CrunchbaseGetter()
    else:
        CB = cb_getter
    
    funding_round_types = ["angel", "pre_seed", "seed", "series_a", "series_b"]
    
    matchings_orgs_df = CB.organisations_enriched.query("id in @matching_ids")
    
    funding_rounds_df = CB.select_funding_rounds(
        org_ids=matching_ids, 
        funding_round_types=funding_round_types
    )
    
    investors_df = (
        CB.funding_rounds_enriched
        .query("funding_round_id in @funding_rounds_df.funding_round_id")
        .groupby("funding_round_id")
        .agg(investor_name=("investor_name", list))
        .reset_index()
    )
    
    funding_rounds_df = (
        funding_rounds_df
        .drop(columns=["investor_name"], errors='ignore')
        .merge(investors_df, on="funding_round_id", how="left")
        .assign(investment_stage=lambda df: df.investment_type.map(investment_type_to_stage, na_action=None))
    )
    
    ts_df = analysis_crunchbase.get_timeseries(
        matchings_orgs_df, funding_rounds_df, period='year', 
        min_year=config.yearly_start_year, max_year=config.yearly_end_year
    )
    
    funding_rounds_startup_df = funding_rounds_df.query("investment_stage in ['early_stage', 'growth_stage']")
    ts_startup_df = analysis_crunchbase.get_timeseries(
        matchings_orgs_df, funding_rounds_startup_df, period='year', 
        min_year=config.yearly_start_year, max_year=config.yearly_end_year
    )
    
    growth_rates = analysis.smoothed_growth(ts_startup_df, 
                                          year_start=config.growth_base_year, 
                                          year_end=config.growth_comparison_year)
    growth_rates_df = pd.DataFrame(growth_rates, columns=[category_name]).T.reset_index().rename(columns={'index': 'theme'})
    growth_magnitude_df = (
        analysis.magnitude_growth(ts_startup_df, 
                                year_start=config.growth_base_year, 
                                year_end=config.growth_comparison_year)
        .assign(theme=category_name)
        .reset_index()
        .rename(columns={'index': 'variable'})
    )
    
    quarterly_start_year = int(config.quarterly_start_quarter.split('-')[0])
    quarterly_end_year = int(config.quarterly_end_quarter.split('-')[0])
    
    ts_quarterly_df = (
        analysis_crunchbase.get_timeseries(
            matchings_orgs_df, funding_rounds_df, period='quarter', 
            min_year=quarterly_start_year, max_year=quarterly_end_year
        )
        .assign(quarter=lambda df: df.time_period.apply(date_to_quarter))
        .query("quarter <= @config.current_quarter")
    )
    
    ts_quarterly_startup_df = (
        analysis_crunchbase.get_timeseries(
            matchings_orgs_df, funding_rounds_startup_df, period='quarter', 
            min_year=quarterly_start_year, max_year=quarterly_end_year
        )
        .assign(quarter=lambda df: df.time_period.apply(date_to_quarter))
        .query("quarter <= @config.current_quarter")
    )
    
    growth_magnitude_quarterly_df = _calculate_quarterly_growth(ts_quarterly_startup_df, category_name, config)
    
    aggregated_funding_types_df = aggregate_by_funding_round_types2(
        funding_rounds_df.assign(investment_type=lambda df: df.investment_stage), 
        period="year", min_year=config.yearly_start_year, max_year=config.yearly_end_year
    )
    aggregated_funding_types_startup_df = aggregate_by_funding_round_types2(
        funding_rounds_startup_df.assign(investment_type=lambda df: df.investment_stage), 
        period="year", min_year=config.yearly_start_year, max_year=config.yearly_end_year
    )
    aggregated_funding_types_quarterly_df = aggregate_by_funding_round_types2(
        funding_rounds_df.assign(investment_type=lambda df: df.investment_stage), 
        period="quarter", min_year=quarterly_start_year, max_year=quarterly_end_year
    )
    aggregated_funding_types_quarterly_startup_df = aggregate_by_funding_round_types2(
        funding_rounds_startup_df.assign(investment_type=lambda df: df.investment_stage), 
        period="quarter", min_year=quarterly_start_year, max_year=quarterly_end_year
    )

    ipos_df = CB.ipos.query("org_id in @matching_ids")
    acquisitions_df = CB.acquisitions.query("acquiree_id in @matching_ids")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    charts_dir = output_dir / "charts"
    csv_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)
    
    csv_files = _save_csv_files({
        'matching_orgs': matchings_orgs_df,
        'funding_rounds': funding_rounds_df,
        'ts_yearly': ts_df,
        'ts_startup_yearly': ts_startup_df,
        'ts_quarterly': ts_quarterly_df,
        'ts_quarterly_startup': ts_quarterly_startup_df,
        'growth_rates': growth_rates_df,
        'growth_magnitude': growth_magnitude_df,
        'growth_magnitude_quarterly': growth_magnitude_quarterly_df,
        'aggregated_funding_types': aggregated_funding_types_df,
        'aggregated_funding_types_startup': aggregated_funding_types_startup_df,
        'aggregated_funding_types_quarterly': aggregated_funding_types_quarterly_df,
        'aggregated_funding_types_quarterly_startup': aggregated_funding_types_quarterly_startup_df,
        'ipos': ipos_df,
        'acquisitions': acquisitions_df
    }, csv_dir, category_name)
    
    chart_files = _generate_charts({
        'ts_quarterly_df': ts_quarterly_df,
        'ts_df': ts_df,
        'ts_quarterly_startup_df': ts_quarterly_startup_df,
        'ts_startup_df': ts_startup_df,
        'aggregated_funding_types_df': aggregated_funding_types_df,
        'aggregated_funding_types_startup_df': aggregated_funding_types_startup_df,
        'aggregated_funding_types_quarterly_df': aggregated_funding_types_quarterly_df,
        'aggregated_funding_types_quarterly_startup_df': aggregated_funding_types_quarterly_startup_df
    }, charts_dir, category_name)
    
    return {
        'csv_files': csv_files,
        'chart_files': chart_files,
        'stats': {
            'companies_analysed': len(matchings_orgs_df),
            'funding_rounds': len(funding_rounds_df),
            'total_funding_gbp': funding_rounds_df['raised_amount_gbp'].sum() if not funding_rounds_df.empty else 0,
            'ipos': len(ipos_df),
            'acquisitions': len(acquisitions_df)
        }
    }

def _generate_charts(data: Dict[str, pd.DataFrame], output_dir: Path, category_name: str) -> List[str]:
    """Generate all charts"""
    chart_files = []
    scale_factor = 2
    
    if not data['ts_df'].empty:
        fig = charts.ts_bar(
            data['ts_df'],
            variable='raised_amount_gbp_total',
            variable_title="Raised amount, £ millions",
            category_column="_category"
        )
        fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name}")
        chart_file = output_dir / f"{category_name}_raised_amount.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        fig = charts.ts_bar(
            data['ts_df'],
            variable='n_rounds',
            variable_title="Number of funding rounds",
            category_column="_category"
        )
        fig = charts.configure_plots(fig, chart_title=f"Number of funding rounds for {category_name}")
        chart_file = output_dir / f"{category_name}_n_rounds.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['ts_quarterly_df'].empty:
        fig = charts.ts_bar(
            data['ts_quarterly_df'],
            variable='raised_amount_gbp_total',
            variable_title="Raised amount, £ millions",
            category_column="_category",
            time_column="quarter"
        )
        fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name}")
        chart_file = output_dir / f"{category_name}_raised_amount_quarterly.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['ts_startup_df'].empty:
        fig = charts.ts_bar(
            data['ts_startup_df'],
            variable='raised_amount_gbp_total',
            variable_title="Raised amount, £ millions",
            category_column="_category"
        )
        fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name} (early and growth stage)")
        chart_file = output_dir / f"{category_name}_raised_amount_startup.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['ts_quarterly_startup_df'].empty:
        fig = charts.ts_bar(
            data['ts_quarterly_startup_df'],
            variable='raised_amount_gbp_total',
            variable_title="Raised amount, £ millions",
            category_column="_category",
            time_column="quarter"
        )
        fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name} (early and growth stage)")
        chart_file = output_dir / f"{category_name}_raised_amount_quarterly_startup.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    # Investment breakdown charts
    if not data['aggregated_funding_types_df'].empty:
        fig = chart_investment_types(data['aggregated_funding_types_df'], time_period_col="year")
        fig = charts.configure_plots(fig, chart_title=f"Breakdown of investment types for {category_name}")
        chart_file = output_dir / f"{category_name}_investment_types.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        fig = chart_investment_types_counts(data['aggregated_funding_types_df'], time_period_col="year")
        fig = charts.configure_plots(fig, chart_title=f"Number of investments by type for {category_name}")
        chart_file = output_dir / f"{category_name}_investment_types_counts.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['aggregated_funding_types_quarterly_df'].empty:
        fig = chart_investment_types(data['aggregated_funding_types_quarterly_df'], time_period_col="quarter")
        fig = charts.configure_plots(fig, chart_title=f"Breakdown of investment types for {category_name}")
        chart_file = output_dir / f"{category_name}_quarterly_investment_types.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        fig = chart_investment_types_counts(data['aggregated_funding_types_quarterly_df'], time_period_col="quarter")
        fig = charts.configure_plots(fig, chart_title=f"Number of investments by type for {category_name}")
        chart_file = output_dir / f"{category_name}_quarterly_investment_types_counts.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['aggregated_funding_types_startup_df'].empty:
        fig = chart_investment_types(data['aggregated_funding_types_startup_df'])
        fig = charts.configure_plots(fig, chart_title=f"Breakdown of investment types for {category_name} (startup stage)")
        chart_file = output_dir / f"{category_name}_investment_types_startup.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        fig = chart_investment_types_counts(data['aggregated_funding_types_startup_df'])
        fig = charts.configure_plots(fig, chart_title=f"Number of investments by type for {category_name} (startup stage)")
        chart_file = output_dir / f"{category_name}_investment_types_startup_counts.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    if not data['aggregated_funding_types_quarterly_startup_df'].empty:
        fig = chart_investment_types(data['aggregated_funding_types_quarterly_startup_df'], time_period_col="quarter")
        fig = charts.configure_plots(fig, chart_title=f"Breakdown of investment types for {category_name} (startup stage)")
        chart_file = output_dir / f"{category_name}_quarterly_investment_types_startup.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        fig = chart_investment_types_counts(data['aggregated_funding_types_quarterly_startup_df'], time_period_col="quarter")
        fig = charts.configure_plots(fig, chart_title=f"Number of investments by type for {category_name} (startup stage)")
        chart_file = output_dir / f"{category_name}_quarterly_investment_types_startup_counts.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    logger.info(f"Generated {len(chart_files)} charts for {category_name}")
    return chart_files

def date_to_quarter(date: datetime.datetime) -> str:
    """Convert datetime to quarter string"""
    return f"{date.year}-Q{date.quarter}"

def chart_investment_types_counts(funding_round_types_df: pd.DataFrame, title: str = "", time_period_col: str = "year"):
    """Create a bar chart of the number of funding rounds by types"""
    fig = _chart_investment_types(
        funding_round_types_df,
        value_column="counts",
        value_label="Number of funding rounds",
        time_period_col=time_period_col,
    )
    fig = charts.configure_titles(fig, title)
    return fig

def chart_investment_types(funding_round_types_df: pd.DataFrame, title: str = "", time_period_col: str = "year"):
    """Create a bar chart of investment amounts by type"""
    fig = _chart_investment_types(
        funding_round_types_df.assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1e3),
        value_column="raised_amount_gbp",
        value_label="Raised amount (£ millions)",
        time_period_col=time_period_col,
    )
    fig = charts.configure_titles(fig, title)
    return fig

def _chart_investment_types(
    funding_round_types_df: pd.DataFrame,
    value_column: str = "raised_amount_gbp",
    value_label: str = "Raised amount (£ millions)",
    colour_column: str = "investment_type",
    colour_label: str = "Investment type",
    time_period_col: str = "year",
    stack_order: str = None,
):
    """Create a bar chart of investment types"""
    import altair as alt
    
    if stack_order is None:
        stack_order = colour_column
    fig = (
        alt.Chart(
            funding_round_types_df,
            width=500,
            height=300,
        )
        .mark_bar()
        .encode(
            x=alt.X(f"{time_period_col}:O", title=""),
            y=alt.Y(f"{value_column}:Q", title=value_label),
            color=alt.Color(
                f"{colour_column}:N",
                title=colour_label,
            ),
            order=alt.Order(stack_order),
            tooltip=[time_period_col, colour_column, value_column],
        )
    )
    return charts.configure_plots(fig)

def aggregate_by_funding_round_types2(funding_rounds_df: pd.DataFrame, min_year: int, max_year: int, period: str = "year") -> pd.DataFrame:
    """Aggregate investment data by funding round type and period"""
    period_key = period[0].capitalize()
    grouped = (
        funding_rounds_df
        .drop_duplicates(["funding_round_id"])
        .assign(time_period=lambda df: pd.to_datetime(df.announced_on))
        .assign(time_period=lambda df: df.time_period.dt.to_period(period_key))
        .groupby(["time_period", "investment_type"])
        .agg(
            counts=("funding_round_id", "count"),
            raised_amount_usd=("raised_amount_usd", "sum"),
            raised_amount_gbp=("raised_amount_gbp", "sum"),
        )
        .reset_index()
        .assign(time_period=lambda df: df.time_period.astype("datetime64[ns]"))
        .pipe(analysis.impute_empty_periods, "time_period", period=period_key, min_year=min_year, max_year=max_year)
    )
    
    if period == "year":
        grouped['year'] = grouped['time_period'].dt.year
    elif period == "quarter":
        grouped['quarter'] = grouped['time_period'].apply(lambda x: f"{x.year}-Q{x.quarter}")
    
    return grouped

def _calculate_quarterly_growth(ts_quarterly_startup_df, category_name, config):
    """Calculate quarterly growth analysis"""
    if ts_quarterly_startup_df.empty:
        return pd.DataFrame()
    
    current_quarter = config.current_quarter
    
    previous_four_quarters = ts_quarterly_startup_df.query("quarter < @current_quarter").sort_values("quarter").tail(4).quarter.tolist()
    
    if len(previous_four_quarters) < 4:
        return pd.DataFrame()
    
    previous_four_quarters_mean_df = (
        ts_quarterly_startup_df
        .query("quarter in @previous_four_quarters")
        .assign(_col="previous_four_quarters")
        .groupby("_col")
        .agg(
            raised_amount_gbp_total=("raised_amount_gbp_total", "mean"),
            n_rounds=("n_rounds", "mean"),
            n_orgs_founded=("n_orgs_founded", "mean")
        )
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    
    present_quarter_df = (
        ts_quarterly_startup_df.query("quarter == @current_quarter")
        .assign(_col="magnitude")
        .groupby("_col")
        .agg(
            raised_amount_gbp_total=("raised_amount_gbp_total", "mean"),
            n_rounds=("n_rounds", "mean"),
            n_orgs_founded=("n_orgs_founded", "mean")
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

def _save_csv_files(dataframes: Dict[str, pd.DataFrame], output_dir: Path, category_name: str) -> List[str]:
    """Save all dataframes as CSV files"""
    csv_files = []
    for name, df in dataframes.items():
        if not df.empty:
            filename = output_dir / f"{category_name}_{name}.csv"
            df.to_csv(filename, index=False)
            csv_files.append(str(filename))
            logger.info(f"Saved {name}: {len(df)} rows to {filename}")
    return csv_files 