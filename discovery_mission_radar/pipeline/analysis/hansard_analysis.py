"""
Hansard analysis
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils import charts, analysis
import logging
import datetime
import re
from ..config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

def produce_hansard_stats(topic_data: Dict[str, Any], output_dir: Path, hansard_getter=None) -> Dict[str, Any]:
    """
    Produce Hansard stats
    
    Args:
        topic_data: Output from get_hansard_data() containing IDs and config
        output_dir: Where to save CSVs and charts
        hansard_getter: Pre-initialized HansardGetter (optional)
        
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
    
    logger.info(f"Producing Hansard stats for {category_name} with {len(matching_ids)} speeches")
    
    try:
        if hansard_getter is None:
            HansardData = policy_update.HansardData()
        else:
            HansardData = hansard_getter
        
        speeches_df = _get_speeches_data(HansardData, config)
        
        selected_df = (
            speeches_df
            .query("speech_id in @matching_ids")
            .drop_duplicates(subset="speech_id")
            .assign(speech_text_norm=lambda df: df.speech.apply(lambda x: re.sub(r"\s+", " ", x)))
            .drop_duplicates(["speakername", "date", "speech_text_norm"])
        )
        
        ts_quarterly_df = (
            selected_df
            .query(config.get_quarterly_analysis_filter())
            .groupby("quarter")
            .agg(speeches=("speech_id", "count"))
            .reset_index()
            .pipe(impute_missing_quarters, 
                  min_quarter=config.quarterly_start_quarter, 
                  max_quarter=config.quarterly_end_quarter)
        )
        
        ts_yearly_df = (
            selected_df
            .query(config.get_yearly_analysis_filter())
            .groupby("year")
            .agg(speeches=("speech_id", "count"))
            .reset_index()
            .assign(year=lambda df: df.year.astype(int))
            .pipe(impute_missing_years, 
                  min_year=config.yearly_start_year, 
                  max_year=config.yearly_end_year)
        )
        
        growth_magnitude_df = (
            analysis.magnitude_growth(ts_yearly_df, 
                                    year_start=config.growth_base_year, 
                                    year_end=config.growth_comparison_year)
            .assign(theme=category_name)
            .reset_index()
            .rename(columns={'index': 'variable'})
        )
        
        growth_magnitude_quarterly_df = _calculate_quarterly_growth(ts_quarterly_df, category_name, config)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = output_dir / "csv"
        charts_dir = output_dir / "charts"
        csv_dir.mkdir(exist_ok=True)
        charts_dir.mkdir(exist_ok=True)
        
        csv_files = _save_csv_files({
            'selected_speeches': selected_df,
            'ts_quarterly': ts_quarterly_df,
            'ts_yearly': ts_yearly_df,
            'growth_magnitude': growth_magnitude_df,
            'growth_magnitude_quarterly': growth_magnitude_quarterly_df
        }, csv_dir, category_name)

        chart_files = _generate_charts({
            'ts_quarterly_df': ts_quarterly_df
        }, charts_dir, category_name)
        
        return {
            'csv_files': csv_files,
            'chart_files': chart_files,
            'stats': {
                'speeches_analysed': len(selected_df),
                'unique_speakers': selected_df['speakername'].nunique() if not selected_df.empty else 0,
                'date_range': f"{selected_df['date'].min()} to {selected_df['date'].max()}" if not selected_df.empty else "N/A"
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing Hansard data: {e}")
        return {
            'csv_files': [],
            'chart_files': [],
            'stats': {
                'speeches_analysed': 0,
                'unique_speakers': 0,
                'date_range': "N/A"
            }
        }

def _get_speeches_data(HansardData, config) -> pd.DataFrame:
    """Get speeches data"""
    missions = [config.current_mission]
    
    speeches_df = (
        HansardData.debates_df
        .query(config.get_date_query_filter())
        .merge(
            HansardData.labelstore_df[['id', 'mission_labels', 'topic_labels']],
            left_on='speech_id',
            right_on='id',
            how='left'
        )
        .assign(mission_labels=lambda df: df.mission_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
        .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
        .explode("mission_labels")
        .query("mission_labels in @missions")
        .explode("topic_labels")
        .assign(quarter=lambda df: df.date.apply(get_quarter_from_date))
        .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
    )
    
    return speeches_df

def _generate_charts(data: Dict[str, pd.DataFrame], output_dir: Path, category_name: str) -> List[str]:
    """Generate all charts"""
    chart_files = []
    scale_factor = 2
    
    if not data['ts_quarterly_df'].empty:
        fig = charts.ts_bar(
            data['ts_quarterly_df'],
            variable="speeches",
            variable_title="Number of speeches",
            time_column="quarter",
        )
        fig = charts.configure_plots(fig, chart_title=f"Number of speeches for {category_name}")
        chart_file = output_dir / f"hansard_{category_name}_quarterly_speeches.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    logger.info(f"Generated {len(chart_files)} Hansard charts for {category_name}")
    return chart_files

def get_quarter_from_date(date: str) -> int:
    """Return the quarter number from a given YYYY-MM-DD date string."""
    _date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return (_date.month-1)//3 + 1

def impute_missing_quarters(df, date_col="quarter", value_col="speeches", min_quarter=None, max_quarter=None):
    """Impute missing quarters"""
    df[date_col] = pd.PeriodIndex(df[date_col], freq='Q')
    
    if min_quarter is None:
        min_quarter = df[date_col].min()
    if max_quarter is None:
        max_quarter = df[date_col].max()
    
    full_range = pd.period_range(start=min_quarter, end=max_quarter, freq='Q')
    
    full_df = pd.DataFrame({date_col: full_range})
    
    df = full_df.merge(df, on=date_col, how='left')
    
    df[value_col] = df[value_col].fillna(0).astype(int)
    
    df[date_col] = df[date_col].astype(str)
    df[date_col] = df[date_col].str.replace("Q", "-Q")
    return df

def impute_missing_years(df, year_col="year", value_col="speeches", min_year=None, max_year=None):
    """Impute missing years"""
    df[year_col] = df[year_col].astype(int)
    
    if min_year is None:
        min_year = df[year_col].min()
    if max_year is None:
        max_year = df[year_col].max()
    
    full_range = range(min_year, max_year + 1)
    
    full_df = pd.DataFrame({year_col: list(full_range)})
    
    df = full_df.merge(df, on=year_col, how='left')
    
    df[value_col] = df[value_col].fillna(0).astype(int)
    
    return df

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
            speeches=("speeches", "mean")
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
            speeches=("speeches", "mean")
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
            filename = output_dir / f"hansard_{category_name}_{name}.csv"
            df.to_csv(filename, index=False)
            csv_files.append(str(filename))
            logger.info(f"Saved Hansard {name}: {len(df)} rows to {filename}")
    return csv_files 