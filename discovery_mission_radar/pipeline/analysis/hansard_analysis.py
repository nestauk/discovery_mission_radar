"""
Hansard Analysis Module.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
import datetime
import re
import altair as alt

from .base import BaseAnalysisModule
from ..config_manager import get_pipeline_config

from discovery_utils.getters import hansard
from discovery_utils.utils import charts, analysis

logger = logging.getLogger(__name__)


class HansardAnalysisModule(BaseAnalysisModule[hansard.HansardGetter]):
    """Hansard analysis module implementation using BaseAnalysisModule ABC."""
    
    def __init__(self, mission: str):
        super().__init__("hansard", mission)
    
    def _create_default_getter(self) -> hansard.HansardGetter:
        """Create default HansardGetter instance."""
        return hansard.HansardGetter()
    
    def _process_topic_data(self, topic_data: Dict[str, Any], getter: hansard.HansardGetter) -> Dict[str, pd.DataFrame]:
        """Process Hansard topic data using existing discovery_utils analysis functions.
        
        Mirrors the existing produce_hansard_stats logic exactly.
        """
        matching_ids = topic_data['ids']
        category_name = topic_data['config']['search_recipe']['category_name']
        config = self.config
        
        try:
            # Get speeches data
            speeches_df = self._get_speeches_data(getter, config)
            
            # Filter to selected speeches and process
            selected_df = (
                speeches_df
                .query("speech_id in @matching_ids")
                .drop_duplicates(subset="speech_id")
                .assign(speech_text_norm=lambda df: df.speech.apply(lambda x: re.sub(r"\s+", " ", x)))
                .drop_duplicates(["speakername", "date", "speech_text_norm"])
            )
            
            # Quarterly analysis
            ts_quarterly_df = (
                selected_df
                .query(config.get_quarterly_analysis_filter())
                .groupby("quarter")
                .agg(speeches=("speech_id", "count"))
                .reset_index()
                .pipe(self._impute_missing_quarters, 
                      min_quarter=config.quarterly_start_quarter, 
                      max_quarter=config.quarterly_end_quarter)
            )
            
            # Yearly analysis
            ts_yearly_df = (
                selected_df
                .query(config.get_yearly_analysis_filter())
                .groupby("year")
                .agg(speeches=("speech_id", "count"))
                .reset_index()
                .assign(year=lambda df: df.year.astype(int))
                .pipe(self._impute_missing_years, 
                      min_year=config.yearly_start_year, 
                      max_year=config.yearly_end_year)
            )
            
            # Growth magnitude calculations
            growth_magnitude_df = (
                analysis.magnitude_growth(ts_yearly_df, 
                                        year_start=config.growth_base_year, 
                                        year_end=config.growth_comparison_year)
                .assign(theme=category_name)
                .reset_index()
                .rename(columns={'index': 'variable'})
            )
            
            # Quarterly growth calculations
            growth_magnitude_quarterly_df = self._calculate_quarterly_growth(ts_quarterly_df, category_name, config)
            
            # Return all analysis results as DataFrames
            return {
                'selected_speeches': selected_df,
                'ts_quarterly': ts_quarterly_df,
                'ts_yearly': ts_yearly_df,
                'growth_magnitude': growth_magnitude_df,
                'growth_magnitude_quarterly': growth_magnitude_quarterly_df
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Hansard data: {e}")
            # Return empty DataFrames on error
            return {
                'selected_speeches': pd.DataFrame(),
                'ts_quarterly': pd.DataFrame(),
                'ts_yearly': pd.DataFrame(),
                'growth_magnitude': pd.DataFrame(),
                'growth_magnitude_quarterly': pd.DataFrame()
            }
    
    def _create_source_charts(self, analysis_results: Dict[str, pd.DataFrame], 
                             charts_dir: Path, category_name: str, scale_factor: int) -> List[str]:
        """Create Hansard-specific charts using discovery_utils.charts.
        
        Mirrors the existing _generate_charts function exactly.
        """
        chart_files = []
        
        try:
            if not analysis_results['ts_quarterly'].empty:
                fig = charts.ts_bar(
                    analysis_results['ts_quarterly'],
                    variable="speeches",
                    variable_title="Number of speeches",
                    time_column="quarter",
                )
                fig = charts.configure_plots(fig, chart_title=f"Number of speeches for {category_name}")
                chart_file = charts_dir / f"hansard_{category_name}_quarterly_speeches.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
        
        except Exception as e:
            self.logger.error(f"Failed to generate Hansard charts: {e}")
        
        return chart_files
    
    def _generate_custom_stats(self, analysis_results: Dict[str, pd.DataFrame], 
                              topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Hansard-specific statistics."""
        selected_df = analysis_results['selected_speeches']
        
        return {
            'speeches_analysed': len(selected_df),
            'unique_speakers': selected_df['speakername'].nunique() if not selected_df.empty else 0,
            'date_range': f"{selected_df['date'].min()} to {selected_df['date'].max()}" if not selected_df.empty else "N/A"
        }
    
    # Helper methods from original implementation
    
    def _get_speeches_data(self, getter, config) -> pd.DataFrame:
        """Get speeches data."""
        missions = [config.current_mission]
        
        speeches_df = (
            getter.get_debates_parquet()
            .query(config.get_date_query_filter())
            .merge(
                getter.get_labelstore(keywords=True)[['id', 'mission_labels', 'topic_labels']],
                left_on='speech_id',
                right_on='id',
                how='left'
            )
            .assign(mission_labels=lambda df: df.mission_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
            .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
            .explode("mission_labels")
            .query("mission_labels in @missions")
            .explode("topic_labels")
            .assign(quarter=lambda df: df.date.apply(self._get_quarter_from_date))
            .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
        )
        
        return speeches_df
    
    def _get_quarter_from_date(self, date):
        """Convert date to quarter."""
        if pd.isna(date):
            return None
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return (date.month - 1) // 3 + 1
    
    def _impute_missing_quarters(self, df, min_quarter, max_quarter):
        """Impute missing quarters with zero values."""
        if df.empty:
            return df
        
        # Generate all quarters in range
        start_year, start_q = map(int, min_quarter.split('-Q'))
        end_year, end_q = map(int, max_quarter.split('-Q'))
        
        all_quarters = []
        year, quarter = start_year, start_q
        while year < end_year or (year == end_year and quarter <= end_q):
            all_quarters.append(f"{year}-Q{quarter}")
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        
        # Create complete dataframe with all quarters
        complete_df = pd.DataFrame({'quarter': all_quarters})
        
        # Merge with existing data, filling missing values with 0
        result_df = complete_df.merge(df, on='quarter', how='left').fillna(0)
        
        return result_df
    
    def _impute_missing_years(self, df, min_year, max_year):
        """Impute missing years with zero values."""
        if df.empty:
            return df
        
        # Generate all years in range
        all_years = list(range(min_year, max_year + 1))
        
        # Create complete dataframe with all years
        complete_df = pd.DataFrame({'year': all_years})
        
        # Merge with existing data, filling missing values with 0
        result_df = complete_df.merge(df, on='year', how='left').fillna(0)
        
        return result_df
    
    def _calculate_quarterly_growth(self, ts_quarterly_df, category_name, config):
        """Calculate quarterly growth analysis."""
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