"""
GTR Analysis Module.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
import datetime
import altair as alt

from .base import BaseAnalysisModule
from ..config_manager import get_pipeline_config

from discovery_utils.getters import gtr
from discovery_utils.utils import analysis_gtr, analysis, charts

logger = logging.getLogger(__name__)


class GtrAnalysisModule(BaseAnalysisModule[gtr.GtrGetter]):
    """GTR analysis module implementation using BaseAnalysisModule ABC."""
    
    def __init__(self, mission: str):
        super().__init__("gtr", mission)
    
    def _create_default_getter(self) -> gtr.GtrGetter:
        """Create default GtrGetter instance."""
        return gtr.GtrGetter(data_version=self.config.get_gtr_data_version())
    
    def _process_topic_data(self, topic_data: Dict[str, Any], getter: gtr.GtrGetter) -> Dict[str, pd.DataFrame]:
        """Process GTR topic data using existing discovery_utils analysis functions.
        
        Mirrors the existing produce_gtr_stats logic exactly.
        """
        matching_ids = topic_data['ids']
        config = self.config
        
        # Get matching projects
        matchings_projects_df = getter.projects_enriched.query("id in @matching_ids")
        
        # Generate yearly timeseries
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
        
        # Growth magnitude calculations
        growth_magnitude_df = (
            analysis.magnitude_growth(ts_df, 
                                    year_start=config.growth_base_year, 
                                    year_end=config.growth_comparison_year)
            .assign(theme=topic_data['config']['search_recipe']['category_name'])
            .reset_index()
            .rename(columns={'index': 'variable'})
        )
        
        # Quarterly analysis
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
            .assign(quarter=lambda df: df.time_period.apply(self._date_to_quarter))
            .query("quarter <= @config.current_quarter")
        )
        
        # Quarterly growth calculations
        growth_magnitude_quarterly_df = self._calculate_quarterly_growth(
            ts_quarterly_df, topic_data['config']['search_recipe']['category_name'], config
        )
        
        # Return all analysis results as DataFrames
        return {
            'matching_projects': matchings_projects_df,
            'ts_yearly': ts_df,
            'ts_quarterly': ts_quarterly_df,
            'growth_magnitude': growth_magnitude_df,
            'growth_magnitude_quarterly': growth_magnitude_quarterly_df
        }
    
    def _create_source_charts(self, analysis_results: Dict[str, pd.DataFrame], 
                             charts_dir: Path, category_name: str, scale_factor: int) -> List[str]:
        """Create GTR-specific charts using discovery_utils.charts.
        
        Mirrors the existing _generate_charts function exactly.
        """
        chart_files = []
        
        try:
            # Yearly charts
            if not analysis_results['ts_yearly'].empty:
                # Number of projects
                fig = charts.ts_bar(
                    analysis_results['ts_yearly'],
                    variable='n_projects',
                    variable_title="Number of projects",
                    category_column="_category",
                )
                fig = charts.configure_plots(fig, chart_title=f"Number of projects for {category_name}")
                chart_file = charts_dir / f"gtr_{category_name}_n_projects.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
                
                # Amount
                fig = charts.ts_bar(
                    analysis_results['ts_yearly'],
                    variable='amount',
                    variable_title="Amount, £ millions",
                    category_column="_category",
                )
                fig = charts.configure_plots(fig, chart_title="")
                chart_file = charts_dir / f"gtr_{category_name}_amount.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            # Quarterly charts
            if not analysis_results['ts_quarterly'].empty:
                # Number of projects quarterly
                fig = charts.ts_bar(
                    analysis_results['ts_quarterly'],
                    variable='n_projects',
                    variable_title="Number of projects",
                    category_column="_category",
                    time_column="quarter"
                )
                fig = charts.configure_plots(fig, chart_title=f"Number of projects for {category_name}")
                chart_file = charts_dir / f"gtr_{category_name}_quarterly_n_projects.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
                
                # Amount quarterly
                fig = charts.ts_bar(
                    analysis_results['ts_quarterly'],
                    variable='amount',
                    variable_title="Amount, £ millions",
                    category_column="_category",
                    time_column="quarter"
                )
                fig = charts.configure_plots(fig, chart_title=f"Amount for {category_name}")
                chart_file = charts_dir / f"gtr_{category_name}_quarterly_amount.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
        
        except Exception as e:
            self.logger.error(f"Failed to generate GTR charts: {e}")
        
        return chart_files
    
    def _generate_custom_stats(self, analysis_results: Dict[str, pd.DataFrame], 
                              topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate GTR-specific statistics."""
        return {
            'projects_analysed': len(analysis_results['matching_projects']),
            'total_funding_millions': analysis_results['ts_yearly']['amount'].sum() 
                                    if not analysis_results['ts_yearly'].empty else 0,
            'average_project_value': analysis_results['matching_projects']['amount'].mean() / 1_000_000
                                   if not analysis_results['matching_projects'].empty else 0
        }
    
    # Helper methods from original implementation
    
    def _date_to_quarter(self, date):
        """Convert date to quarter string format."""
        if pd.isna(date):
            return None
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return f"{date.year}-Q{(date.month - 1) // 3 + 1}"
    
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