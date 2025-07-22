"""
Crunchbase Analysis Module.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
import datetime
import altair as alt

from .base import BaseAnalysisModule
from ..config_manager import get_pipeline_config

from discovery_utils.getters import crunchbase as cb
from discovery_utils.utils import analysis_crunchbase, analysis, charts
from discovery_utils.utils.io import remap_dict

logger = logging.getLogger(__name__)

investment_stages = {
    'early_stage': ['pre_seed', 'seed', 'angel', 'series_a', 'series_b', 'convertible_note',
                   'equity_crowdfunding', 'product_crowdfunding', 'non_equity_assistance', 'initial_coin_offering'],
    'growth_stage': ['series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j'],
    'late_stage': ['private_equity', 'post_ipo_equity', 'post_ipo_debt', 'post_ipo_secondary', 'secondary_market'],
    'other': ['corporate_round', 'debt_financing', 'grant', 'series_unknown', 'undisclosed']
}
investment_type_to_stage = remap_dict(investment_stages)


class CrunchbaseAnalysisModule(BaseAnalysisModule[cb.CrunchbaseGetter]):
    """Crunchbase analysis module implementation using BaseAnalysisModule ABC."""
    
    def __init__(self, mission: str):
        super().__init__("crunchbase", mission)
    
    def _create_default_getter(self) -> cb.CrunchbaseGetter:
        """Create default CrunchbaseGetter instance."""
        return cb.CrunchbaseGetter()
    
    def _process_topic_data(self, topic_data: Dict[str, Any], getter: cb.CrunchbaseGetter) -> Dict[str, pd.DataFrame]:
        """Process Crunchbase topic data using existing discovery_utils analysis functions.
        
        Mirrors the existing produce_cb_stats logic exactly.
        """
        matching_ids = topic_data['ids']
        config = self.config
        
        # Core funding round types
        funding_round_types = ["angel", "pre_seed", "seed", "series_a", "series_b"]
        
        # Get matching organisations
        matchings_orgs_df = getter.organisations_enriched.query("id in @matching_ids")
        
        # Get funding rounds
        funding_rounds_df = getter.select_funding_rounds(
            org_ids=matching_ids, 
            funding_round_types=funding_round_types
        )
        
        # Get investors for funding rounds
        investors_df = (
            getter.funding_rounds_enriched
            .query("funding_round_id in @funding_rounds_df.funding_round_id")
            .groupby("funding_round_id")
            .agg(investor_name=("investor_name", list))
            .reset_index()
        )
        
        # Merge funding rounds with investors
        funding_rounds_df = (
            funding_rounds_df
            .drop(columns=["investor_name"], errors='ignore')
            .merge(investors_df, on="funding_round_id", how="left")
            .assign(investment_stage=lambda df: df.investment_type.map(investment_type_to_stage, na_action=None))
        )
        
        # Generate yearly timeseries
        ts_df = analysis_crunchbase.get_timeseries(
            matchings_orgs_df, funding_rounds_df, period='year', 
            min_year=config.yearly_start_year, max_year=config.yearly_end_year
        )
        
        # Generate startup-focused timeseries (early + growth stage)
        funding_rounds_startup_df = funding_rounds_df.query("investment_stage in ['early_stage', 'growth_stage']")
        ts_startup_df = analysis_crunchbase.get_timeseries(
            matchings_orgs_df, funding_rounds_startup_df, period='year', 
            min_year=config.yearly_start_year, max_year=config.yearly_end_year
        )
        
        # Growth calculations
        growth_rates = analysis.smoothed_growth(ts_startup_df, 
                                              year_start=config.growth_base_year, 
                                              year_end=config.growth_comparison_year)
        growth_rates_df = pd.DataFrame(growth_rates, columns=[topic_data['config']['search_recipe']['category_name']]).T.reset_index().rename(columns={'index': 'theme'})
        
        growth_magnitude_df = (
            analysis.magnitude_growth(ts_startup_df, 
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
            analysis_crunchbase.get_timeseries(
                matchings_orgs_df, funding_rounds_df, period='quarter', 
                min_year=quarterly_start_year, max_year=quarterly_end_year
            )
            .assign(quarter=lambda df: df.time_period.apply(self._date_to_quarter))
            .query("quarter <= @config.current_quarter")
        )
        
        ts_quarterly_startup_df = (
            analysis_crunchbase.get_timeseries(
                matchings_orgs_df, funding_rounds_startup_df, period='quarter', 
                min_year=quarterly_start_year, max_year=quarterly_end_year
            )
            .assign(quarter=lambda df: df.time_period.apply(self._date_to_quarter))
            .query("quarter <= @config.current_quarter")
        )
        
        # Quarterly growth calculations
        growth_magnitude_quarterly_df = self._calculate_quarterly_growth(
            ts_quarterly_startup_df, topic_data['config']['search_recipe']['category_name'], config
        )
        
        # Funding type aggregations
        aggregated_funding_types_df = self._aggregate_by_funding_round_types(
            funding_rounds_df.assign(investment_type=lambda df: df.investment_stage), 
            period="year", min_year=config.yearly_start_year, max_year=config.yearly_end_year
        )
        
        aggregated_funding_types_startup_df = self._aggregate_by_funding_round_types(
            funding_rounds_startup_df.assign(investment_type=lambda df: df.investment_stage), 
            period="year", min_year=config.yearly_start_year, max_year=config.yearly_end_year
        )
        
        aggregated_funding_types_quarterly_df = self._aggregate_by_funding_round_types(
            funding_rounds_df.assign(investment_type=lambda df: df.investment_stage), 
            period="quarter", min_year=quarterly_start_year, max_year=quarterly_end_year
        )
        
        aggregated_funding_types_quarterly_startup_df = self._aggregate_by_funding_round_types(
            funding_rounds_startup_df.assign(investment_type=lambda df: df.investment_stage), 
            period="quarter", min_year=quarterly_start_year, max_year=quarterly_end_year
        )
        
        # IPOs and acquisitions
        ipos_df = getter.ipos.query("org_id in @matching_ids")
        acquisitions_df = getter.acquisitions.query("acquiree_id in @matching_ids")
        
        # Return all analysis results as DataFrames
        return {
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
        }
    
    def _create_source_charts(self, analysis_results: Dict[str, pd.DataFrame], 
                             charts_dir: Path, category_name: str, scale_factor: int) -> List[str]:
        """Create Crunchbase-specific charts using discovery_utils.charts.
        
        Mirrors the existing _generate_charts function exactly.
        """
        chart_files = []
        
        try:
            # Yearly charts
            if not analysis_results['ts_yearly'].empty:
                # Funding raised over time
                fig = charts.ts_bar(
                    analysis_results['ts_yearly'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="_category"
                )
                fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name}")
                chart_file = charts_dir / f"{category_name}_raised_amount.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
                
                # Number of funding rounds
                fig = charts.ts_bar(
                    analysis_results['ts_yearly'],
                    variable='n_rounds',
                    variable_title="Number of funding rounds",
                    category_column="_category"
                )
                fig = charts.configure_plots(fig, chart_title=f"Number of funding rounds for {category_name}")
                chart_file = charts_dir / f"{category_name}_n_rounds.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            # Quarterly charts
            if not analysis_results['ts_quarterly'].empty:
                fig = charts.ts_bar(
                    analysis_results['ts_quarterly'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="_category",
                    time_column="quarter"
                )
                fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name}")
                chart_file = charts_dir / f"{category_name}_raised_amount_quarterly.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            # Startup-focused charts
            if not analysis_results['ts_startup_yearly'].empty:
                fig = charts.ts_bar(
                    analysis_results['ts_startup_yearly'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="_category"
                )
                fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name} (early and growth stage)")
                chart_file = charts_dir / f"{category_name}_raised_amount_startup.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            if not analysis_results['ts_quarterly_startup'].empty:
                fig = charts.ts_bar(
                    analysis_results['ts_quarterly_startup'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="_category",
                    time_column="quarter"
                )
                fig = charts.configure_plots(fig, chart_title=f"Funding raised over time for {category_name} (early and growth stage)")
                chart_file = charts_dir / f"{category_name}_raised_amount_quarterly_startup.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            # Investment breakdown charts
            if not analysis_results['aggregated_funding_types'].empty:
                fig = charts.ts_bar(
                    analysis_results['aggregated_funding_types'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="investment_type"
                )
                fig = charts.configure_plots(fig, chart_title=f"Investment breakdown for {category_name}")
                chart_file = charts_dir / f"{category_name}_investment_breakdown.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
            
            if not analysis_results['aggregated_funding_types_quarterly'].empty:
                fig = charts.ts_bar(
                    analysis_results['aggregated_funding_types_quarterly'],
                    variable='raised_amount_gbp_total',
                    variable_title="Raised amount, £ millions",
                    category_column="investment_type",
                    time_column="quarter"
                )
                fig = charts.configure_plots(fig, chart_title=f"Investment breakdown for {category_name}")
                chart_file = charts_dir / f"{category_name}_investment_breakdown_quarterly.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
        
        except Exception as e:
            self.logger.error(f"Failed to generate Crunchbase charts: {e}")
        
        return chart_files
    
    def _generate_custom_stats(self, analysis_results: Dict[str, pd.DataFrame], 
                              topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Crunchbase-specific statistics."""
        return {
            'companies_analysed': len(analysis_results['matching_orgs']),
            'funding_rounds': len(analysis_results['funding_rounds']),
            'total_funding_gbp': analysis_results['funding_rounds']['raised_amount_gbp'].sum() 
                               if not analysis_results['funding_rounds'].empty else 0,
            'ipos': len(analysis_results['ipos']),
            'acquisitions': len(analysis_results['acquisitions'])
        }
    
    # Helper methods from original implementation
    
    def _date_to_quarter(self, date):
        """Convert date to quarter string format."""
        if pd.isna(date):
            return None
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return f"{date.year}-Q{(date.month - 1) // 3 + 1}"
    
    def _calculate_quarterly_growth(self, ts_quarterly_startup_df, category_name, config):
        """Calculate quarterly growth analysis."""
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
    
    def _aggregate_by_funding_round_types(self, funding_rounds_df, period="year", min_year=2018, max_year=2024):
        """Aggregate funding rounds by investment type."""
        if funding_rounds_df.empty:
            return pd.DataFrame()
        
        # Restore original logic: create time_period from announced_on
        period_key = period[0].capitalize()  # 'Y' or 'Q'
        df = funding_rounds_df.drop_duplicates(["funding_round_id"]).copy()
        df["time_period"] = pd.to_datetime(df["announced_on"])
        df["time_period"] = df["time_period"].dt.to_period(period_key)
        
        # Impute empty periods if needed (optional, as in original)
        # df = analysis.impute_empty_periods(df, "time_period", period=period_key, min_year=min_year, max_year=max_year)
        
        if period == "year":
            df["year"] = df["time_period"].dt.year
            time_col = "year"
        elif period == "quarter":
            df["quarter"] = df["time_period"].apply(self._date_to_quarter)
            time_col = "quarter"
        else:
            time_col = "time_period"
        
        aggregated_df = (
            df
            .groupby([time_col, 'investment_type'])
            .agg(
                raised_amount_gbp_total=('raised_amount_gbp', 'sum'),
                n_rounds=('funding_round_id', 'count')
            )
            .reset_index()
        )
        
        return aggregated_df 