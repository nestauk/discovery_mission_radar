"""
Cross-topic aggregation and radar charts
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.utils import charts
import logging

logger = logging.getLogger(__name__)

def produce_radar_charts(topic_results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    """
    Generate cross-topic radar charts
    
    Args:
        topic_results: List of results from individual topic analysis
        output_dir: Where to save aggregated analysis and charts
        
    Returns:
        {
            'radar_data': pd.DataFrame,     # Data for radar charts
            'comparison_charts': List[str], # Cross-topic comparison charts
            'aggregated_csv': str          # Path to aggregated data CSV
        }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating cross-topic radar charts for {len(topic_results)} topics")
    
    # Combine data from all topics
    all_investment_data = _combine_investment_data(topic_results)
    all_research_data = _combine_research_data(topic_results)
    all_policy_data = _combine_policy_data(topic_results)
    
    radar_data = _generate_radar_data(all_investment_data, all_research_data, all_policy_data)
    
    aggregated_csv = output_dir / "cross_topic_aggregated_data.csv"
    radar_data.to_csv(aggregated_csv, index=False)
    
    comparison_charts = _generate_comparison_charts(
        all_investment_data, all_research_data, all_policy_data, output_dir
    )
    
    logger.info(f"Generated {len(comparison_charts)} cross-topic comparison charts")
    
    return {
        'radar_data': radar_data,
        'comparison_charts': comparison_charts,
        'aggregated_csv': str(aggregated_csv)
    }

def _combine_investment_data(topic_results: List[Dict]) -> pd.DataFrame:
    """Combine investment data from all topics"""
    all_investment_data = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        if 'crunchbase' in topic_result:
            cb_data = topic_result['crunchbase']
            csv_files = cb_data.get('csv_files', [])
            for csv_file in csv_files:
                if 'ts_startup_yearly' in csv_file:
                    try:
                        ts_df = pd.read_csv(csv_file)
                        ts_df['theme'] = topic_name
                        all_investment_data.append(ts_df)
                    except Exception as e:
                        logger.warning(f"Could not load investment data for {topic_name}: {e}")
    
    if all_investment_data:
        return pd.concat(all_investment_data, ignore_index=True)
    else:
        return pd.DataFrame()

def _combine_research_data(topic_results: List[Dict]) -> pd.DataFrame:
    """Combine research data from all topics"""
    all_research_data = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        if 'gtr' in topic_result:
            gtr_data = topic_result['gtr']
            csv_files = gtr_data.get('csv_files', [])
            for csv_file in csv_files:
                if 'ts_yearly' in csv_file:
                    try:
                        ts_df = pd.read_csv(csv_file)
                        ts_df['theme'] = topic_name
                        all_research_data.append(ts_df)
                    except Exception as e:
                        logger.warning(f"Could not load research data for {topic_name}: {e}")
    
    if all_research_data:
        return pd.concat(all_research_data, ignore_index=True)
    else:
        return pd.DataFrame()

def _combine_policy_data(topic_results: List[Dict]) -> pd.DataFrame:
    """Combine policy data from all topics"""
    all_policy_data = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        if 'hansard' in topic_result:
            hansard_data = topic_result['hansard']
            csv_files = hansard_data.get('csv_files', [])
            for csv_file in csv_files:
                if 'ts_quarterly' in csv_file:
                    try:
                        ts_df = pd.read_csv(csv_file)
                        ts_df['theme'] = topic_name
                        all_policy_data.append(ts_df)
                    except Exception as e:
                        logger.warning(f"Could not load policy data for {topic_name}: {e}")
    
    if all_policy_data:
        return pd.concat(all_policy_data, ignore_index=True)
    else:
        return pd.DataFrame()

def _generate_radar_data(investment_df: pd.DataFrame, research_df: pd.DataFrame, policy_df: pd.DataFrame) -> pd.DataFrame:
    """Generate radar chart data structure"""
    radar_data_components = []
    
    # Investment radar data
    if not investment_df.empty:
        investment_radar = (
            investment_df
            .query("year >= 2020 and year <= 2024")
            .groupby('theme')
            .agg({
                'raised_amount_gbp_total': 'sum',
                'n_rounds': 'sum'
            })
            .reset_index()
            .assign(data_source='investment')
        )
        radar_data_components.append(investment_radar)
    
        # Research radar data
    if not research_df.empty:
        research_radar = (
            research_df
            .query("year >= 2020 and year <= 2024")
            .groupby('theme')
            .agg({
                'amount': 'sum',
                'n_projects': 'sum'
            })
            .reset_index()
            .assign(data_source='research')
        )
        radar_data_components.append(research_radar)
    
    # Policy radar data
    if not policy_df.empty:
        policy_radar = (
            policy_df
            .query("quarter >= '2023-Q1'")
            .groupby('theme')
            .agg({
                'speeches': 'sum'
            })
            .reset_index()
            .assign(data_source='policy')
        )
        radar_data_components.append(policy_radar)
    
    if radar_data_components:
        return pd.concat(radar_data_components, ignore_index=True)
    else:
        return pd.DataFrame()

def _generate_comparison_charts(investment_df: pd.DataFrame, research_df: pd.DataFrame, 
                              policy_df: pd.DataFrame, output_dir: Path) -> List[str]:
    """Generate cross-topic comparison charts"""
    chart_files = []
    scale_factor = 2
    
    # Investment comparison charts
    if not investment_df.empty:
        themes = investment_df['theme'].unique()
        categories_to_show = themes[:6]
        
        # Investment trends over time
        fig = charts.ts_smooth(
            ts=investment_df.query("theme in @categories_to_show"),
            variable="raised_amount_gbp_total",
            variable_title="Raised amount, £ millions",
            category_column="theme",
            categories_to_show=categories_to_show,
            time_column="year"
        )
        fig = charts.configure_plots(fig, chart_title="Investment trends across topics")
        chart_file = output_dir / "cross_topic_investment_trends.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        # Number of rounds comparison
        fig = charts.ts_smooth(
            ts=investment_df.query("theme in @categories_to_show"),
            variable="n_rounds",
            variable_title="Number of funding rounds",
            category_column="theme",
            categories_to_show=categories_to_show,
            time_column="year"
        )
        fig = charts.configure_plots(fig, chart_title="Funding rounds across topics")
        chart_file = output_dir / "cross_topic_funding_rounds.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    # Research comparison charts
    if not research_df.empty:
        themes = research_df['theme'].unique()
        categories_to_show = themes[:6]
        
        # Research funding trends
        fig = charts.ts_smooth(
            ts=research_df.query("theme in @categories_to_show"),
            variable="amount",
            variable_title="Research funding, £ millions",
            category_column="theme",
            categories_to_show=categories_to_show,
            time_column="year"
        )
        fig = charts.configure_plots(fig, chart_title="Research funding trends across topics")
        chart_file = output_dir / "cross_topic_research_trends.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
        
        # Number of projects comparison
        fig = charts.ts_smooth(
            ts=research_df.query("theme in @categories_to_show"),
            variable="n_projects",
            variable_title="Number of projects",
            category_column="theme",
            categories_to_show=categories_to_show,
            time_column="year"
        )
        fig = charts.configure_plots(fig, chart_title="Research projects across topics")
        chart_file = output_dir / "cross_topic_research_projects.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    # Policy comparison charts
    if not policy_df.empty:
        themes = policy_df['theme'].unique()
        categories_to_show = themes[:6]  # Limit to first 6 for readability
        
        # Policy attention trends
        fig = charts.ts_bar(
            ts=policy_df.query("theme in @categories_to_show"),
            variable="speeches",
            variable_title="Number of speeches",
            category_column="theme",
            categories_to_show=categories_to_show,
            time_column="quarter"
        )
        fig = charts.configure_plots(fig, chart_title="Policy attention across topics")
        chart_file = output_dir / "cross_topic_policy_trends.png"
        fig.save(str(chart_file), scale_factor=scale_factor)
        chart_files.append(str(chart_file))
    
    return chart_files 