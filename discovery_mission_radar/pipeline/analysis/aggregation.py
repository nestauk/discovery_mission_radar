"""
Cross-topic aggregation and radar charts
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.utils import charts
import altair as alt
import logging
import yaml

from .crunchbase_analysis import investment_stages, investment_type_to_stage
from ..config_manager import get_pipeline_config

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
    
    # Get configuration
    config = get_pipeline_config()
    current_mission = config.current_mission
    categories_to_show = config.categories_to_show.get(current_mission, [])
    
    # Create mapping from internal topic names to display names
    topic_name_mapping = _create_topic_display_name_mapping(topic_results, config)
    
    # Combine data from all topics
    all_investment_data = _combine_investment_data(topic_results)
    all_research_data = _combine_research_data(topic_results)
    all_policy_data = _combine_policy_data(topic_results)
    
    # Apply display name mapping
    if not all_investment_data.empty:
        all_investment_data = _apply_display_name_mapping(all_investment_data, topic_name_mapping)
    if not all_research_data.empty:
        all_research_data = _apply_display_name_mapping(all_research_data, topic_name_mapping)
    if not all_policy_data.empty:
        all_policy_data = _apply_display_name_mapping(all_policy_data, topic_name_mapping)
    
    if not all_investment_data.empty and not _is_reconstructed_data(topic_results):
        all_investment_data = _apply_investment_stage_filtering(all_investment_data)
    
    radar_data = _generate_radar_data(all_investment_data, all_research_data, all_policy_data, config)
    
    aggregated_csv = output_dir / "cross_topic_aggregated_data.csv"
    radar_data.to_csv(aggregated_csv, index=False)
    
    # Convert categories_to_show to display names for filtering
    display_categories_to_show = [topic_name_mapping.get(cat, cat) for cat in categories_to_show]
    
    comparison_charts = _generate_comparison_charts(
        all_investment_data, all_research_data, all_policy_data, output_dir, display_categories_to_show, config
    )
    
    logger.info(f"Generated {len(comparison_charts)} cross-topic comparison charts")
    
    return {
        'radar_data': radar_data,
        'comparison_charts': comparison_charts,
        'aggregated_csv': str(aggregated_csv)
    }

def _apply_investment_stage_filtering(investment_df: pd.DataFrame) -> pd.DataFrame:
    """Apply investment stage mapping and filter for early/growth stage"""
    return (
        investment_df
        .assign(investment_stage=lambda df: df.get('investment_type', pd.Series()).map(investment_type_to_stage))
        .query("investment_stage in ['early_stage', 'growth_stage']")
    )

def _is_reconstructed_data(topic_results: List[Dict[str, Any]]) -> bool:
    """
    Check if topic results are reconstructed from CSV files vs fresh analysis.
    Reconstructed data only contains csv_files paths, not full analysis results.
    """
    if not topic_results:
        return False
    
    # Check first topic result - if it only has csv_files, it's reconstructed
    first_result = topic_results[0]
    for source in ['crunchbase', 'gtr', 'hansard']:
        if source in first_result:
            source_data = first_result[source]
            # Reconstructed data only has csv_files, fresh data has stats, etc.
            if isinstance(source_data, dict) and list(source_data.keys()) == ['csv_files']:
                return True
    
    return False

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

def _generate_radar_data(investment_df: pd.DataFrame, research_df: pd.DataFrame, policy_df: pd.DataFrame, config) -> pd.DataFrame:
    """Generate radar chart data structure"""
    radar_data_components = []
    
    if not investment_df.empty:
        investment_radar = (
            investment_df
            .query(f"year >= {config.growth_base_year} and year <= {config.growth_comparison_year}")
            .groupby('theme')
            .agg({
                'raised_amount_gbp_total': 'sum',
                'n_rounds': 'sum'
            })
            .reset_index()
            .assign(data_source='investment')
        )
        radar_data_components.append(investment_radar)
    
    if not research_df.empty:
        research_radar = (
            research_df
            .query(f"year >= {config.growth_base_year} and year <= {config.growth_comparison_year}")
            .groupby('theme')
            .agg({
                'amount': 'sum',
                'n_projects': 'sum'
            })
            .reset_index()
            .assign(data_source='research')
        )
        radar_data_components.append(research_radar)
    
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
                              policy_df: pd.DataFrame, output_dir: Path, categories_to_show, config) -> List[str]:
    """Generate cross-topic comparison charts"""
    chart_files = []
    scale_factor = 2
    
    current_theme = alt.themes.active
    alt.themes.enable('default')
    
    try:
        # Filter data to only include categories we want to show
        if categories_to_show:
            if not investment_df.empty:
                investment_df = investment_df.query("theme in @categories_to_show")
            if not research_df.empty:
                research_df = research_df.query("theme in @categories_to_show")
            if not policy_df.empty:
                policy_df = policy_df.query("theme in @categories_to_show")
        
        # Investment comparison charts
        if not investment_df.empty:
            # Investment trends over time
            fig = charts.ts_smooth(
                ts=investment_df,
                variable="raised_amount_gbp_total",
                variable_title="Raised amount, £ millions",
                category_column="theme",
                categories_to_show=categories_to_show,
                time_column="year"
            )
            fig = charts.configure_plots(fig, 
                                       chart_title="Investment trends across topics",
                                       chart_subtitle="Early and growth stage funding only")
            fig = _configure_text_colors(fig)
            chart_file = output_dir / "cross_topic_investment_trends.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
            
            # Number of rounds comparison
            fig = charts.ts_smooth(
                ts=investment_df,
                variable="n_rounds",
                variable_title="Number of funding rounds",
                category_column="theme",
                categories_to_show=categories_to_show,
                time_column="year"
            )
            fig = charts.configure_plots(fig, 
                                       chart_title="Funding rounds across topics",
                                       chart_subtitle="Early and growth stage funding only")
            fig = _configure_text_colors(fig)
            chart_file = output_dir / "cross_topic_funding_rounds.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
        
        # Research comparison charts
        if not research_df.empty:
            # Research funding trends
            fig = charts.ts_smooth(
                ts=research_df,
                variable="amount",
                variable_title="Research funding, £ millions",
                category_column="theme",
                categories_to_show=categories_to_show,
                time_column="year"
            )
            fig = charts.configure_plots(fig, 
                                       chart_title="Research funding trends across topics",
                                       chart_subtitle="UKRI Gateway to Research data")
            fig = _configure_text_colors(fig)
            chart_file = output_dir / "cross_topic_research_trends.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
            
            # Number of projects comparison
            fig = charts.ts_smooth(
                ts=research_df,
                variable="n_projects",
                variable_title="Number of projects",
                category_column="theme",
                categories_to_show=categories_to_show,
                time_column="year"
            )
            fig = charts.configure_plots(fig, 
                                       chart_title="Research projects across topics",
                                       chart_subtitle="UKRI Gateway to Research data")
            fig = _configure_text_colors(fig)
            chart_file = output_dir / "cross_topic_research_projects.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
        
        # Policy comparison charts
        if not policy_df.empty:
            # Policy attention trends (quarterly)
            fig = charts.ts_bar(
                ts=policy_df,
                variable="speeches",
                variable_title="Number of speeches",
                category_column="theme",
                categories_to_show=categories_to_show,
                time_column="quarter"
            )
            fig = charts.configure_plots(fig, 
                                       chart_title="Policy attention across topics",
                                       chart_subtitle="Number of speeches in House of Commons")
            fig = _configure_text_colors(fig)
            chart_file = output_dir / "cross_topic_policy_trends.png"
            fig.save(str(chart_file), scale_factor=scale_factor)
            chart_files.append(str(chart_file))
            
            # Generate annual policy chart if we have enough data
            annual_policy_df = _aggregate_policy_to_annual(policy_df)
            if not annual_policy_df.empty:
                fig = charts.ts_smooth(
                    ts=annual_policy_df,
                    variable="speeches",
                    variable_title="Number of speeches",
                    category_column="theme",
                    categories_to_show=categories_to_show,
                    time_column="year"
                )
                fig = charts.configure_plots(fig, 
                                           chart_title="Annual policy attention across topics",
                                           chart_subtitle="Number of speeches in House of Commons")
                fig = _configure_text_colors(fig)
                chart_file = output_dir / "cross_topic_policy_trends_annual.png"
                fig.save(str(chart_file), scale_factor=scale_factor)
                chart_files.append(str(chart_file))
        
    finally:
        # Restore the original theme
        alt.themes.enable(current_theme)
    
    return chart_files

def _configure_text_colors(fig):
    """Add explicit text color configuration to ensure visibility in saved charts"""
    return (
        fig
        .configure_axis(
            labelColor="black",
            titleColor="black",
            labelFontSize=13,
            titleFontSize=13,
            labelFont="Arial, sans-serif",
            titleFont="Arial, sans-serif",
            gridColor="grey",
            gridDash=[1, 7]
        )
        .configure_legend(
            labelColor="black",
            titleColor="black", 
            labelFontSize=13,
            titleFontSize=13,
            labelFont="Arial, sans-serif",
            titleFont="Arial, sans-serif"
        )
        .configure_title(
            color="black",
            subtitleColor="black",
            fontSize=16,
            subtitleFontSize=13,
            font="Arial, sans-serif",
            subtitleFont="Arial, sans-serif"
        )
        .configure_view(strokeWidth=0)
        .configure(
            background="white"
        )
    )

def _aggregate_policy_to_annual(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Convert quarterly policy data to annual for trend analysis"""
    if policy_df.empty:
        return pd.DataFrame()
    
    try:
        annual_policy = (
            policy_df
            .assign(year=lambda df: df.quarter.str[:4].astype(int))
            .groupby(['theme', 'year'])
            .agg({'speeches': 'sum'})
            .reset_index()
        )
        return annual_policy
    except Exception as e:
        logger.warning(f"Could not aggregate policy data to annual: {e}")
        return pd.DataFrame()

def _create_topic_display_name_mapping(topic_results: List[Dict[str, Any]], config) -> Dict[str, str]:
    """Create a mapping from internal topic names to display names"""
    topic_name_mapping = {}
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        
        if 'config' in topic_result and 'search_recipe' in topic_result['config']:
            display_name = topic_result['config']['search_recipe'].get('category_name', topic_name)
            topic_name_mapping[topic_name] = display_name
        else:
            try:
                config_dir = Path(__file__).parent.parent.parent / "config"
                topic_config_path = config_dir / "topics" / config.current_mission / f"config_{topic_name}.yaml"
                
                if topic_config_path.exists():
                    with open(topic_config_path, 'r') as f:
                        topic_config = yaml.safe_load(f)
                    display_name = topic_config.get('search_recipe', {}).get('category_name', topic_name)
                    topic_name_mapping[topic_name] = display_name
                else:
                    # Fallback to internal name if no config found
                    topic_name_mapping[topic_name] = topic_name
            except Exception as e:
                logger.warning(f"Could not load config for topic {topic_name}: {e}")
                topic_name_mapping[topic_name] = topic_name
    
    return topic_name_mapping

def _apply_display_name_mapping(df: pd.DataFrame, topic_name_mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply display name mapping to the DataFrame"""
    df['theme'] = df['theme'].map(topic_name_mapping)
    return df 