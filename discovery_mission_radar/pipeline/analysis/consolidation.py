"""
Consolidation module for generating final consolidated CSV files and uploading to Google Sheets
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

from discovery_utils.utils import google
from discovery_mission_radar.pipeline.config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

def consolidate_all_topics(topic_results: List[Dict[str, Any]], output_dir: Path, mission: str) -> Dict[str, str]:
    """
    Consolidate all topic results into final CSV files
    
    Args:
        topic_results: List of results from individual topic analysis
        output_dir: Where to save consolidated CSV files
        
    Returns:
        Dictionary mapping CSV file names to their paths
    """
    config = get_pipeline_config(mission)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Consolidating data from {len(topic_results)} topics")
    
    consolidated_files = {}
    
    cb_stats, cb_stats_quarterly, cb_companies, cb_funding, cb_ipos, cb_acquisitions = _consolidate_crunchbase_data(topic_results, mission)
    
    ukri_stats, ukri_stats_quarterly, ukri_projects = _consolidate_gtr_data(topic_results, mission)
    
    hansard_stats, hansard_stats_quarterly, hansard_speeches = _consolidate_hansard_data(topic_results, mission)
    
    # Define data to save with their corresponding filenames
    data_to_save = [
        (cb_stats, "crunchbase_stats.csv"),
        (cb_stats_quarterly, "crunchbase_stats_quarterly.csv"),
        (cb_companies, "crunchbase_companies.csv"),
        (cb_funding, "crunchbase_funding.csv"),
        (cb_ipos, "crunchbase_ipos.csv"),
        (cb_acquisitions, "crunchbase_acquisitions.csv"),
        (ukri_stats, "ukri_stats.csv"),
        (ukri_stats_quarterly, "ukri_stats_quarterly.csv"),
        (ukri_projects, "ukri_projects.csv"),
        (hansard_stats, "hansard_stats.csv"),
        (hansard_stats_quarterly, "hansard_stats_quarterly.csv"),
        (hansard_speeches, "hansard_speeches.csv")
    ]
    
    # Save each non-empty dataframe
    for df, filename in data_to_save:
        if not df.empty:
            file_path = output_dir / filename
            df.to_csv(file_path, index=False)
            consolidated_files[filename.replace('.csv', '')] = str(file_path)
            logger.info(f"Saved {filename} with {len(df)} rows")
    logger.info(f"Consolidation complete. Generated {len(consolidated_files)} files")
    
    # Upload to Google Sheets if enabled
    if config.google_sheets_enabled and config.upload_aggregated_data and config.google_sheets_id:
        try:
            _upload_to_google_sheets(
                cb_stats, cb_stats_quarterly, cb_companies, cb_funding, cb_ipos, cb_acquisitions,
                ukri_stats, ukri_stats_quarterly, ukri_projects,
                hansard_stats, hansard_stats_quarterly, hansard_speeches,
                config.google_sheets_id
            )
        except Exception as e:
            logger.error(f"Failed to upload to Google Sheets: {e}")
            # Don't fail the entire consolidation if Google Sheets upload fails
    
    return consolidated_files

def _consolidate_crunchbase_data(topic_results: List[Dict[str, Any]], mission: str) -> tuple:
    """Consolidate Crunchbase data from all topics"""
    all_stats = []
    all_stats_quarterly = []
    all_companies = []
    all_funding = []
    all_ipos = []
    all_acquisitions = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        
        if 'crunchbase' in topic_result:
            cb_data = topic_result['crunchbase']
            csv_files = cb_data.get('csv_files', [])
            
            for csv_file in csv_files:
                try:
                    if 'growth_magnitude' in csv_file and 'quarterly' not in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats.append(df)
                    
                    elif 'growth_magnitude_quarterly' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats_quarterly.append(df)
                    
                    elif 'matching_orgs' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_companies.append(df)
                    
                    elif 'funding_rounds' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_funding.append(df)
                    
                    elif 'ipos' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_ipos.append(df)
                    
                    elif 'acquisitions' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_acquisitions.append(df)
                        
                except Exception as e:
                    logger.warning(f"Could not load CB file {csv_file} for {topic_name}: {e}")
    
    return (
        pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        pd.concat(all_stats_quarterly, ignore_index=True) if all_stats_quarterly else pd.DataFrame(),
        pd.concat(all_companies, ignore_index=True) if all_companies else pd.DataFrame(),
        pd.concat(all_funding, ignore_index=True) if all_funding else pd.DataFrame(),
        pd.concat(all_ipos, ignore_index=True) if all_ipos else pd.DataFrame(),
        pd.concat(all_acquisitions, ignore_index=True) if all_acquisitions else pd.DataFrame()
    )

def _consolidate_gtr_data(topic_results: List[Dict[str, Any]], mission: str) -> tuple:
    """Consolidate GTR (UKRI) data from all topics"""
    all_stats = []
    all_stats_quarterly = []
    all_projects = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        
        if 'gtr' in topic_result:
            gtr_data = topic_result['gtr']
            csv_files = gtr_data.get('csv_files', [])
            
            for csv_file in csv_files:
                if 'growth_magnitude' in csv_file and 'quarterly' not in csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load GTR stats for {topic_name}: {e}")
                
                elif 'growth_magnitude_quarterly' in csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats_quarterly.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load GTR stats quarterly for {topic_name}: {e}")
                
                elif 'matching_projects' in csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_projects.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load GTR projects for {topic_name}: {e}")
    
    return (
        pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        pd.concat(all_stats_quarterly, ignore_index=True) if all_stats_quarterly else pd.DataFrame(),
        pd.concat(all_projects, ignore_index=True) if all_projects else pd.DataFrame()
    )

def _consolidate_hansard_data(topic_results: List[Dict[str, Any]], mission: str) -> tuple:
    """Consolidate Hansard data from all topics"""
    all_stats = []
    all_stats_quarterly = []
    all_speeches = []
    
    for topic_result in topic_results:
        topic_name = topic_result['topic_name']
        
        if 'hansard' in topic_result:
            hansard_data = topic_result['hansard']
            csv_files = hansard_data.get('csv_files', [])
            
            for csv_file in csv_files:
                try:
                    if 'growth_magnitude' in csv_file and 'quarterly' not in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats.append(df)
                    
                    elif 'growth_magnitude_quarterly' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_stats_quarterly.append(df)
                    
                    elif 'speeches' in csv_file:
                        df = pd.read_csv(csv_file)
                        df['topic'] = topic_name
                        all_speeches.append(df)
                        
                except Exception as e:
                    logger.warning(f"Could not load Hansard file {csv_file} for {topic_name}: {e}")
    
    return (
        pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        pd.concat(all_stats_quarterly, ignore_index=True) if all_stats_quarterly else pd.DataFrame(),
        pd.concat(all_speeches, ignore_index=True) if all_speeches else pd.DataFrame()
    )

def _upload_to_google_sheets(
    cb_stats: pd.DataFrame, cb_stats_quarterly: pd.DataFrame, cb_companies: pd.DataFrame,
    cb_funding: pd.DataFrame, cb_ipos: pd.DataFrame, cb_acquisitions: pd.DataFrame,
    ukri_stats: pd.DataFrame, ukri_stats_quarterly: pd.DataFrame, ukri_projects: pd.DataFrame,
    hansard_stats: pd.DataFrame, hansard_stats_quarterly: pd.DataFrame, hansard_speeches: pd.DataFrame,
    sheet_id: str
) -> None:
    """Upload consolidated DataFrames to Google Sheets.
    
    Args:
        Various DataFrames containing consolidated data
        sheet_id: Google Sheets ID to upload to
    """
    logger.info(f"Uploading consolidated data to Google Sheets: {sheet_id}")
    
    # Prepare dataframes for upload (only non-empty ones)
    dataframes = {}
    
    # Define mapping of dataframe names to their variables
    dataframe_mapping = {
        'crunchbase_stats': cb_stats,
        'crunchbase_stats_quarterly': cb_stats_quarterly,
        #'crunchbase_companies': cb_companies, # TODO: bug - sheet upload stalls, upload manually
        'crunchbase_funding': cb_funding,
        'crunchbase_ipos': cb_ipos,
        'crunchbase_acquisitions': cb_acquisitions,
        'ukri_stats': ukri_stats,
        'ukri_stats_quarterly': ukri_stats_quarterly,
        'ukri_projects': ukri_projects,
        'hansard_stats': hansard_stats,
        'hansard_stats_quarterly': hansard_stats_quarterly,
        # TODO: bug - sheet upload error - input contains more than the maximum of 50000 characters in a single cell
        # upload manually
        #'hansard_speeches': hansard_speeches 
    }
    
    # Add non-empty dataframes to the dataframes dict
    dataframes.update({
        name: df for name, df in dataframe_mapping.items() 
        if not df.empty
    })
    if dataframes:
        # Upload to Google Sheets
        google.upload_data_to_gsheet(sheet_id, dataframes)
        logger.info(f"Successfully uploaded {len(dataframes)} sheets to Google Sheets")
    else:
        logger.warning("No data to upload to Google Sheets") 