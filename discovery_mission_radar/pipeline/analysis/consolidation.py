"""
Consolidation module for generating final consolidated CSV files
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def consolidate_all_topics(topic_results: List[Dict[str, Any]], output_dir: Path) -> Dict[str, str]:
    """
    Consolidate all topic results into final CSV files
    
    Args:
        topic_results: List of results from individual topic analysis
        output_dir: Where to save consolidated CSV files
        
    Returns:
        Dictionary mapping CSV file names to their paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Consolidating data from {len(topic_results)} topics")
    
    consolidated_files = {}
    
    cb_stats, cb_stats_quarterly, cb_companies, cb_funding, cb_ipos, cb_acquisitions = _consolidate_crunchbase_data(topic_results)
    
    ukri_stats, ukri_stats_quarterly, ukri_projects = _consolidate_gtr_data(topic_results)
    
    hansard_stats, hansard_stats_quarterly, hansard_speeches = _consolidate_hansard_data(topic_results)
    
    if not cb_stats.empty:
        cb_stats_path = output_dir / "crunchbase_stats.csv"
        cb_stats.to_csv(cb_stats_path, index=False)
        consolidated_files['crunchbase_stats'] = str(cb_stats_path)
        logger.info(f"Saved crunchbase_stats.csv with {len(cb_stats)} rows")
    
    if not cb_stats_quarterly.empty:
        cb_stats_quarterly_path = output_dir / "crunchbase_stats_quarterly.csv"
        cb_stats_quarterly.to_csv(cb_stats_quarterly_path, index=False)
        consolidated_files['crunchbase_stats_quarterly'] = str(cb_stats_quarterly_path)
        logger.info(f"Saved crunchbase_stats_quarterly.csv with {len(cb_stats_quarterly)} rows")
    
    if not cb_companies.empty:
        cb_companies_path = output_dir / "crunchbase_companies.csv"
        cb_companies.to_csv(cb_companies_path, index=False)
        consolidated_files['crunchbase_companies'] = str(cb_companies_path)
        logger.info(f"Saved crunchbase_companies.csv with {len(cb_companies)} rows")
    
    if not cb_funding.empty:
        cb_funding_path = output_dir / "crunchbase_funding.csv"
        cb_funding.to_csv(cb_funding_path, index=False)
        consolidated_files['crunchbase_funding'] = str(cb_funding_path)
        logger.info(f"Saved crunchbase_funding.csv with {len(cb_funding)} rows")
    
    if not cb_ipos.empty:
        cb_ipos_path = output_dir / "crunchbase_ipos.csv"
        cb_ipos.to_csv(cb_ipos_path, index=False)
        consolidated_files['crunchbase_ipos'] = str(cb_ipos_path)
        logger.info(f"Saved crunchbase_ipos.csv with {len(cb_ipos)} rows")
    
    if not cb_acquisitions.empty:
        cb_acquisitions_path = output_dir / "crunchbase_acquisitions.csv"
        cb_acquisitions.to_csv(cb_acquisitions_path, index=False)
        consolidated_files['crunchbase_acquisitions'] = str(cb_acquisitions_path)
        logger.info(f"Saved crunchbase_acquisitions.csv with {len(cb_acquisitions)} rows")
    
    if not ukri_stats.empty:
        ukri_stats_path = output_dir / "ukri_stats.csv"
        ukri_stats.to_csv(ukri_stats_path, index=False)
        consolidated_files['ukri_stats'] = str(ukri_stats_path)
        logger.info(f"Saved ukri_stats.csv with {len(ukri_stats)} rows")
    
    if not ukri_stats_quarterly.empty:
        ukri_stats_quarterly_path = output_dir / "ukri_stats_quarterly.csv"
        ukri_stats_quarterly.to_csv(ukri_stats_quarterly_path, index=False)
        consolidated_files['ukri_stats_quarterly'] = str(ukri_stats_quarterly_path)
        logger.info(f"Saved ukri_stats_quarterly.csv with {len(ukri_stats_quarterly)} rows")
    
    if not ukri_projects.empty:
        ukri_projects_path = output_dir / "ukri_projects.csv"
        ukri_projects.to_csv(ukri_projects_path, index=False)
        consolidated_files['ukri_projects'] = str(ukri_projects_path)
        logger.info(f"Saved ukri_projects.csv with {len(ukri_projects)} rows")
    
    if not hansard_stats.empty:
        hansard_stats_path = output_dir / "hansard_stats.csv"
        hansard_stats.to_csv(hansard_stats_path, index=False)
        consolidated_files['hansard_stats'] = str(hansard_stats_path)
        logger.info(f"Saved hansard_stats.csv with {len(hansard_stats)} rows")
    
    if not hansard_stats_quarterly.empty:
        hansard_stats_quarterly_path = output_dir / "hansard_stats_quarterly.csv"
        hansard_stats_quarterly.to_csv(hansard_stats_quarterly_path, index=False)
        consolidated_files['hansard_stats_quarterly'] = str(hansard_stats_quarterly_path)
        logger.info(f"Saved hansard_stats_quarterly.csv with {len(hansard_stats_quarterly)} rows")
    
    if not hansard_speeches.empty:
        hansard_speeches_path = output_dir / "hansard_speeches.csv"
        hansard_speeches.to_csv(hansard_speeches_path, index=False)
        consolidated_files['hansard_speeches'] = str(hansard_speeches_path)
        logger.info(f"Saved hansard_speeches.csv with {len(hansard_speeches)} rows")
    
    logger.info(f"Consolidation complete. Generated {len(consolidated_files)} files")
    return consolidated_files

def _consolidate_crunchbase_data(topic_results: List[Dict[str, Any]]) -> tuple:
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

def _consolidate_gtr_data(topic_results: List[Dict[str, Any]]) -> tuple:
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

def _consolidate_hansard_data(topic_results: List[Dict[str, Any]]) -> tuple:
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