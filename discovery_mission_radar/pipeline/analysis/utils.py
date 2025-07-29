"""
Shared utilities for analysis modules in the Mission Radar Pipeline.

This module contains common functionality used across multiple analysis modules,
including directory setup, CSV file operations, and statistics generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisUtils:
    """Shared utilities for analysis module operations."""
    
    @staticmethod
    def setup_output_directories(output_dir: Path) -> Dict[str, Path]:
        """Setup standard output directory structure for analysis modules.
        
        Creates directory structure:
        - output_dir/csv/     (for CSV files)
        - output_dir/charts/  (for chart files)
        
        Args:
            output_dir: Base output directory path
            
        Returns:
            Dictionary with directory paths:
            {
                'output_dir': Path,    # Base output directory
                'csv_dir': Path,       # CSV files directory
                'charts_dir': Path     # Charts directory
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_dir = output_dir / "csv"
        charts_dir = output_dir / "charts"
        
        csv_dir.mkdir(exist_ok=True)
        charts_dir.mkdir(exist_ok=True)
        
        directories = {
            'output_dir': output_dir,
            'csv_dir': csv_dir,
            'charts_dir': charts_dir
        }
        
        logger.debug(f"Created output directories: {list(directories.keys())}")
        return directories
    
    @staticmethod
    def save_dataframes_to_csv(dataframes: Dict[str, pd.DataFrame], 
                              output_dir: Path, category_name: str) -> List[str]:
        """Save analysis DataFrames to CSV files with standardised naming.
        
        Generates consistent file names based on category and DataFrame keys:
        - {category_name}_{dataframe_key}.csv
        
        Args:
            dataframes: Dictionary mapping names to DataFrames
            output_dir: Directory to save CSV files
            category_name: Category name for file naming
            
        Returns:
            List of saved CSV file paths
        """
        csv_files = []
        output_dir = Path(output_dir)
        
        for key, df in dataframes.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{category_name}_{key}.csv"
                file_path = output_dir / filename
                
                try:
                    df.to_csv(file_path, index=False)
                    csv_files.append(str(file_path))
                    logger.debug(f"Saved {key} data to {file_path} ({len(df)} rows)")
                    
                except Exception as e:
                    logger.warning(f"Failed to save {key} data to CSV: {e}")
            else:
                logger.debug(f"Skipping empty or invalid DataFrame: {key}")
        
        logger.info(f"Saved {len(csv_files)} CSV files for category '{category_name}'")
        return csv_files
    
    @staticmethod
    def generate_standard_stats(analysis_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate standard statistics from analysis DataFrames.
        
        Calculates common metrics that apply across all analysis modules:
        - Row counts for each DataFrame
        - Data completeness metrics
        - Basic summary statistics
        
        Args:
            analysis_results: Dictionary mapping names to DataFrames
            
        Returns:
            Dictionary with standard statistics
        """
        stats = {
            'dataframes_generated': len(analysis_results),
            'total_rows': 0,
            'dataframe_counts': {},
            'empty_dataframes': []
        }
        
        for key, df in analysis_results.items():
            if isinstance(df, pd.DataFrame):
                row_count = len(df)
                stats['dataframe_counts'][key] = row_count
                stats['total_rows'] += row_count
                
                if df.empty:
                    stats['empty_dataframes'].append(key)
            else:
                logger.warning(f"Non-DataFrame found in analysis_results: {key}")
                stats['empty_dataframes'].append(key)
        
        if stats['dataframes_generated'] > 0:
            stats['data_completeness'] = (
                (stats['dataframes_generated'] - len(stats['empty_dataframes'])) / 
                stats['dataframes_generated'] * 100
            )
        else:
            stats['data_completeness'] = 0.0
        
        logger.debug(f"Generated standard stats: {stats['dataframes_generated']} DataFrames, {stats['total_rows']} total rows")
        return stats
    
    @staticmethod
    def validate_analysis_results(analysis_results: Dict[str, pd.DataFrame], 
                                 required_keys: List[str] = None) -> bool:
        """Validate analysis results structure and content.
        
        Checks that analysis results contain expected DataFrames and data.
        
        Args:
            analysis_results: Dictionary mapping names to DataFrames
            required_keys: Optional list of required DataFrame keys
            
        Returns:
            True if validation passes, False otherwise
        """
        if not isinstance(analysis_results, dict):
            logger.error("analysis_results must be a dictionary")
            return False
        
        if not analysis_results:
            logger.warning("analysis_results is empty")
            return False
        
        if required_keys:
            missing_keys = set(required_keys) - set(analysis_results.keys())
            if missing_keys:
                logger.error(f"Missing required keys in analysis_results: {missing_keys}")
                return False
        
        # Validate DataFrame types
        invalid_types = []
        for key, value in analysis_results.items():
            if not isinstance(value, pd.DataFrame):
                invalid_types.append(key)
        
        if invalid_types:
            logger.error(f"Non-DataFrame values found: {invalid_types}")
            return False
        
        logger.debug("Analysis results validation passed")
        return True
    
    @staticmethod
    def merge_category_stats(category_stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge statistics from multiple categories for aggregated reporting.
        
        Combines statistics from multiple analysis runs for cross-topic analysis.
        
        Args:
            category_stats_list: List of statistics dictionaries from different categories
            
        Returns:
            Merged statistics dictionary
        """
        if not category_stats_list:
            return {}
        
        merged_stats = {
            'total_categories': len(category_stats_list),
            'total_dataframes': 0,
            'total_rows_across_all': 0,
            'categories_with_data': 0,
            'categories_without_data': 0,
            'average_completeness': 0.0
        }
        
        completeness_values = []
        
        for stats in category_stats_list:
            if isinstance(stats, dict):
                merged_stats['total_dataframes'] += stats.get('dataframes_generated', 0)
                merged_stats['total_rows_across_all'] += stats.get('total_rows', 0)
                
                completeness = stats.get('data_completeness', 0.0)
                completeness_values.append(completeness)
                
                if completeness > 0:
                    merged_stats['categories_with_data'] += 1
                else:
                    merged_stats['categories_without_data'] += 1
        
        if completeness_values:
            merged_stats['average_completeness'] = sum(completeness_values) / len(completeness_values)
        
        logger.info(f"Merged stats for {merged_stats['total_categories']} categories")
        return merged_stats
    
    @staticmethod
    def create_summary_dataframe(analysis_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create summary DataFrame with metadata about analysis results.
        
        Generates a summary table with information about each DataFrame in the results.
        
        Args:
            analysis_results: Dictionary mapping names to DataFrames
            
        Returns:
            Summary DataFrame with columns: name, rows, columns, memory_usage, is_empty
        """
        summary_data = []
        
        for name, df in analysis_results.items():
            if isinstance(df, pd.DataFrame):
                summary_data.append({
                    'dataframe_name': name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'is_empty': df.empty,
                    'column_names': ', '.join(df.columns.tolist()[:5])  # First 5 columns
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            totals = {
                'dataframe_name': 'TOTAL',
                'rows': summary_df['rows'].sum(),
                'columns': summary_df['columns'].sum(),
                'memory_usage_mb': summary_df['memory_usage_mb'].sum(),
                'is_empty': False,
                'column_names': f"{len(summary_df)} DataFrames"
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
        
        return summary_df 

    @staticmethod
    def country_to_region(country_code: str) -> str:
        """Map a country code to its region using a predefined mapping."""
        REGION_TO_COUNTRIES = {
            "North America + Australia": ["USA", "CAN", "AUS", "NZL"],
            "South + Central America": [
                "VEN", "ARG", "BRA", "CHL", "COL", "PER", "URY", "PRY", "ECU", "BOL", "GUY", "SUR", "MEX", "CRI", "SLV", "GTM", "HND", "PAN", "NIC",
            ],
            "Europe": [
                "IRL", "LUX", "CHE", "ESP", "DEU", "FRA", "FIN", "SWE", "NLD", "BEL", "DNK", "CZE", "POL", "EST", "AUT", "ITA", "ROU", "CYP", "NOR", "PRT", "BGR", "BLR", "SVN", "ARM", "HUN", "ISL", "LVA", "LTU", "HRV", "MKD", "BIH", "SRB", "SVK", "GEO", "MDA", "ALB", "SMR", "AND", "GIB", "FRO", "LIE", "IMN", "GGY", "JEY", "ALA",
            ],
            "UK": ["GBR"],
            "Asia": [
                "IND", "HKG", "ISR", "RUS", "KOR", "SGP", "JPN", "ARE", "CHN", "PHL", "IDN", "THA", "TUR", "MYS", "TWN", "PAK", "LBN", "ARM", "BGD", "KWT", "VNM", "MDV", "JOR", "LKA", "IRN", "SYR", "KAZ", "UZB", "IRQ", "OMN", "PSE", "TJK", "BTN", "TLS", "MAC", "MMR", "MNG", "KHM", "LAO", "BRN",
            ],
            "Africa": [
                "ZAF", "MUS", "EGY", "GHA", "KEN", "NGA", "MAR", "CIV", "ETH", "TUN", "MOZ", "UGA", "SEN", "ZWE", "RWA", "SDN",
            ],
            "Middle East": ["SAU", "ARE", "KWT", "QAT", "OMN", "IRQ", "IRN", "SYR", "JOR", "LBN", "ISR", "YEM"],
            "Rest of the World": [None, "BMU", "TTO", "GLP", "CYM", "IMN"],
        }
        for region, countries in REGION_TO_COUNTRIES.items():
            if country_code in countries:
                return region
        return "Rest of the World" 