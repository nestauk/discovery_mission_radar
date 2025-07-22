"""
Abstract base class for Mission Radar Pipeline analysis modules.

This module provides the common interface and shared functionality for all analysis modules.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import pandas as pd
import altair as alt

from .utils import AnalysisUtils
from ..config_manager import get_pipeline_config
from discovery_utils.utils import charts

# Type variable for the getter type (e.g., CrunchbaseGetter, GtrGetter, etc.)
GetterType = TypeVar('GetterType')


class BaseAnalysisModule(ABC, Generic[GetterType]):
    """Abstract base class for all analysis modules in the pipeline.
    
    This class provides a common structure for data analysis and visualisation whilst allowing each analysis module
    to implement its specific logic.
    """
    
    def __init__(self, source_name: str, mission: str):
        """Initialise the analysis module.
        
        Args:
            source_name: Name of the analysis source (e.g., 'crunchbase', 'gtr', 'hansard')
            mission: Mission identifier (AHL or ASF)
        """
        self.source_name = source_name
        self.mission = mission.upper()
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
        self.config = get_pipeline_config(self.mission)
    
    def analyse_topic(self, topic_data: Dict[str, Any], output_dir: Path, 
                     getter: Optional[GetterType] = None) -> Dict[str, Any]:
        """Analyse topic data and generate outputs (Template Method).
        
        This is the main public interface that follows the template method pattern:
        1. Setup output directories
        2. Process topic data using specific implementation
        3. Save results using shared utilities
        4. Generate charts using discovery_utils.charts
        5. Return standardised format
        
        Args:
            topic_data: Output from data source containing IDs and config
            output_dir: Directory to save charts and CSV files
            getter: Pre-initialised getter instance (optional)
            
        Returns:
            Dictionary with standardised format:
            {
                'csv_files': List[str],     # Paths to saved CSV files
                'chart_files': List[str],   # Paths to saved chart files
                'stats': Dict[str, Any]     # Analysis statistics
            }
        """
        self.logger.info(f"Analysing {self.source_name} data for {topic_data.get('topic_name', 'unknown topic')}")
        
        # Setup output directories using shared utilities
        directories = AnalysisUtils.setup_output_directories(output_dir)
        
        # Create getter if not provided
        if getter is None:
            getter = self._create_default_getter()
        
        # Process topic data using source-specific implementation
        analysis_results = self._process_topic_data(topic_data, getter)
        
        # Save CSV files using shared utilities
        csv_files = AnalysisUtils.save_dataframes_to_csv(
            analysis_results, 
            directories['csv_dir'], 
            topic_data['config']['search_recipe']['category_name']
        )
        
        # Generate charts using discovery_utils.charts
        chart_files = self._generate_charts(analysis_results, directories['charts_dir'], topic_data)
        
        # Generate standard statistics
        stats = AnalysisUtils.generate_standard_stats(analysis_results)
        
        # Add source-specific statistics
        stats.update(self._generate_custom_stats(analysis_results, topic_data))
        
        result = {
            'csv_files': csv_files,
            'chart_files': chart_files,
            'stats': stats
        }
        
        self.logger.info(f"Completed {self.source_name} analysis: {len(csv_files)} CSV files, {len(chart_files)} charts")
        return result
    
    # Abstract methods that must be implemented by each analysis module
    
    @abstractmethod
    def _create_default_getter(self) -> GetterType:
        """Create default getter instance for this analysis module.
        
        Returns:
            Initialised getter instance (e.g., CrunchbaseGetter())
        """
        pass
    
    @abstractmethod
    def _process_topic_data(self, topic_data: Dict[str, Any], getter: GetterType) -> Dict[str, pd.DataFrame]:
        """Process topic data and return analysis results as DataFrames.
        
        This method should:
        1. Extract relevant IDs from topic_data
        2. Use getter to retrieve additional data
        3. Perform analysis calculations using discovery_utils.utils.analysis* modules
        4. Return results as named DataFrames
        
        Args:
            topic_data: Output from data source containing IDs and config
            getter: Initialised getter instance
            
        Returns:
            Dictionary mapping result names to DataFrames:
            {
                'matching_items': pd.DataFrame,      # Items that matched the topic
                'ts_yearly': pd.DataFrame,           # Yearly timeseries data
                'ts_quarterly': pd.DataFrame,        # Quarterly timeseries data
                'growth_magnitude': pd.DataFrame,    # Growth calculations
                # ... other analysis-specific DataFrames
            }
        """
        pass
    
    @abstractmethod
    def _generate_custom_stats(self, analysis_results: Dict[str, pd.DataFrame], 
                              topic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate source-specific statistics from analysis results.
        
        Args:
            analysis_results: DataFrames returned by _process_topic_data
            topic_data: Original topic data
            
        Returns:
            Dictionary with source-specific statistics
        """
        pass
    
    def _generate_charts(self, analysis_results: Dict[str, pd.DataFrame], 
                        charts_dir: Path, topic_data: Dict[str, Any]) -> List[str]:
        """Generate charts using discovery_utils.charts.
        
        This method creates charts using the existing discovery_utils.charts module
        following the same pattern as current analysis modules.
        
        Args:
            analysis_results: DataFrames from analysis
            charts_dir: Directory to save charts
            topic_data: Original topic data
            
        Returns:
            List of chart file paths
        """
        chart_files = []
        category_name = topic_data['config']['search_recipe']['category_name']
        scale_factor = 2
        
        # Store current theme and enable default
        current_theme = alt.themes.active
        alt.themes.enable('default')
        
        try:
            # Generate source-specific charts using discovery_utils.charts
            chart_files = self._create_source_charts(analysis_results, charts_dir, category_name, scale_factor)
            
        except Exception as e:
            self.logger.error(f"Failed to generate charts: {e}")
            
        finally:
            # Restore original theme
            alt.themes.enable(current_theme)
        
        self.logger.info(f"Generated {len(chart_files)} charts for {category_name}")
        return chart_files
    
    @abstractmethod
    def _create_source_charts(self, analysis_results: Dict[str, pd.DataFrame], 
                             charts_dir: Path, category_name: str, scale_factor: int) -> List[str]:
        """Create source-specific charts using discovery_utils.charts.
        
        This method should use discovery_utils.charts functions like:
        - charts.ts_bar() for timeseries bar charts
        - charts.ts_smooth() for smooth timeseries charts
        - charts.configure_plots() for styling
        
        Args:
            analysis_results: DataFrames from analysis
            charts_dir: Directory to save charts
            category_name: Category name for chart titles and filenames
            scale_factor: Scale factor for chart saving
            
        Returns:
            List of saved chart file paths
        """
        pass
    
    def validate_topic_data(self, topic_data: Dict[str, Any]) -> None:
        """Validate topic data for this analysis module.
        
        This provides common validation for all analysis modules.
        Subclasses can override this method if they need additional validation.
        
        Args:
            topic_data: Topic data dictionary to validate
            
        Raises:
            ValueError: If topic data is invalid
            KeyError: If required keys are missing
        """
        if 'ids' not in topic_data:
            raise ValueError("Missing 'ids' in topic_data")
        
        if 'config' not in topic_data:
            raise ValueError("Missing 'config' in topic_data")
        
        if 'search_recipe' not in topic_data['config']:
            raise ValueError("Missing 'search_recipe' in topic_data config")
        
        if 'category_name' not in topic_data['config']['search_recipe']:
            raise ValueError("Missing 'category_name' in search_recipe")
        
        if not isinstance(topic_data['ids'], list):
            raise ValueError("'ids' must be a list")
        
        if not topic_data['ids']:
            self.logger.warning("No IDs found in topic_data - analysis may produce empty results") 