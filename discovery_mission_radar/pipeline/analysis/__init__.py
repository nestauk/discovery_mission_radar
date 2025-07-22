"""
Analysis module for Mission Radar Pipeline.

This module provides class-based analysis modules following ABC pattern
using existing discovery_utils functionality.
"""

# Import base classes
from .base import BaseAnalysisModule

# Import ABC implementation classes  
from .crunchbase_analysis import CrunchbaseAnalysisModule
from .gtr_analysis import GtrAnalysisModule
from .hansard_analysis import HansardAnalysisModule

# Import aggregation and consolidation functions
from .aggregation import produce_radar_charts
from .consolidation import consolidate_all_topics

__all__ = [
    # Base classes
    'BaseAnalysisModule',
    
    # ABC implementation classes  
    'CrunchbaseAnalysisModule',
    'GtrAnalysisModule', 
    'HansardAnalysisModule',
    
    # Aggregation functions
    'produce_radar_charts',
    'consolidate_all_topics'
] 