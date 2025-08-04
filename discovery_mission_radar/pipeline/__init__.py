"""Mission Radar Pipeline - Production-ready analysis pipeline for innovation ecosystems.

This package provides a modular, configuration-driven pipeline for analyzing 
innovation ecosystems across government mission areas using data from 
Crunchbase, Gateway to Research (GtR), and Hansard.
"""

__version__ = "1.0.0"
__author__ = "Methods Discovery"

from .config_manager import PipelineConfig, get_pipeline_config
from .runner import MissionRadarRunner