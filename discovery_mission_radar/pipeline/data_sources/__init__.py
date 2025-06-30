"""
Simple data sources module for Mission Radar Pipeline.

This module provides simple data fetching functions that replace complex
data preparation classes, mirroring notebook patterns exactly.
"""

from .crunchbase import get_cb_data, validate_cb_config
from .gtr import get_gtr_data, validate_gtr_config  
from .hansard import get_hansard_data, validate_hansard_config

__all__ = [
    'get_cb_data', 'validate_cb_config',
    'get_gtr_data', 'validate_gtr_config', 
    'get_hansard_data', 'validate_hansard_config'
] 