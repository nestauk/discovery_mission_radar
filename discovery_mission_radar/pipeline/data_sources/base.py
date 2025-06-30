"""
Abstract base class for Mission Radar Pipeline data sources.

This module provides the common interface and shared functionality for all data sources
including caching, validation, and standardised return formats.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import pandas as pd

# Type variable for the getter type (e.g., CrunchbaseGetter, GtrGetter, etc.)
GetterType = TypeVar('GetterType')


class BaseDataSource(ABC, Generic[GetterType]):
    """Abstract base class for all data sources in the Mission Radar Pipeline.
    
    This class implements the Template Method pattern, providing a common structure
    for data fetching, caching, and validation whilst allowing each data source
    to implement its specific logic.
    """
    
    def __init__(self, source_name: str):
        """Initialise the data source.
        
        Args:
            source_name: Name of the data source (e.g., 'crunchbase', 'gtr', 'hansard')
        """
        self.source_name = source_name
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
    
    def get_data(self, topic_name: str, cache_dir: Path, config: Dict[str, Any], 
                 getter: Optional[GetterType] = None, **kwargs) -> Dict[str, Any]:
        """Get data for a topic with caching (Template Method).
        
        This is the main public interface that follows the template method pattern:
        1. Check cache validity
        2. Load from cache if valid, otherwise fetch fresh data
        3. Cache and return results
        
        Args:
            topic_name: Name of the topic (e.g., "hydrogen_energy")
            cache_dir: Directory for caching results
            config: Topic configuration dictionary
            getter: Pre-initialised getter instance (optional)
            **kwargs: Additional source-specific arguments
            
        Returns:
            Dictionary with standardised format:
            {
                'ids': List[str],           # Relevant IDs
                'topic_name': str,          # Topic name
                'config': Dict,             # Original config
                '{items}_count': int,       # Total items found
                'relevant_count': int       # Items after relevance check
            }
        """
        cache_file = cache_dir / f"{topic_name}_{self.source_name}_data.json"
        
        # Check cache first
        if self._is_cache_valid(cache_file, max_age_hours=24):
            self.logger.info(f"Loading cached {self.source_name} data for {topic_name}")
            return self._load_cached_result(cache_file)
        
        self.logger.info(f"Processing fresh {self.source_name} data for {topic_name}")
        
        # Create getter if not provided
        if getter is None:
            getter = self._create_default_getter()
        
        # Fetch fresh data using source-specific implementation
        relevant_ids, total_count = self._fetch_fresh_data(topic_name, config, getter, cache_dir, **kwargs)
        
        # Prepare standardised result
        result = {
            'ids': relevant_ids,
            'topic_name': topic_name,
            'config': config,
            f'{self._get_item_type()}_count': total_count,
            'relevant_count': len(relevant_ids)
        }
        
        # Cache result
        self._cache_result(result, cache_file, cache_dir)
        
        self.logger.info(f"Cached {len(relevant_ids)} relevant {self._get_item_type()} for {topic_name}")
        return result
    
    # Abstract methods that must be implemented by each data source
    
    @abstractmethod
    def _create_default_getter(self) -> GetterType:
        """Create default getter instance for this data source.
        
        Returns:
            Initialised getter instance (e.g., CrunchbaseGetter())
        """
        pass
    
    @abstractmethod
    def _fetch_fresh_data(self, topic_name: str, config: Dict[str, Any], 
                         getter: GetterType, cache_dir: Path, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from the source and return relevant IDs.
        
        This method should:
        1. Get initial data using the getter
        2. Apply any pre-filtering
        3. Run relevance checks if needed
        4. Return (relevant_ids, total_count)
        
        Args:
            topic_name: Name of the topic
            config: Topic configuration dictionary
            getter: Initialised getter instance
            cache_dir: Cache directory for intermediate files
            **kwargs: Additional source-specific arguments
            
        Returns:
            Tuple of (relevant_ids, total_count_before_relevance_check)
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration for this data source.
        
        Should raise appropriate exceptions for invalid configurations.
        
        Args:
            config: Topic configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
            KeyError: If required keys are missing
        """
        pass
    
    @abstractmethod
    def _get_item_type(self) -> str:
        """Get the name of items this source returns (e.g., 'companies', 'projects', 'speeches').
        
        Used for generating consistent field names in the return dictionary.
        
        Returns:
            String name of the item type
        """
        pass
    
    # Concrete methods - shared implementation
    
    def _is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file exists and is recent enough.
        
        Args:
            cache_file: Path to the cache file
            max_age_hours: Maximum age in hours before cache is considered stale
            
        Returns:
            True if cache exists and is fresh, False otherwise
        """
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            return age_hours < max_age_hours
        return False
    
    def _cache_result(self, result: Dict[str, Any], cache_file: Path, cache_dir: Path) -> None:
        """Cache result to file.
        
        Args:
            result: Result dictionary to cache
            cache_file: Path to the cache file
            cache_dir: Cache directory (will be created if needed)
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _load_cached_result(self, cache_file: Path) -> Dict[str, Any]:
        """Load cached result from file.
        
        Args:
            cache_file: Path to the cache file
            
        Returns:
            Loaded result dictionary
        """
        with open(cache_file, 'r') as f:
            return json.load(f) 