"""
Abstract base class for Mission Radar Pipeline data sources.

This module provides the common interface and shared functionality for all data sources
including validation and standardised return formats. LLM relevance checks
use persistent S3 caching without expiry. Expensive operations like GTR enrichment
maintain local caching for efficiency.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

import pandas as pd

# Type variable for the getter type (e.g., CrunchbaseGetter, GtrGetter, etc.)
GetterType = TypeVar('GetterType')


class BaseDataSource(ABC, Generic[GetterType]):
    """Abstract base class for all data sources in the Mission Radar Pipeline.
    
    This class implements the Template Method pattern, providing a common structure
    for data fetching and validation whilst allowing each data source
    to implement its specific logic.
    
    LLM relevance checks use persistent S3 caching without time-based expiry,
    ensuring that once an entity is labelled, it doesn't need to be rechecked.
    Expensive operations like GTR enrichment maintain local caching for efficiency.
    Main data results are always fetched fresh without caching.
    """
    
    def __init__(self, source_name: str):
        """Initialise the data source.
        
        Args:
            source_name: Name of the data source (e.g., 'crunchbase', 'gtr', 'hansard')
        """
        self.source_name = source_name
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
    
    def get_data(self, topic_name: str, cache_dir: Path, config: Dict[str, Any], 
                 getter: Optional[GetterType] = None, mission: Optional[str] = None, 
                 pipeline_config: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Get data for a topic without local data caching (Template Method).
        
        This is the main public interface that follows the template method pattern:
        1. Always fetch fresh data (no local data caching)
        2. Return results in standardised format
        
        LLM relevance checks within _fetch_fresh_data use persistent S3 caching
        without time-based expiry. Only expensive operations like GTR enrichment
        maintain local caching.
        
        Args:
            topic_name: Name of the topic (e.g., "hydrogen_energy")
            cache_dir: Directory for caching intermediate files (LLM results, GTR enrichment)
            config: Topic configuration dictionary
            getter: Pre-initialised getter instance (optional)
            mission: Current mission (AHL/ASF) for S3 cache paths (optional)
            pipeline_config: Pipeline configuration for Argilla integration (optional)
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
        self.logger.info(f"Processing fresh {self.source_name} data for {topic_name}")
        
        # Create getter if not provided
        if getter is None:
            getter = self._create_default_getter()
        
        relevant_ids, total_count = self._fetch_fresh_data(
            topic_name, config, getter, cache_dir, mission=mission, 
            pipeline_config=pipeline_config, **kwargs
        )
        
        # Prepare standardised result
        result = {
            'ids': relevant_ids,
            'topic_name': topic_name,
            'config': config,
            f'{self._get_item_type()}_count': total_count,
            'relevant_count': len(relevant_ids)
        }
        
        self.logger.info(f"Found {len(relevant_ids)} relevant {self._get_item_type()} for {topic_name}")
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
        3. Run relevance checks if needed (using persistent S3 caching)
        4. Return (relevant_ids, total_count)
        
        Args:
            topic_name: Name of the topic
            config: Topic configuration dictionary
            getter: Initialised getter instance
            cache_dir: Cache directory for intermediate files
            **kwargs: Additional source-specific arguments (including mission and pipeline_config)
            
        Returns:
            Tuple of (relevant_ids, total_count_before_relevance_check)
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration for this data source.
        
        This provides common validation for all data sources that use search_recipe
        with category_name. Subclasses can override this method if they need
        additional source-specific validation.
        
        Args:
            config: Topic configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
            KeyError: If required keys are missing
        """
        if 'search_recipe' not in config:
            raise ValueError("Missing 'search_recipe' in config")
        
        search_recipe = config['search_recipe']
        if 'category_name' not in search_recipe:
            raise ValueError("Missing 'category_name' in search_recipe")
        
        category_name = search_recipe['category_name']
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("'category_name' must be a non-empty string")
    
    @abstractmethod
    def _get_item_type(self) -> str:
        """Get the name of items this source returns (e.g., 'companies', 'projects', 'speeches').
        
        Used for generating consistent field names in the return dictionary.
        
        Returns:
            String name of the item type
        """
        pass 