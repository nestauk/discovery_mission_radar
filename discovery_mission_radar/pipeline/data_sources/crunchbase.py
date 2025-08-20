"""
Crunchbase data fetching
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.getters import crunchbase
from discovery_utils.utils.llm import batch_check
import asyncio
import time
import json
import logging

from .base import BaseDataSource
from .utils import run_llm_relevance_check, load_llm_prompts

logger = logging.getLogger(__name__)


class CrunchbaseDataSource(BaseDataSource[crunchbase.CrunchbaseGetter]):
    """Crunchbase data source implementation."""
    
    def __init__(self):
        super().__init__("crunchbase")
    
    def _create_default_getter(self) -> crunchbase.CrunchbaseGetter:
        """Create default CrunchbaseGetter instance."""
        return crunchbase.CrunchbaseGetter()
    
    def _get_item_type(self) -> str:
        """Return the item type for Crunchbase (companies)."""
        return "companies"
    
    def _fetch_fresh_data(self, topic_name: str, config: Dict[str, Any], 
                         getter: crunchbase.CrunchbaseGetter, cache_dir: Path, 
                         mission: str = None, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from Crunchbase and return relevant company IDs."""
        
        # Get companies by category
        category_name = config['search_recipe']['category_name']
        companies_df = getter.get_companies_in_nesta_categories("topic_labels", [category_name])
        
        self.logger.info(f"Found {len(companies_df)} companies for category '{category_name}'")
        
        # Run LLM relevance check with mission context
        relevant_ids = self._run_relevance_check(companies_df, config, cache_dir, topic_name, getter, mission, **kwargs)
        
        return relevant_ids, len(companies_df)
    
    def _run_relevance_check(self, companies_df: pd.DataFrame, config: Dict, 
                            cache_dir: Path, topic_name: str, cb_getter: crunchbase.CrunchbaseGetter,
                            mission: str = None, **kwargs) -> List[str]:
        """Run LLM relevance check for Crunchbase companies."""
        
        # Get organisation text for relevance checking
        selected_texts_df = cb_getter.get_organisation_text(companies_df)
        
        # Merge companies with their text data
        companies_with_text = (
            companies_df
            .merge(selected_texts_df[['id', 'text']], left_on='id', right_on='id', how='left')
        )
        
        custom_instructions = self._get_mission_specific_instructions(mission or "Unknown")
        
        return run_llm_relevance_check(
            companies_with_text,
            config,
            cache_dir,
            topic_name,
            "crunchbase",
            mission or "Unknown",
            pipeline_config=kwargs.get('pipeline_config'),
            custom_instructions=custom_instructions
        )
    
    def _get_mission_specific_instructions(self, mission: str) -> str:
        """Get mission/source instructions from central YAML; no in-code prompts."""
        try:
            base_config_dir = Path(__file__).parent.parent / "config"
            prompts = load_llm_prompts(base_config_dir)
            key = mission if mission in prompts else "default"
            mission_prompts = prompts.get(key, {})
            source_key = getattr(self, 'source_name', '').lower() or 'crunchbase'
            text = (
                mission_prompts.get(source_key)
                or mission_prompts.get("default")
                or (prompts.get("default", {}) or {}).get("default")
            )
            if text:
                return text
            self.logger.warning(f"No LLM prompt found in YAML for mission={mission}, source={source_key}; proceeding without custom instructions")
            return ""
        except Exception as e:
            self.logger.warning(f"Failed loading LLM prompts: {e}; proceeding without custom instructions")
            return ""