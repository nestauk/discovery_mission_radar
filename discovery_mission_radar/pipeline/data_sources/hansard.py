"""
Hansard data fetching
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils.llm import batch_check
import asyncio
import re
import logging

from .base import BaseDataSource
from .utils import run_llm_relevance_check

logger = logging.getLogger(__name__)


class HansardDataSource(BaseDataSource[policy_update.HansardData]):
    """Hansard data source implementation."""
    
    def __init__(self):
        super().__init__("hansard")
    
    def _create_default_getter(self) -> policy_update.HansardData:
        """Create default HansardData instance."""
        return policy_update.HansardData()
    
    def _get_item_type(self) -> str:
        """Return the item type for Hansard (speeches)."""
        return "speeches"
    
    def _fetch_fresh_data(self, topic_name: str, config: Dict[str, Any], 
                         getter: policy_update.HansardData, cache_dir: Path, use_llm_check: bool = False, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from Hansard and return relevant speech IDs."""
        
        # Get speeches by category (this uses existing classification)
        category_name = config['search_recipe']['category_name']
        speeches_df = self._get_speeches_data(getter, category_name)
        
        self.logger.info(f"Found {len(speeches_df)} speeches for category '{category_name}'")
        
        if use_llm_check:
            # Run LLM relevance check if requested
            relevant_ids = self._run_relevance_check(speeches_df, config, cache_dir, topic_name)
        else:
            # Just return all IDs (mirroring original notebook behaviour)
            relevant_ids = speeches_df['speech_id'].unique().tolist()
            
        return relevant_ids, len(speeches_df)
    
    def _get_speeches_data(self, HansardData: policy_update.HansardData, category_name: str) -> pd.DataFrame:
        """Get speeches data for a specific category"""
        
        speeches_df = (
            HansardData.debates_df
            .merge(
                HansardData.labelstore_df[['id', 'topic_labels']],
                left_on='speech_id',
                right_on='id',
                how='left'
            )
            .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
            .explode("topic_labels")
            .query("topic_labels == @category_name")
            .assign(quarter=lambda df: df.date.apply(_get_quarter_from_date))
            .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
        )
        
        return speeches_df
    
    def _run_relevance_check(self, speeches_df: pd.DataFrame, config: Dict, 
                           cache_dir: Path, topic_name: str) -> List[str]:
        """Run LLM relevance check for Hansard speeches."""
        
        # Define custom instructions for Hansard
        custom_instructions = """
        Mark the text as 'yes' if the parliamentary speech mentions or discusses the technology/topic in a meaningful way, including:
        - Policy discussions about the technology
        - Questions about government support or regulation
        - Debates about implementation or challenges
        - References to the technology in broader energy or climate discussions
        
        Mark as 'no' if:
        - The technology is only mentioned in passing
        - The speech is primarily about unrelated topics
        - The mention is purely incidental or part of a list
        """
        
        # Prepare speeches with text for LLM check
        speeches_with_text = speeches_df.copy()
        speeches_with_text['text'] = speeches_with_text['speech'].apply(lambda x: re.sub(r"\s+", " ", str(x)))
        
        # Use shared LLM relevance check function
        return run_llm_relevance_check(
            speeches_with_text,
            config,
            cache_dir,
            topic_name,
            "hansard",
            id_column='speech_id',
            text_column='text',
            custom_instructions=custom_instructions
        )
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate Hansard configuration."""
        if 'search_recipe' not in config:
            raise ValueError("Missing 'search_recipe' in config")
        
        search_recipe = config['search_recipe']
        if 'category_name' not in search_recipe:
            raise ValueError("Missing 'category_name' in search_recipe")
        
        category_name = search_recipe['category_name']
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("'category_name' must be a non-empty string")


def _get_quarter_from_date(date: str) -> int:
    """Get quarter from date string"""
    month = pd.to_datetime(date).month
    return (month - 1) // 3 + 1 