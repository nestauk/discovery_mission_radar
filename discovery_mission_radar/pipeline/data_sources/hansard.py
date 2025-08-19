"""
Hansard data fetching
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.getters import hansard
from discovery_utils.utils.llm import batch_check
import asyncio
import re
import logging

from .base import BaseDataSource
from .utils import run_llm_relevance_check

logger = logging.getLogger(__name__)


class HansardDataSource(BaseDataSource[hansard.HansardGetter]):
    """Hansard data source implementation."""
    
    def __init__(self):
        super().__init__("hansard")
    
    def _create_default_getter(self) -> hansard.HansardGetter:
        """Create default HansardGetter instance."""
        return hansard.HansardGetter()
    
    def _get_item_type(self) -> str:
        """Return the item type for Hansard (speeches)."""
        return "speeches"
    
    def _fetch_fresh_data(self, topic_name: str, config: Dict[str, Any], 
                         getter: hansard.HansardGetter, cache_dir: Path, 
                         use_llm_check: bool = False, mission: str = None, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from Hansard and return relevant speech IDs."""
        
        # Get speeches by category (this uses existing classification)
        category_name = config['search_recipe']['category_name']
        speeches_df = self._get_speeches_data(getter, category_name)
        
        self.logger.info(f"Found {len(speeches_df)} speeches for category '{category_name}'")
        
        if use_llm_check:
            # Run LLM relevance check if requested with mission context
            relevant_ids = self._run_relevance_check(speeches_df, config, cache_dir, topic_name, mission, **kwargs)
        else:
            # Just return all IDs (mirroring original notebook behaviour)
            relevant_ids = speeches_df['speech_id'].unique().tolist()
            
        return relevant_ids, len(speeches_df)
    
    def _get_speeches_data(self, getter: hansard.HansardGetter, category_name: str) -> pd.DataFrame:
        """Get speeches data for the given category."""
        
        speeches_df = (
            getter.get_debates_parquet()
            .merge(
                getter.get_labelstore(keywords=True)[['id', 'topic_labels']],
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
                           cache_dir: Path, topic_name: str, mission: str = None, **kwargs) -> List[str]:
        """Run LLM relevance check for Hansard speeches."""
        
        # Get mission-aware custom instructions
        custom_instructions = self._get_mission_specific_instructions(mission or "Unknown")
        
        # Prepare speeches with text for LLM check
        speeches_with_text = speeches_df.copy()
        speeches_with_text['text'] = speeches_with_text['speech'].apply(lambda x: re.sub(r"\s+", " ", str(x)))
        
        # Use shared LLM relevance check function with pipeline config for Argilla
        return run_llm_relevance_check(
            speeches_with_text,
            config,
            cache_dir,
            topic_name,
            "hansard",
            mission or "Unknown",
            pipeline_config=kwargs.get('pipeline_config'),
            id_column='speech_id',
            text_column='text',
            custom_instructions=custom_instructions
        )
    
    def _get_mission_specific_instructions(self, mission: str) -> str:
        """Get custom instructions tailored to the specific mission."""
        
        if mission == "ASF":
            return """
            Mark the text as 'yes' if the parliamentary speech mentions or discusses the technology/topic in a meaningful way, including:
            - Policy discussions about the technology or energy solutions
            - Questions about government support, regulation, or funding for the technology
            - Debates about implementation, deployment, or challenges related to the technology
            - References to the technology in broader energy, climate, or sustainability discussions
            - Discussions about targets, strategies, or plans involving the technology
            
            Mark as 'no' if:
            - The technology is only mentioned in passing or as part of a long list
            - The speech is primarily about unrelated topics
            - The mention is purely incidental or tangential
            - The discussion is about general energy policy without specific reference to the technology
            - The text would be better captured by one of the other categories (comma separated) mentioned in this list:  
            Bioenergy (biofuels), Biomass heating, Carbon capture and storage, District heating and heat networks, Energy grid, Geothermal energy, 
            Heat pumps, Hydrogen energy, Hydrogen heating, Micro CHP, Solar thermal heating, Energy storage (batteries), Solar power, Wind power
            """
            
        elif mission == "AHL":
            return """
            Mark the text as 'yes' if the parliamentary speech mentions or discusses the topic in a meaningful way, including:
            - Policy discussions about food, nutrition, health, or obesity-related matters
            - Questions about government support, regulation, or funding for health/food interventions
            - Debates about implementation or challenges related to food systems, dietary health, or obesity prevention
            - References to the topic in broader health, food policy, or public health discussions
            - Discussions about health targets, strategies, or plans involving the topic
            - Food environment, food labelling, food advertising, or food access policy discussions
            
            Mark as 'no' if:
            - The topic is only mentioned in passing or as part of a long list
            - The speech is primarily about unrelated topics (e.g., unrelated health issues, non-food policies)
            - The mention is purely incidental or tangential
            - The discussion is about general health policy without specific reference to the topic
            - The text would be better captured by one of the other categories (comma separated) mentioned in this list:
            Alternative proteins (general), Plant-based foods, Cloud kitchens, Food delivery apps, Fermentation, Food advertisement, 
            Food tech (general), Health (general), Insects as food, Kitchen technology, Lab-grown meat, Meal kits, 
            Personalised nutrition, Restaurants, Food retail, Supply chain, Weight management, Weight-loss drugs, 
            Food reformulation (general), Food reformulation (sugar), Food reformulation (salt), Food reformulation (fat), Food reformulation (fiber)
            """
            
        else:
            # Generic instructions for unknown missions
            return """
            Mark the text as 'yes' if the parliamentary speech mentions or discusses the topic in a meaningful way, including:
            - Policy discussions about the topic
            - Questions about government support or regulation related to the topic
            - Debates about implementation or challenges
            - References to the topic in broader relevant policy discussions
            
            Mark as 'no' if:
            - The topic is only mentioned in passing
            - The speech is primarily about unrelated topics
            - The mention is purely incidental or part of a list
            """


def _get_quarter_from_date(date: str) -> int:
    """Get quarter from date string"""
    month = pd.to_datetime(date).month
    return (month - 1) // 3 + 1 