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
from .utils import run_llm_relevance_check

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
        
        # Get mission-aware custom instructions
        custom_instructions = self._get_mission_specific_instructions(mission or "Unknown")
        
        # Use shared LLM relevance check function with mission context and pipeline config for Argilla
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
        """Get custom instructions tailored to the specific mission."""
        
        if mission == "ASF":
            return """
            Mark the text as 'yes' if (one or more of the following):
            - If the technology defined by the scope above, is the main focus
            - If the technology is one of the components or activities described by the text. For example the technology could be mentioned
                as part of a larger project or business including other technologies, or be mentioned as one of the use cases or case studies.
            - If the text describes a company and the technology is the main focus, or a part of a broader range of the company's activities and offerings.
            - If the text is about a component or critical element of the technology defined above
            - If the text describes a technology or process and explicitly mentions that it can be applied on the technology defined above to improve its performance or efficiency.

            If the text is about heating technology but application target is not mentioned then assume it could be relevant for households or buildings (as opposed to industrial applications).

            However, mark it as 'no' if (one or more of the following):
            - The activities, or business described in the text does not have a discernible impact on or connection with the technology.
            - If the technology is mentioned only in passing or as a minor example in a broader discussion, for example, 
                in only one sentence within a long text with many sentences, or at the very end of a long description.
            - The technology is mentioned only as a negative example (eg "unlike [technology]...")
            - The text mentions heat pumps for heating swimming pools  
            - The text would be better captured by one of the other categories (comma separated) mentioned in this list:  
            Bioenergy (biofuels), Biomass heating, Carbon capture and storage, District heating and heat networks, Energy grid, Geothermal energy, 
            Heat pumps, Hydrogen energy, Hydrogen heating, Micro CHP, Solar thermal heating, Energy storage (batteries), Solar power, Wind power
            """
            
        elif mission == "AHL":
            return """
            Mark the text as 'yes' if (one or more of the following):
            - If the topic/area defined by the scope above is the main focus of the company or business
            - If the topic is one of the components or business areas described by the text. For example the topic could be mentioned
                as part of a larger food/health company including other business areas, or be mentioned as one of the use cases or products.
            - If the text describes a company and the topic is the main focus, or a part of a broader range of the company's activities and offerings.
            - If the text is about a component or critical element of the topic/area defined above
            - If the text describes products, services, or approaches that explicitly mention they can be applied to the topic defined above to improve health outcomes or address food-related challenges.

            Consider food technology, nutrition, obesity prevention, food systems, dietary solutions, food delivery, food production, and related health business areas as relevant contexts.

            However, mark it as 'no' if (one or more of the following):
            - The activities, or business described in the text does not have a discernible impact on or connection with the topic.
            - If the topic is mentioned only in passing or as a minor example in a broader discussion, for example, 
                in only one sentence within a long text with many sentences, or at the very end of a long description.
            - The topic is mentioned only as a negative example (eg "unlike [topic]...")
            - The text would be better captured by one of the other categories (comma separated) mentioned in this list:
            Alternative proteins (general), Plant-based foods, Cloud kitchens, Food delivery apps, Fermentation, Food advertisement, 
            Food tech (general), Health (general), Insects as food, Kitchen technology, Lab-grown meat, Meal kits, 
            Personalised nutrition, Restaurants, Food retail, Supply chain, Weight management, Weight-loss drugs, 
            Food reformulation (general), Food reformulation (sugar), Food reformulation (salt), Food reformulation (fat), Food reformulation (fiber)
            """
            
        else:
            # Generic instructions for unknown missions
            return """
            Mark the text as 'yes' if (one or more of the following):
            - If the topic/area defined by the scope above is the main focus of the company or business
            - If the topic is one of the components or business areas described by the text
            - If the text describes a company where the topic is the main focus, or a significant part of the activities
            - If the text is about a component or critical element of the topic defined above
            - If the text describes products or services that can be applied to the topic defined above

            However, mark it as 'no' if (one or more of the following):
            - The activities described in the text do not have a discernible impact on or connection with the topic.
            - If the topic is mentioned only in passing or as a minor example
            - The topic is mentioned only as a negative example
            """