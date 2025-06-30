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
                         getter: crunchbase.CrunchbaseGetter, cache_dir: Path, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from Crunchbase and return relevant company IDs."""
        
        # Get companies by category
        category_name = config['search_recipe']['category_name']
        companies_df = getter.get_companies_in_nesta_categories("topic_labels", [category_name])
        
        self.logger.info(f"Found {len(companies_df)} companies for category '{category_name}'")
        
        # Run LLM relevance check
        relevant_ids = self._run_relevance_check(companies_df, config, cache_dir, topic_name, getter)
        
        return relevant_ids, len(companies_df)
    
    def _run_relevance_check(self, companies_df: pd.DataFrame, config: Dict, 
                            cache_dir: Path, topic_name: str, cb_getter: crunchbase.CrunchbaseGetter) -> List[str]:
        """Run LLM relevance check for Crunchbase companies."""
        
        # Get organisation text for relevance checking
        selected_texts_df = cb_getter.get_organisation_text(companies_df)
        
        # Merge companies with their text data
        companies_with_text = (
            companies_df
            .merge(selected_texts_df[['id', 'text']], left_on='id', right_on='id', how='left')
        )
        
        # Define custom instructions for Crunchbase
        custom_instructions = """ 
        Mark the text as 'yes' if (one or more of the following):
        - If the technology defined by the scope above, is the main focus
        - If the technology is one of the components or activities described by the text. For example the technology could be  mentioned
            as part of a larger project or business including other technologies, or be mentioned as one of the use cases or case studies.
        - If the text describes a company and the technology is the main focus, or a part of a broader range of the company's activities and offerings.
        - If the text is about a component or critical element of the technology defined above
        - If the text describes a technology or process and explicitly mentions that it can be applied on the technology defined above to improve it's performance or efficiency.

        If the text is about heating technology but application target is not mentioned then assume it could be relevant for households or buildings (as opposed to an industrial applications).

        However, mark it as 'no' if (one or more of the following):
        - The activities, or business described in the text does not have a discernable impact on or connection with the technology.
        - If the technology is mentioned only in passing or as a minor example in a broader discussion, for example, 
            in only one sentence within a long text with many sentences, or at the very end of a long description.
        - The technology is mentioned only as a negative example (eg "unlike [technology]...")
        - The text mentions heat pumps for heating swimming pools  
        - The text would be better captured by one of the other categories (comma separated) mentioned in this list:  
        Bioenergy (biofuels), Biomass heating, Carbon capture and storage, District heating and heat networks, Energy grid, Geothermal energy, 
        Heat pumps, Hydrogen energy, Hydrogen heating, Micro CHP, Solar thermal heating, Energy storage (batteries), Solar power, Wind power
        """
        
        # Use shared LLM relevance check function
        return run_llm_relevance_check(
            companies_with_text,
            config,
            cache_dir,
            topic_name,
            "crunchbase",
            custom_instructions=custom_instructions
        )
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate Crunchbase configuration."""
        if 'search_recipe' not in config:
            raise ValueError("Missing 'search_recipe' in config")
        
        search_recipe = config['search_recipe']
        if 'category_name' not in search_recipe:
            raise ValueError("Missing 'category_name' in search_recipe")
        
        category_name = search_recipe['category_name']
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("'category_name' must be a non-empty string")