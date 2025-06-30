"""
GTR data fetching
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.getters import gtr
from discovery_utils.utils import keywords as kw
import time
import json
import logging
import asyncio
from discovery_utils.utils.llm import batch_check
from ..config_manager import get_pipeline_config

from .base import BaseDataSource
from .utils import run_llm_relevance_check

logger = logging.getLogger(__name__)


class GtrDataSource(BaseDataSource[gtr.GtrGetter]):
    """GtR data source implementation."""
    
    def __init__(self):
        super().__init__("gtr")
    
    def _create_default_getter(self) -> gtr.GtrGetter:
        """Create default GtrGetter instance."""
        return gtr.GtrGetter()
    
    def _get_item_type(self) -> str:
        """Return the item type for GtR (projects)."""
        return "projects"
    
    def _fetch_fresh_data(self, topic_name: str, config: Dict[str, Any], 
                         getter: gtr.GtrGetter, cache_dir: Path, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from GtR and return relevant project IDs."""
        
        enrichment_df = self._get_or_create_enrichment_data(getter, cache_dir)
        self.logger.info(f"Using enrichment data with {len(enrichment_df)} entries")
        
        # Get projects by category
        category_name = config['search_recipe']['category_name']
        projects_df = self._get_projects_from_config(getter, enrichment_df, config)
        
        self.logger.info(f"Found {len(projects_df)} projects for category '{category_name}' after pre-filtering")
        
        # Run LLM relevance check only on pre-filtered projects
        relevant_ids = self._run_relevance_check(projects_df, config, cache_dir, topic_name, getter)
        
        return relevant_ids, len(projects_df)
    
    def _get_or_create_enrichment_data(self, GTR: gtr.GtrGetter, cache_dir: Path) -> pd.DataFrame:
        """Get or create enrichment data"""
        enrichment_cache_file = cache_dir / "gtr_labelled_projects.csv"
        
        # Check if we have cached enrichment data
        if enrichment_cache_file.exists():
            self.logger.info("Loading cached GTR enrichment data")
            try:
                enrichment_df = (
                    pd.read_csv(enrichment_cache_file)
                    .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if isinstance(x, str) else []))
                    .assign(mission_labels=lambda df: df.mission_labels.apply(lambda x: x.split(",") if isinstance(x, str) else []))
                )
                return enrichment_df
            except Exception as e:
                self.logger.warning(f"Could not load cached enrichment data: {e}. Will regenerate.")
        
        self.logger.info("Generating GTR enrichment data using discovery_utils.utils.keywords (this may take 5-6 minutes)")
        pipeline_config = get_pipeline_config()
        start_date = pipeline_config.data_start_date
        end_date = pipeline_config.data_end_date
        
        new_projects = (
            GTR.projects_enriched
            .query("(start >= @start_date and start <= @end_date)")
        )
        new_projects_text = GTR.get_projects_text().query("id in @new_projects.id.to_list()")
        
        enrichment_df = kw.enrich_topic_labels(new_projects_text)  # This will take 5-6 minutes
        
        # Cache the enrichment data
        cache_dir.mkdir(parents=True, exist_ok=True)
        enrichment_df.to_csv(enrichment_cache_file, index=False)
        self.logger.info(f"Generated and cached GTR enrichment data with {len(enrichment_df)} entries")
        
        return enrichment_df
    
    def _get_projects_in_nesta_categories(self, GTR: gtr.GtrGetter, enrichment_df: pd.DataFrame, 
                                        category_type: str, categories: List[str]) -> pd.DataFrame:
        """Get all projects belonging to the provided categories"""
        matching_ids = (
            enrichment_df
            .explode(category_type)
            .query(f"{category_type} in @categories")
            .id.to_list()
        )
        return GTR.projects_enriched.query("id in @matching_ids").drop_duplicates(subset="id")
    
    def _get_projects_from_config(self, GTR: gtr.GtrGetter, enrichment_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Get projects from config"""
        category_name = config["search_recipe"]["category_name"]
        return self._get_projects_in_nesta_categories(GTR, enrichment_df, "topic_labels", [category_name])
    
    def _run_relevance_check(self, projects_df: pd.DataFrame, config: Dict, 
                           cache_dir: Path, topic_name: str, gtr_getter: gtr.GtrGetter) -> List[str]:
        """Run LLM relevance check for GtR projects."""
        
        # Get projects text for relevance checking
        selected_texts_df = gtr_getter.get_projects_text().query("id in @projects_df.id.to_list()")
        
        # Merge projects with their text data
        projects_with_text = (
            projects_df
            .merge(selected_texts_df[['id', 'text']], left_on='id', right_on='id', how='left')
        )
        
        # Define custom instructions for GtR
        custom_instructions = """ 
        Mark the text as 'yes' if (one or more of the following):
        - If the technology defined by the scope above, is the main focus
        - If the technology is one of the components or activities described by the text. For example the technology could be  mentioned
            as part of a larger project or business including other technologies, or be mentioned as one of the use cases or case studies.
        - If the text describes a project and the technology is the main focus, or a part of a broader range of the project's activities and offerings.
        - If the text is about a component or critical element of the technology defined above
        - If the text describes a technology or process and explicitly mentions that it can be applied on the technology defined above to improve it's performance or efficiency.

        If the text is about heating technology but application target is not mentioned then assume it could be relevant for households or buildings (as opposed to an industrial applications).

        However, mark it as 'no' if (one or more of the following):
        - The activities, or research described in the text does not have a discernable impact on or connection with the technology.
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
            projects_with_text,
            config,
            cache_dir,
            topic_name,
            "gtr",
            custom_instructions=custom_instructions
        )
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate GtR configuration."""
        if 'search_recipe' not in config:
            raise ValueError("Missing 'search_recipe' in config")
        
        search_recipe = config['search_recipe']
        if 'category_name' not in search_recipe:
            raise ValueError("Missing 'category_name' in search_recipe")
        
        category_name = search_recipe['category_name']
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("'category_name' must be a non-empty string") 