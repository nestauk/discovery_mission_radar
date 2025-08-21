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
from .utils import run_llm_relevance_check, load_llm_prompts

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
                         getter: gtr.GtrGetter, cache_dir: Path, 
                         mission: str = None, **kwargs) -> tuple[List[str], int]:
        """Fetch fresh data from GtR and return relevant project IDs."""
        
        enrichment_df = self._get_or_create_enrichment_data(getter, cache_dir, mission)
        self.logger.info(f"Using enrichment data with {len(enrichment_df)} entries")
        
        # Get projects by category
        category_name = config['search_recipe']['category_name']
        projects_df = self._get_projects_from_config(getter, enrichment_df, config)
        
        self.logger.info(f"Found {len(projects_df)} projects for category '{category_name}' after pre-filtering")
            
        relevant_ids = self._run_relevance_check(projects_df, config, cache_dir, topic_name, getter, mission, **kwargs)
        
        return relevant_ids, len(projects_df)
    
    def _get_or_create_enrichment_data(self, GTR: gtr.GtrGetter, cache_dir: Path, mission: str) -> pd.DataFrame:
        """Get or create enrichment data"""
        enrichment_cache_file = cache_dir / "gtr_labelled_projects.csv"
        
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
        
        self.logger.info("Enriching GTR data")
        pipeline_config = get_pipeline_config(mission)
        start_date = pipeline_config.data_start_date
        end_date = pipeline_config.data_end_date
        
        new_projects = (
            GTR.projects_enriched
            .query("(start >= @start_date and start <= @end_date)")
        )
        new_projects_text = GTR.get_projects_text().query("id in @new_projects.id.to_list()")
        
        enrichment_df = kw.enrich_topic_labels(new_projects_text) 
        
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
                           cache_dir: Path, topic_name: str, gtr_getter: gtr.GtrGetter,
                           mission: str = None, **kwargs) -> List[str]:
        """Run LLM relevance check for GtR projects."""
        
        # Get projects text for relevance checking
        selected_texts_df = gtr_getter.get_projects_text().query("id in @projects_df.id.to_list()")
        
        # Merge projects with their text data
        projects_with_text = (
            projects_df
            .merge(selected_texts_df[['id', 'text']], left_on='id', right_on='id', how='left')
        )
        
        # Get mission-aware custom instructions
        custom_instructions = self._get_mission_specific_instructions(mission or "Unknown")
        
        return run_llm_relevance_check(
            projects_with_text,
            config,
            cache_dir,
            topic_name,
            "gtr",
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
            source_key = getattr(self, 'source_name', '').lower() or 'gtr'
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