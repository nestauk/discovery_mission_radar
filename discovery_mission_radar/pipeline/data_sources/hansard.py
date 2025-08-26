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
from .utils import run_llm_relevance_check, load_llm_prompts

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
            relevant_ids = self._run_relevance_check(speeches_df, config, cache_dir, topic_name, mission, getter=getter, **kwargs)
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
                           cache_dir: Path, topic_name: str, mission: str = None, getter: hansard.HansardGetter = None, **kwargs) -> List[str]:
        """Run LLM relevance check for Hansard speeches."""
        
        # Build mission- and topic-aware custom instructions
        custom_instructions = self._build_custom_instructions(mission or "Unknown", config, topic_name)
        
        # Prepare speeches with text and neighbour context for LLM check
        speeches_with_text = speeches_df.copy()
        speeches_with_text['speech'] = speeches_with_text['speech'].apply(lambda x: re.sub(r"\s+", " ", str(x)))

        try:
            # Build previous/next speech context from full debates
            full_debates_df = getter.get_debates_parquet() if getter else None
        except Exception:
            full_debates_df = None

        if full_debates_df is not None and not full_debates_df.empty:
            # Keep original order; ensure clean text
            ordered = full_debates_df.copy()
            ordered['speech'] = ordered['speech'].apply(lambda x: re.sub(r"\s+", " ", str(x)))
            ordered = ordered[['speech_id', 'speakername', 'speech']].reset_index(drop=True)
            ordered['prev_speech'] = ordered['speech'].shift(1)
            ordered['prev_speaker'] = ordered['speakername'].shift(1)
            ordered['next_speech'] = ordered['speech'].shift(-1)
            ordered['next_speaker'] = ordered['speakername'].shift(-1)
            neighbour_cols = ['speech_id', 'prev_speech', 'prev_speaker', 'next_speech', 'next_speaker']
            speeches_with_text = speeches_with_text.merge(ordered[neighbour_cols], on='speech_id', how='left')
            
            def _compose_context(row):
                prev_block = (
                    f"# PREVIOUS SPEECH\nSpeaker: {row['prev_speaker']}\nFull speech: {row['prev_speech']}\n"
                    if isinstance(row.get('prev_speech'), str) and len(row.get('prev_speech')) > 0
                    else "# PREVIOUS SPEECH\nNo previous speech available.\n"
                )
                cur_block = (
                    f"# SPEECH\nSpeaker: {row['speakername']}\nFull speech: {row['speech']}\n"
                )
                next_block = (
                    f"# NEXT SPEECH\nSpeaker: {row['next_speaker']}\nFull speech: {row['next_speech']}\n"
                    if isinstance(row.get('next_speech'), str) and len(row.get('next_speech')) > 0
                    else "# NEXT SPEECH\nNo next speech available.\n"
                )
                return prev_block + cur_block + next_block
            speeches_with_text['text_with_context'] = speeches_with_text.apply(_compose_context, axis=1)
            text_column = 'text_with_context'
        else:
            # Fallback to using only the current speech text
            speeches_with_text['text'] = speeches_with_text['speech']
            text_column = 'text'
        
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
            text_column=text_column,
            custom_instructions=custom_instructions
        )
    
    def _get_mission_specific_instructions(self, mission: str) -> str:
        """Get mission/source instructions from central YAML; no in-code prompts."""
        try:
            base_config_dir = Path(__file__).parent.parent.parent / "config"
            prompts = load_llm_prompts(base_config_dir)
            key = mission if mission in prompts else "default"
            mission_prompts = prompts.get(key, {})
            source_key = getattr(self, 'source_name', '').lower() or 'hansard'
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

    def _build_custom_instructions(self, mission: str, topic_config: Dict, topic_name: str) -> str:
        """Return only mission/source YAML prompt (no extra in-code guidance)."""
        return self._get_mission_specific_instructions(mission)


def _get_quarter_from_date(date: str) -> int:
    """Get quarter from date string"""
    month = pd.to_datetime(date).month
    return (month - 1) // 3 + 1 