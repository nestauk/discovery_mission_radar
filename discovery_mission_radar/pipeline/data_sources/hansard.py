"""
Hansard data fetching
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils.llm import batch_check
import asyncio
import time
import json
import logging
import re
from ..config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

def get_hansard_data(topic_name: str, cache_dir: Path, config: Dict[str, Any], hansard_getter=None, use_llm_check: bool = False) -> Dict[str, Any]:
    """
    Get Hansard speech IDs for a topic with caching.
    
    Mirrors notebook get_ids_from_config() pattern exactly.
    
    Args:
        topic_name: Name of the topic (e.g., "hydrogen_energy")
        cache_dir: Directory for caching results
        config: Topic configuration dictionary
        hansard_getter: Pre-initialized HansardGetter (optional)
        use_llm_check: Whether to use LLM relevance checking (False by default, like original notebooks)
        
    Returns:
        {
            'ids': List[str],           # Relevant speech IDs
            'topic_name': str,          # Topic name
            'config': Dict,             # Original config
            'speeches_count': int,      # Total speeches found
            'relevant_count': int       # Speeches after relevance check
        }
    """
    cache_file = cache_dir / f"{topic_name}_hansard_data.json"
    
    # Check cache first
    if _is_cache_valid(cache_file, max_age_hours=24):
        logger.info(f"Loading cached Hansard data for {topic_name}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Processing fresh Hansard data for {topic_name}")
    
    # Fresh data processing
    if hansard_getter is None:
        HansardData = policy_update.HansardData()
    else:
        HansardData = hansard_getter
    
    # Get speeches by category
    category_name = config['search_recipe']['category_name']
    speeches_df = _get_speeches_data(HansardData, category_name)
    
    logger.info(f"Found {len(speeches_df)} speeches for topic '{category_name}'")
    
    if use_llm_check:
        # Run LLM relevance check
        logger.info("Running LLM relevance check (this may take a while)")
        relevant_ids = _run_relevance_check(speeches_df, config, cache_dir, topic_name)
    else:
        # Use simple topic label filtering
        logger.info("Using pre-existing topic labels (no LLM check)")
        relevant_ids = speeches_df['speech_id'].tolist()
    
    # Prepare result
    result = {
        'ids': relevant_ids,
        'topic_name': topic_name,
        'config': config,
        'speeches_count': len(speeches_df),
        'relevant_count': len(relevant_ids)
    }
    
    # Cache result
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Cached {len(relevant_ids)} relevant speeches for {topic_name}")
    return result

def _is_cache_valid(cache_file: Path, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is recent enough"""
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return age_hours < max_age_hours
    return False

def _get_speeches_data(HansardData, category_name: str) -> pd.DataFrame:
    """Get speeches data"""
    # Get configuration for dates and missions
    pipeline_config = get_pipeline_config()
    missions = [pipeline_config.current_mission]
    
    speeches_df = (
        HansardData.debates_df
        .query(pipeline_config.get_date_query_filter())
        .merge(
            HansardData.labelstore_df[['id', 'mission_labels', 'topic_labels']],
            left_on='speech_id',
            right_on='id',
            how='left'
        )
        .assign(mission_labels=lambda df: df.mission_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
        .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if (type(x) is str) else []))
        .explode("mission_labels")
        .query("mission_labels in @missions")
        .explode("topic_labels")
        .assign(quarter=lambda df: df.date.apply(_get_quarter_from_date))
        .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
    )
    
    # Filter by category/topic label
    filtered_speeches = (
        speeches_df
        .query(f"topic_labels == '{category_name}'")
        .drop_duplicates("speech_id")
    )
    
    logger.info(f"Found {len(filtered_speeches)} speeches for topic '{category_name}' after label filtering")
    return filtered_speeches

async def _run_relevance_check_async(speeches_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Run LLM relevance check"""
    # Check for existing relevance cache
    relevance_cache_file = cache_dir / f"{topic_name}_hansard_relevance_check.jsonl"
    
    # Get all current speech IDs that need to be checked
    current_speech_ids = set(speeches_df['speech_id'].tolist())
    
    # Load existing results if available
    existing_results = {}
    if relevance_cache_file.exists():
        try:
            existing_df = pd.read_json(relevance_cache_file, lines=True)
            existing_results = dict(zip(existing_df['id'], existing_df['is_relevant']))
            logger.info(f"Loaded {len(existing_results)} existing Hansard relevance check results for {topic_name}")
        except Exception as e:
            logger.warning(f"Could not read existing Hansard relevance check results: {e}")
            existing_results = {}
    
    # Identify entities that need LLM checking
    already_checked_ids = set(existing_results.keys())
    new_ids_to_check = current_speech_ids - already_checked_ids
    
    logger.info(f"Hansard relevance check status for {topic_name}:")
    logger.info(f"  Current entities: {len(current_speech_ids)}")
    logger.info(f"  Already checked: {len(already_checked_ids)}")
    logger.info(f"  New entities to check: {len(new_ids_to_check)}")
    
    # Run LLM check only on new entities (if any)
    if new_ids_to_check:
        logger.info(f"Running Hansard LLM relevance check on {len(new_ids_to_check)} new entities for {topic_name}")
        
        # Prepare text data for LLM checking - only for new entities
        text_data = {}
        for _, row in speeches_df.iterrows():
            if row['speech_id'] in new_ids_to_check:
                speech_text = str(row.get('speech', ''))
                # Normalize text
                normalized_text = re.sub(r"\s+", " ", speech_text)
                text_data[row['speech_id']] = normalized_text
        
        # Generate system message using existing method
        system_message = batch_check.generate_relevance_check_system_message(config)
        
        # Add custom instructions for parliamentary speeches
        system_message += """
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
        
        fields = [
            {"name": "explanation", "type": "str", "description": "A short, 1-sentence explanation of the answer (max 25 words)."},
            {"name": "is_relevant", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},    
        ]

        processor = batch_check.LLMProcessor(
            output_path=str(relevance_cache_file),
            system_message=system_message,
            session_name="mission_studio",
            output_fields=fields,
        )
        
        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the async check with progressive robustness to handle large datasets
        try:
            await _llm_processing(processor, text_data, relevance_cache_file)
        except Exception as e:
            logger.error(f"All Hansard LLM processing attempts failed: {e}")
            raise
        
        # Reload all results (existing + new)
        relevant_check_df = pd.read_json(relevance_cache_file, lines=True)
        logger.info(f"Hansard LLM relevance check completed for {topic_name}: {len(new_ids_to_check)} new entities processed")
    else:
        logger.info(f"All Hansard entities already checked for {topic_name}, skipping LLM calls")
        relevant_check_df = pd.DataFrame([
            {'id': entity_id, 'is_relevant': is_relevant}
            for entity_id, is_relevant in existing_results.items()
        ])
    
    # Filter for relevant speeches
    matching_ids = relevant_check_df.query("is_relevant == 'yes'")['id'].tolist()
    
    # Only include IDs that are in our current speech dataset
    current_matching_ids = [speech_id for speech_id in matching_ids if speech_id in current_speech_ids]
    
    speeches_without_results = len(current_speech_ids) - len(set(relevant_check_df['id'].tolist()) & current_speech_ids)
    if speeches_without_results > 0:
        logger.warning(f"Found {speeches_without_results} Hansard speeches without relevance check results - they will be excluded")
    
    logger.info(f"Hansard relevance check complete: {len(current_matching_ids)} relevant speeches out of {len(speeches_df)} total")
    return current_matching_ids

def _run_relevance_check(speeches_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Synchronous wrapper for async relevance check"""
    return asyncio.run(_run_relevance_check_async(speeches_df, config, cache_dir, topic_name))

def _get_quarter_from_date(date: str) -> int:
    """Return the quarter number from a given YYYY-MM-DD date string."""
    import datetime
    _date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return (_date.month-1)//3 + 1

def validate_hansard_config(config: Dict[str, Any]) -> None:
    """Basic validation - fail fast on missing required fields"""
    if 'search_recipe' not in config:
        raise ValueError("Missing required Hansard config field: search_recipe")
    
    if 'category_name' not in config['search_recipe']:
        raise ValueError("Missing required Hansard config field: search_recipe.category_name")

async def _llm_processing(processor, check_data: Dict, output_file: Path) -> None:
    """
    Robust LLM processing with timeout detection and automatic recovery.
    Handles hanging/stuck LLM processors by using timeouts and batch reduction.
    """
    # Check how many items are already processed
    processed_ids = set()
    if output_file.exists():
        try:
            existing_df = pd.read_json(output_file, lines=True)
            processed_ids = set(existing_df['id'].tolist())
            logger.info(f"Found {len(processed_ids)} already processed items, resuming...")
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}")
    
    # Filter out already processed items
    remaining_data = {k: v for k, v in check_data.items() if k not in processed_ids}
    total_items = len(check_data)
    remaining_items = len(remaining_data)
    
    if remaining_items == 0:
        logger.info("All items already processed!")
        return
        
    logger.info(f"Processing {remaining_items}/{total_items} remaining items")
    
    batch_size = 10
    sleep_time = 0.5
    timeout_minutes = 5
    
    try:
        await asyncio.wait_for(
            processor.process_text_data(remaining_data, batch_size=batch_size, sleep_time=sleep_time),
            timeout=timeout_minutes * 60
        )
        logger.info("LLM processing completed successfully!")
        
    except asyncio.TimeoutError:
        logger.error(f"LLM processing timed out after {timeout_minutes} minutes!")
        raise
    except Exception as e:
        logger.error(f"LLM processing failed with error: {e}")
        raise 