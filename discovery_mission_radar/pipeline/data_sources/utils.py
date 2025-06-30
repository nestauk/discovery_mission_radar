"""
Shared utilities for data sources in the Mission Radar Pipeline.

This module contains common functionality used across multiple data sources,
particularly for LLM relevance checking and async processing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from discovery_utils.utils.llm import batch_check

logger = logging.getLogger(__name__)


async def run_llm_relevance_check_async(
    items_df: pd.DataFrame, 
    config: Dict, 
    cache_dir: Path, 
    topic_name: str,
    source_name: str,
    id_column: str = 'id',
    text_column: str = 'text',
    custom_instructions: str = ""
) -> List[str]:
    """Run async LLM relevance check with caching.
    
    This function provides a common interface for LLM relevance checking
    across all data sources with intelligent caching to avoid re-processing
    items that have already been checked.
    
    Args:
        items_df: DataFrame containing items to check (must have id_column)
        config: Topic configuration dictionary
        cache_dir: Directory for caching results
        topic_name: Name of the topic being processed
        source_name: Name of the data source (for cache file naming)
        id_column: Name of the ID column in items_df
        text_column: Name of the text column in items_df  
        custom_instructions: Additional instructions for the LLM
        
    Returns:
        List of IDs that were marked as relevant
    """
    # Check for existing relevance cache
    relevance_cache_file = cache_dir / f"{topic_name}_{source_name}_relevance_check.jsonl"
    
    # Get all current item IDs that need to be checked
    current_item_ids = set(items_df[id_column].tolist())
    
    # Load existing results if available
    existing_results = {}
    if relevance_cache_file.exists():
        try:
            existing_df = pd.read_json(relevance_cache_file, lines=True)
            existing_results = dict(zip(existing_df['id'], existing_df['is_relevant']))
            logger.info(f"Loaded {len(existing_results)} existing {source_name} relevance check results for {topic_name}")
        except Exception as e:
            logger.warning(f"Could not read existing {source_name} relevance check results: {e}")
            existing_results = {}
    
    # Identify entities that need LLM checking
    already_checked_ids = set(existing_results.keys())
    new_ids_to_check = current_item_ids - already_checked_ids
    
    logger.info(f"{source_name} relevance check status for {topic_name}:")
    logger.info(f"  Current entities: {len(current_item_ids)}")
    logger.info(f"  Already checked: {len(already_checked_ids)}")
    logger.info(f"  New entities to check: {len(new_ids_to_check)}")
    
    # Run LLM check only on new entities (if any)
    if new_ids_to_check:
        logger.info(f"Running {source_name} LLM relevance check on {len(new_ids_to_check)} new entities for {topic_name}")
        
        # Get text data for only the new entities
        new_items_df = items_df[items_df[id_column].isin(new_ids_to_check)]
        check_data = dict(zip(new_items_df[id_column], new_items_df[text_column]))
        
        # Generate system message
        system_message = batch_check.generate_relevance_check_system_message(config)
        
        # Add custom instructions if provided
        if custom_instructions:
            system_message += custom_instructions
        
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
        
        try:
            await _llm_processing(processor, check_data, relevance_cache_file)
        except Exception as e:
            logger.error(f"All LLM processing attempts failed: {e}")
            raise
            
        relevant_check_df = pd.read_json(relevance_cache_file, lines=True)
        logger.info(f"LLM relevance check completed for {topic_name}: {len(new_ids_to_check)} new entities processed")
    else:
        logger.info(f"All entities already checked for {topic_name}, skipping LLM calls")
        relevant_check_df = pd.DataFrame([
            {'id': entity_id, 'is_relevant': is_relevant}
            for entity_id, is_relevant in existing_results.items()
        ])
    
    # Merge results and filter for relevant items
    relevant_checked_df = (
        items_df
        .merge(relevant_check_df[['id', 'is_relevant']], left_on=id_column, right_on='id', how='left')
    )
    
    items_without_results = relevant_checked_df['is_relevant'].isna().sum()
    if items_without_results > 0:
        logger.warning(f"Found {items_without_results} items without relevance check results - they will be excluded")
    
    matching_ids = relevant_checked_df.query("is_relevant == 'yes'")[id_column].tolist()
    logger.info(f"Relevance check complete: {len(matching_ids)} relevant items out of {len(items_df)} total")
    return matching_ids


def run_llm_relevance_check(
    items_df: pd.DataFrame, 
    config: Dict, 
    cache_dir: Path, 
    topic_name: str,
    source_name: str,
    id_column: str = 'id',
    text_column: str = 'text',
    custom_instructions: str = ""
) -> List[str]:
    """Synchronous wrapper for async LLM relevance check.
    
    Args:
        items_df: DataFrame containing items to check
        config: Topic configuration dictionary
        cache_dir: Directory for caching results
        topic_name: Name of the topic being processed
        source_name: Name of the data source
        id_column: Name of the ID column in items_df
        text_column: Name of the text column in items_df
        custom_instructions: Additional instructions for the LLM
        
    Returns:
        List of IDs that were marked as relevant
    """
    return asyncio.run(run_llm_relevance_check_async(
        items_df, config, cache_dir, topic_name, source_name, 
        id_column, text_column, custom_instructions
    ))


async def _llm_processing(processor, check_data: Dict, output_file: Path) -> None:
    """Process data through LLM with retry logic.
    
    Args:
        processor: LLMProcessor instance
        check_data: Dictionary of {id: text} to process
        output_file: Path to output file
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logger.info(f"LLM processing attempt {attempt + 1}/{max_retries}")
            await processor.process_text_data(check_data, batch_size=10, sleep_time=0.5)
            logger.info("LLM processing completed successfully")
            return
        except Exception as e:
            logger.warning(f"LLM processing attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All LLM processing attempts failed")
                raise
            else:
                logger.info(f"Retrying LLM processing (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(5)  # Wait 5 seconds before retry 