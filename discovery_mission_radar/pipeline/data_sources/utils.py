"""
Shared utilities for data sources in the Mission Radar Pipeline.

This module contains common functionality used across multiple data sources,
particularly for LLM relevance checking and async processing with S3 caching.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from discovery_utils.utils.llm import batch_check
from discovery_utils.utils.s3 import s3_client, _download_obj, upload_obj

logger = logging.getLogger(__name__)


class S3CacheManager:
    """Manages S3 caching for LLM relevance check results."""
    
    def __init__(self, mission: str, bucket_name: Optional[str] = None):
        """Initialise S3 cache manager.
        
        Args:
            mission: Current mission (AHL/ASF)
            bucket_name: S3 bucket name (defaults to S3_BUCKET env var)
        """
        self.mission = mission
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET")
        if not self.bucket_name:
            logger.warning("S3_BUCKET environment variable not set. S3 caching disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.s3 = s3_client()
    
    def get_s3_path(self, topic_name: str, source_name: str) -> str:
        """Generate S3 path for cache file.
        
        Args:
            topic_name: Name of the topic
            source_name: Name of the data source
            
        Returns:
            S3 path for the cache file
        """
        return f"data/mission_radar/llm_relevance_check/{self.mission}/{source_name}/{topic_name}_{source_name}_relevance_check.jsonl"
    
    def download_cache_from_s3(self, topic_name: str, source_name: str, local_file: Path) -> bool:
        """Download cache file from S3 to local path.
        
        Args:
            topic_name: Name of the topic
            source_name: Name of the data source  
            local_file: Local file path to save to
            
        Returns:
            True if download successful, False otherwise
        """
        if not self.enabled:
            return False
            
        s3_path = self.get_s3_path(topic_name, source_name)
        
        try:
            logger.info(f"Attempting to download cache from S3: {s3_path}")
            
            # Use boto3 directly to download the file to avoid file extension limitations
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_path)
            content = response['Body'].read().decode('utf-8')
            
            # Ensure local directory exists
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to local file
            with open(local_file, 'w') as f:
                f.write(content)
                
            logger.info(f"Successfully downloaded cache from S3 to {local_file}")
            return True
            
        except Exception as e:
            logger.info(f"Could not download cache from S3: {e}")
            return False
    
    async def upload_cache_to_s3_async(self, topic_name: str, source_name: str, local_file: Path) -> bool:
        """Upload cache file from local path to S3 asynchronously.
        
        Args:
            topic_name: Name of the topic
            source_name: Name of the data source
            local_file: Local file path to upload from
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.enabled or not local_file.exists():
            return False
            
        s3_path = self.get_s3_path(topic_name, source_name)
        
        try:
            logger.debug(f"Uploading cache to S3: {s3_path}")
            
            # Read file content as bytes
            with open(local_file, 'rb') as f:
                content = f.read()
            
            # Upload asynchronously using boto3 directly to avoid file extension limitations
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_path,
                    Body=content,
                    ContentType='application/x-ndjson'  # Proper MIME type for JSONL
                )
            )
            
            logger.debug(f"Successfully uploaded cache to S3: {s3_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not upload cache to S3: {e}")
            return False


async def run_llm_relevance_check_async(
    items_df: pd.DataFrame, 
    config: Dict, 
    cache_dir: Path, 
    topic_name: str,
    source_name: str,
    mission: str,
    pipeline_config: Optional[Dict] = None,
    id_column: str = 'id',
    text_column: str = 'text',
    custom_instructions: str = ""
) -> List[str]:
    """Run async LLM relevance check with S3 and local caching, with optional Argilla integration.
    
    This function provides a common interface for LLM relevance checking
    across all data sources with intelligent caching to avoid re-processing
    items that have already been checked. Uses S3 for persistent storage
    with local fallback. Integrates with Argilla for manual review if enabled.
    
    Args:
        items_df: DataFrame containing items to check (must have id_column)
        config: Topic configuration dictionary
        cache_dir: Directory for caching results
        topic_name: Name of the topic being processed
        source_name: Name of the data source (for cache file naming)
        mission: Current mission (AHL/ASF)
        pipeline_config: Pipeline configuration
        id_column: Name of the ID column in items_df
        text_column: Name of the text column in items_df  
        custom_instructions: Additional instructions for the LLM
        
    Returns:
        List of IDs that were marked as relevant
    """
    # Initialise S3 cache manager
    s3_cache = S3CacheManager(mission)
    
    # Local cache file path
    relevance_cache_file = cache_dir / f"{topic_name}_{source_name}_relevance_check.jsonl"
    
    # Try to download from S3 first, then fall back to local cache
    s3_downloaded = s3_cache.download_cache_from_s3(topic_name, source_name, relevance_cache_file)
    if s3_downloaded:
        logger.info(f"Downloaded {source_name} cache from S3 for {topic_name}")
    elif relevance_cache_file.exists():
        logger.info(f"Using local {source_name} cache for {topic_name}")
    else:
        logger.info(f"No existing {source_name} cache found for {topic_name}")
    
    # Argilla: Import completed manual reviews and apply overrides
    if pipeline_config and pipeline_config.get('argilla', {}).get('enabled', False):
        try:
            from .argilla import import_from_argilla, apply_manual_overrides
            
            quarter = pipeline_config.get('current_period', {}).get('quarter', '2025-Q2')
            argilla_config = pipeline_config['argilla']
            
            logger.info(f"Argilla integration enabled - importing manual reviews for {topic_name}")
            manual_reviews = import_from_argilla(topic_name, quarter, mission, argilla_config, source_name)
            
            if manual_reviews:
                logger.info(f"Found {len(manual_reviews)} completed manual reviews for {topic_name}")
                apply_manual_overrides(relevance_cache_file, manual_reviews)
            else:
                logger.info(f"No completed manual reviews found for {topic_name}")
                
        except ImportError:
            logger.warning("Argilla package not available - skipping manual review import")
        except Exception as e:
            logger.warning(f"Failed to import manual reviews from Argilla: {e}")
    
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
            await _llm_processing_with_s3_upload(
                processor, check_data, relevance_cache_file, s3_cache, topic_name, source_name
            )
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
    
    # Argilla: Sample and export new entities for manual review
    if pipeline_config and pipeline_config.get('argilla', {}).get('enabled', False) and new_ids_to_check:
        try:
            from .argilla import get_entities_to_sample, export_to_argilla
            
            quarter = pipeline_config.get('current_period', {}).get('quarter', '2025-Q2')
            argilla_config = pipeline_config['argilla']
            sample_size = argilla_config.get('sampling', {}).get('base_sample_size', 10)
            
            logger.info(f"Argilla sampling enabled - selecting entities for manual review")
            
            # Get entities to sample from the newly processed ones
            new_items_with_results = items_df[items_df[id_column].isin(new_ids_to_check)].copy()
            new_results_df = relevant_check_df[relevant_check_df['id'].isin(new_ids_to_check)]
            
            # Merge LLM results with entity data for sampling
            entities_for_sampling = new_items_with_results.merge(
                new_results_df[['id', 'is_relevant', 'explanation']], 
                left_on=id_column, 
                right_on='id', 
                how='inner'
            )
            
            if len(entities_for_sampling) > 0:
                entities_to_sample = await get_entities_to_sample(
                    topic_name, quarter, mission, entities_for_sampling, 
                    sample_size, cache_dir, s3_cache, source_name
                )
                
                if entities_to_sample:
                    export_success = export_to_argilla(
                        entities_to_sample, entities_for_sampling, 
                        topic_name, quarter, mission, argilla_config, source_name
                    )
                    
                    if export_success:
                        logger.info(f"Exported {len(entities_to_sample)} entities to Argilla for {topic_name}")
                    else:
                        logger.warning(f"Failed to export entities to Argilla")
                else:
                    logger.debug(f"No new entities selected for sampling (avoiding duplicates)")
            else:
                logger.debug(f"No entities available for Argilla sampling")
                
        except ImportError:
            logger.warning("Argilla package not available - skipping sampling and export")
        except Exception as e:
            logger.warning(f"Failed to sample and export to Argilla: {e}")
    
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
    mission: str,
    pipeline_config: Optional[Dict] = None,
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
        mission: Current mission (AHL/ASF)
        pipeline_config: Pipeline configuration for Argilla integration (optional)
        id_column: Name of the ID column in items_df
        text_column: Name of the text column in items_df
        custom_instructions: Additional instructions for the LLM
        
    Returns:
        List of IDs that were marked as relevant
    """
    return asyncio.run(run_llm_relevance_check_async(
        items_df, config, cache_dir, topic_name, source_name, mission,
        pipeline_config, id_column, text_column, custom_instructions
    ))


async def _llm_processing_with_s3_upload(
    processor, 
    check_data: Dict, 
    output_file: Path, 
    s3_cache: S3CacheManager,
    topic_name: str,
    source_name: str
) -> None:
    """Process data through LLM with adaptive batching strategy and S3 uploads.
    
    Uses two-stage batching for large datasets (>100 items) to prevent getting stuck,
    and discovery-utils native batching for smaller datasets for simplicity.
    
    Args:
        processor: LLMProcessor instance
        check_data: Dictionary of {id: text} to process
        output_file: Path to output file
        s3_cache: S3 cache manager instance
        topic_name: Name of the topic
        source_name: Name of the data source
    """
    max_retries = 3
    data_size = len(check_data)
    use_two_stage = data_size > 100  # Use two-stage batching for larger datasets
    
    for attempt in range(max_retries):
        try:
            logger.info(f"LLM processing attempt {attempt + 1}/{max_retries} ({data_size} items, {'two-stage' if use_two_stage else 'native'} batching)")
            
            if use_two_stage:
                # Two-stage batching: we control the outer batching, discovery-utils handles inner batching
                await _process_with_two_stage_batching(
                    processor, check_data, output_file, s3_cache, topic_name, source_name
                )
            else:
                # Native batching: let discovery-utils handle everything with periodic S3 uploads
                await _process_with_native_batching(
                    processor, check_data, output_file, s3_cache, topic_name, source_name
                )
            
            logger.info("LLM processing completed successfully")
            return
            
        except Exception as e:
            logger.warning(f"LLM processing attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All LLM processing attempts failed")
                raise
            else:
                # Exponential backoff: 2^attempt seconds (minimum 10s)
                wait_time = max(10, 2 ** attempt)
                logger.info(f"Retrying LLM processing (attempt {attempt + 2}/{max_retries}) after {wait_time}s")
                await asyncio.sleep(wait_time)


async def _process_with_native_batching(
    processor,
    check_data: Dict,
    output_file: Path,
    s3_cache: S3CacheManager,
    topic_name: str,
    source_name: str
) -> None:
    """Process small datasets using discovery-utils native batching with periodic S3 uploads."""
    # Start periodic S3 upload task in background
    upload_task = asyncio.create_task(
        _periodic_s3_upload(s3_cache, topic_name, source_name, output_file)
    )
    
    try:
        # Let discovery-utils handle all batching internally
        await processor.process_text_data(check_data)
        
    finally:
        # Cancel periodic uploads and do final upload
        upload_task.cancel()
        try:
            await upload_task
        except asyncio.CancelledError:
            pass
        
        # Final upload to S3 after completion
        await s3_cache.upload_cache_to_s3_async(topic_name, source_name, output_file)


async def _process_with_two_stage_batching(
    processor,
    check_data: Dict,
    output_file: Path,
    s3_cache: S3CacheManager,
    topic_name: str,
    source_name: str
) -> None:
    """Process large datasets using two-stage batching for better control and progress visibility."""
    outer_batch_size = 50  # Our outer batch size
    max_inner_retries = 3  # Retries per outer batch
    
    # Convert check_data to list of items for outer batch processing
    items = list(check_data.items())
    total_items = len(items)
    
    logger.info(f"Processing {total_items} items in outer batches of {outer_batch_size}")
    
    for i in range(0, total_items, outer_batch_size):
        outer_batch_items = items[i:i + outer_batch_size]
        outer_batch_data = dict(outer_batch_items)
        outer_batch_num = (i // outer_batch_size) + 1
        total_outer_batches = (total_items + outer_batch_size - 1) // outer_batch_size
        
        logger.info(f"Processing outer batch {outer_batch_num}/{total_outer_batches} ({len(outer_batch_items)} items)")
        
        # Retry logic per outer batch
        for inner_attempt in range(max_inner_retries):
            try:
                # Process this outer batch with discovery-utils (which will create its own inner batches)
                await processor.process_text_data(outer_batch_data)
                logger.debug(f"Outer batch {outer_batch_num} completed successfully")
                break  # Success, move to next outer batch
                
            except Exception as e:
                logger.warning(f"Outer batch {outer_batch_num} attempt {inner_attempt + 1}/{max_inner_retries} failed: {e}")
                if inner_attempt == max_inner_retries - 1:
                    logger.error(f"Outer batch {outer_batch_num} failed after {max_inner_retries} attempts")
                    raise
                else:
                    # Exponential backoff for inner retries: 2^attempt seconds
                    wait_time = 2 ** inner_attempt
                    logger.info(f"Retrying outer batch {outer_batch_num} (attempt {inner_attempt + 2}/{max_inner_retries}) after {wait_time}s")
                    await asyncio.sleep(wait_time)
        
        # Upload progress to S3 after each successful outer batch
        if output_file.exists():
            upload_success = await s3_cache.upload_cache_to_s3_async(topic_name, source_name, output_file)
            if upload_success:
                logger.debug(f"Uploaded outer batch {outer_batch_num} progress to S3")
        
        # Sleep between outer batches (except for the last one)
        if i + outer_batch_size < total_items:
            await asyncio.sleep(0.5)


async def _periodic_s3_upload(
    s3_cache: S3CacheManager,
    topic_name: str,
    source_name: str,
    output_file: Path,
    upload_interval: int = 30
) -> None:
    """Periodically upload progress to S3 while processing is running.
    
    Args:
        s3_cache: S3 cache manager instance
        topic_name: Name of the topic
        source_name: Name of the data source
        output_file: Path to output file
        upload_interval: Upload interval in seconds
    """
    try:
        while True:
            await asyncio.sleep(upload_interval)
            
            if output_file.exists():
                success = await s3_cache.upload_cache_to_s3_async(topic_name, source_name, output_file)
                if success:
                    logger.debug(f"Periodic S3 upload completed for {source_name} {topic_name}")
                    
    except asyncio.CancelledError:
        # Task was cancelled, which is expected when processing completes
        logger.debug("Periodic S3 upload task cancelled") 