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

logger = logging.getLogger(__name__)

def get_cb_data(topic_name: str, cache_dir: Path, config: Dict[str, Any], cb_getter=None) -> Dict[str, Any]:
    """
    Get Crunchbase company IDs for a topic with caching.
    
    Args:
        topic_name: Name of the topic (e.g., "hydrogen_energy")
        cache_dir: Directory for caching results
        config: Topic configuration dictionary
        cb_getter: Pre-initialized CrunchbaseGetter (optional)
        
    Returns:
        {
            'ids': List[str],           # Relevant company IDs
            'topic_name': str,          # Topic name
            'config': Dict,             # Original config
            'companies_count': int,     # Total companies found
            'relevant_count': int       # Companies after relevance check
        }
    """
    cache_file = cache_dir / f"{topic_name}_cb_data.json"
    
    # Check cache first
    if _is_cache_valid(cache_file, max_age_hours=24):
        logger.info(f"Loading cached Crunchbase data for {topic_name}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Processing fresh Crunchbase data for {topic_name}")
    
    # Fresh data processing
    if cb_getter is None:
        CB = crunchbase.CrunchbaseGetter()
    else:
        CB = cb_getter
    
    # Get companies by category
    category_name = config['search_recipe']['category_name']
    companies_df = CB.get_companies_in_nesta_categories("topic_labels", [category_name])
    
    logger.info(f"Found {len(companies_df)} companies for category '{category_name}'")
    
    # Run LLM relevance check
    relevant_ids = _run_relevance_check(CB, companies_df, config, cache_dir, topic_name)
    
    # Prepare result
    result = {
        'ids': relevant_ids,
        'topic_name': topic_name,
        'config': config,
        'companies_count': len(companies_df),
        'relevant_count': len(relevant_ids)
    }
    
    # Cache result
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Cached {len(relevant_ids)} relevant companies for {topic_name}")
    return result

def _is_cache_valid(cache_file: Path, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is recent enough"""
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return age_hours < max_age_hours
    return False

async def _run_relevance_check_async(CB, companies_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Run LLM relevance check"""
    # Get organisation text for relevance checking
    selected_texts_df = CB.get_organisation_text(companies_df)
    
    # Check for existing relevance cache
    relevance_cache_file = cache_dir / f"{topic_name}_relevance_check.jsonl"
    
    # Get all current company IDs that need to be checked
    current_company_ids = set(companies_df['id'].tolist())
    
    # Load existing results if available
    existing_results = {}
    if relevance_cache_file.exists():
        try:
            existing_df = pd.read_json(relevance_cache_file, lines=True)
            existing_results = dict(zip(existing_df['id'], existing_df['is_relevant']))
            logger.info(f"Loaded {len(existing_results)} existing relevance check results for {topic_name}")
        except Exception as e:
            logger.warning(f"Could not read existing relevance check results: {e}")
            existing_results = {}
    
    # Identify entities that need LLM checking
    already_checked_ids = set(existing_results.keys())
    new_ids_to_check = current_company_ids - already_checked_ids
    
    logger.info(f"Relevance check status for {topic_name}:")
    logger.info(f"  Current entities: {len(current_company_ids)}")
    logger.info(f"  Already checked: {len(already_checked_ids)}")
    logger.info(f"  New entities to check: {len(new_ids_to_check)}")
    
    # Run LLM check only on new entities (if any)
    if new_ids_to_check:
        logger.info(f"Running LLM relevance check on {len(new_ids_to_check)} new entities for {topic_name}")
        
        # Get text data for only the new entities
        new_texts_df = selected_texts_df[selected_texts_df['id'].isin(new_ids_to_check)]
        check_data = dict(zip(new_texts_df['id'], new_texts_df['text']))
        
        # Generate system message
        system_message = batch_check.generate_relevance_check_system_message(config)
        
        # Add the detailed custom instructions
        system_message += """ 
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
            await _robust_llm_processing(processor, check_data, relevance_cache_file)
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
    
    # Merge and filter for relevant companies
    relevant_checked_df = (
        companies_df
        .merge(relevant_check_df[['id', 'is_relevant']], left_on='id', right_on='id', how='left')
        .merge(selected_texts_df[['id', 'text']], left_on='id', right_on='id', how='left')
    )
    
    companies_without_results = relevant_checked_df['is_relevant'].isna().sum()
    if companies_without_results > 0:
        logger.warning(f"Found {companies_without_results} companies without relevance check results - they will be excluded")
    
    matching_ids = relevant_checked_df.query("is_relevant == 'yes'").id.tolist()
    logger.info(f"Relevance check complete: {len(matching_ids)} relevant companies out of {len(companies_df)} total")
    return matching_ids

def _run_relevance_check(CB, companies_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Synchronous wrapper for async relevance check"""
    return asyncio.run(_run_relevance_check_async(CB, companies_df, config, cache_dir, topic_name))

def validate_cb_config(config: Dict[str, Any]) -> None:
    """Basic validation - fail fast on missing required fields"""
    if 'search_recipe' not in config:
        raise ValueError("Missing required CB config field: search_recipe")
    
    if 'category_name' not in config['search_recipe']:
        raise ValueError("Missing required CB config field: search_recipe.category_name")

async def _robust_llm_processing(processor, check_data: Dict, output_file: Path) -> None:
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