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

logger = logging.getLogger(__name__)

def get_gtr_data(topic_name: str, cache_dir: Path, config: Dict[str, Any], gtr_getter=None) -> Dict[str, Any]:
    """
    Get GTR project IDs for a topic with caching.
    
    Mirrors notebook get_ids_from_config() pattern exactly.
    
    Args:
        topic_name: Name of the topic (e.g., "hydrogen_energy")
        cache_dir: Directory for caching results
        config: Topic configuration dictionary
        gtr_getter: Pre-initialized GtrGetter (optional)
        
    Returns:
        {
            'ids': List[str],           # Relevant project IDs
            'topic_name': str,          # Topic name
            'config': Dict,             # Original config
            'projects_count': int,      # Total projects found
            'relevant_count': int       # Projects after relevance check
        }
    """
    cache_file = cache_dir / f"{topic_name}_gtr_data.json"
    
    # Check cache first
    if _is_cache_valid(cache_file, max_age_hours=24):
        logger.info(f"Loading cached GTR data for {topic_name}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Processing fresh GTR data for {topic_name}")
    
    if gtr_getter is None:
        GTR = gtr.GtrGetter()
    else:
        GTR = gtr_getter
    
    enrichment_df = _get_or_create_enrichment_data(GTR, cache_dir)
    logger.info(f"Using enrichment data with {len(enrichment_df)} entries")
    
    # Get projects by category
    category_name = config['search_recipe']['category_name']
    projects_df = _get_projects_from_config(GTR, enrichment_df, config)
    
    logger.info(f"Found {len(projects_df)} projects for category '{category_name}' after pre-filtering")
    
    # Run LLM relevance check only on pre-filtered projects
    relevant_ids = _run_relevance_check(GTR, projects_df, config, cache_dir, topic_name)
    
    # Prepare result
    result = {
        'ids': relevant_ids,
        'topic_name': topic_name,
        'config': config,
        'projects_count': len(projects_df),
        'relevant_count': len(relevant_ids)
    }
    
    # Cache result
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)

    
    logger.info(f"Cached {len(relevant_ids)} relevant projects for {topic_name}")
    return result

def _is_cache_valid(cache_file: Path, max_age_hours: int = 24) -> bool:
    """Check if cache file exists and is recent enough"""
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return age_hours < max_age_hours
    return False

def _get_or_create_enrichment_data(GTR, cache_dir: Path) -> pd.DataFrame:
    """Get or create enrichment data"""
    enrichment_cache_file = cache_dir / "gtr_labelled_projects.csv"
    
    # Check if we have cached enrichment data
    if enrichment_cache_file.exists():
        logger.info("Loading cached GTR enrichment data")
        try:
            enrichment_df = (
                pd.read_csv(enrichment_cache_file)
                .assign(topic_labels=lambda df: df.topic_labels.apply(lambda x: x.split(",") if isinstance(x, str) else []))
                .assign(mission_labels=lambda df: df.mission_labels.apply(lambda x: x.split(",") if isinstance(x, str) else []))
            )
            return enrichment_df
        except Exception as e:
            logger.warning(f"Could not load cached enrichment data: {e}. Will regenerate.")
    
    logger.info("Generating GTR enrichment data using discovery_utils.utils.keywords (this may take 5-6 minutes)")
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
    logger.info(f"Generated and cached GTR enrichment data with {len(enrichment_df)} entries")
    
    return enrichment_df

def _get_projects_in_nesta_categories(GTR, enrichment_df: pd.DataFrame, category_type: str, categories: List[str]) -> pd.DataFrame:
    """Get all projects belonging to the provided categories"""
    matching_ids = (
        enrichment_df
        .explode(category_type)
        .query(f"{category_type} in @categories")
        .id.to_list()
    )
    return GTR.projects_enriched.query("id in @matching_ids").drop_duplicates(subset="id")

def _get_projects_from_config(GTR, enrichment_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Get projects from config"""
    category_name = config["search_recipe"]["category_name"]
    return _get_projects_in_nesta_categories(GTR, enrichment_df, "topic_labels", [category_name])

def _run_relevance_check(GTR, projects_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Synchronous wrapper for async relevance check"""
    return asyncio.run(_run_relevance_check_async(GTR, projects_df, config, cache_dir, topic_name))

async def _run_relevance_check_async(GTR, projects_df: pd.DataFrame, config: Dict, cache_dir: Path, topic_name: str) -> List[str]:
    """Run LLM relevance check"""
    # Check for existing relevance cache
    relevance_cache_file = cache_dir / f"{topic_name}_gtr_relevance_check.jsonl"
    
    # Get all current project IDs that need to be checked
    current_project_ids = set(projects_df['id'].tolist())
    
    # Load existing results if available
    existing_results = {}
    if relevance_cache_file.exists():
        try:
            existing_df = pd.read_json(relevance_cache_file, lines=True)
            existing_results = dict(zip(existing_df['id'], existing_df['is_relevant']))
            logger.info(f"Loaded {len(existing_results)} existing GTR relevance check results for {topic_name}")
        except Exception as e:
            logger.warning(f"Could not read existing GTR relevance check results: {e}")
            existing_results = {}
    
    # Identify entities that need LLM checking
    already_checked_ids = set(existing_results.keys())
    new_ids_to_check = current_project_ids - already_checked_ids
    
    logger.info(f"GTR relevance check status for {topic_name}:")
    logger.info(f"  Current entities: {len(current_project_ids)}")
    logger.info(f"  Already checked: {len(already_checked_ids)}")
    logger.info(f"  New entities to check: {len(new_ids_to_check)}")
    
    # Run LLM check only on new entities (if any)
    if new_ids_to_check:
        logger.info(f"Running GTR LLM relevance check on {len(new_ids_to_check)} new entities for {topic_name}")
        
        # Get projects text for relevance checking - only for new entities
        selected_texts_df = GTR.get_projects_text().query("id in @new_ids_to_check")
        check_data = dict(zip(selected_texts_df['id'], selected_texts_df['text']))
        
        # Generate system message
        system_message = batch_check.generate_relevance_check_system_message(config)
        
        system_message += """ 
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
        
        fields = [
            {"name": "explanation", "type": "str", "description": "A short, 1-sentence explanation of the answer (max 25 words)."},
            {"name": "is_relevant", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},    
        ]    

        processor = batch_check.LLMProcessor(
            model_name="gpt-4o-mini",
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
            logger.error(f"All GTR LLM processing attempts failed: {e}")
            raise
        
        relevant_check_df = pd.read_json(relevance_cache_file, lines=True)
        logger.info(f"GTR LLM relevance check completed for {topic_name}: {len(new_ids_to_check)} new entities processed")
    else:
        logger.info(f"All GTR entities already checked for {topic_name}, skipping LLM calls")
        relevant_check_df = pd.DataFrame([
            {'id': entity_id, 'is_relevant': is_relevant}
            for entity_id, is_relevant in existing_results.items()
        ])
    
    # Merge and filter for relevant projects
    relevant_checked_df = (
        projects_df
        .merge(relevant_check_df[['id', 'is_relevant']], left_on='id', right_on='id', how='left')
    )
    
    projects_without_results = relevant_checked_df['is_relevant'].isna().sum()
    if projects_without_results > 0:
        logger.warning(f"Found {projects_without_results} GTR projects without relevance check results - they will be excluded")
    
    matching_ids = relevant_checked_df.query("is_relevant == 'yes'").id.tolist()
    logger.info(f"GTR relevance check complete: {len(matching_ids)} relevant projects out of {len(projects_df)} total")
    return matching_ids

def validate_gtr_config(config: Dict[str, Any]) -> None:
    """Basic validation - fail fast on missing required fields"""
    if 'search_recipe' not in config:
        raise ValueError("Missing required GTR config field: search_recipe")
    
    if 'category_name' not in config['search_recipe']:
        raise ValueError("Missing required GTR config field: search_recipe.category_name")

async def _llm_processing(processor, check_data: Dict, output_file: Path) -> None:
    """
    Robust LLM processing with timeout detection and automatic recovery.
    Handles hanging/stuck LLM processors by using timeouts and batch reduction.
    """
    # Check how many items are already processed (resume capability)
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