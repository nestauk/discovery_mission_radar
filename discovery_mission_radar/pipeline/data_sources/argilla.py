"""
Argilla integration for manual review of LLM relevance checks.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import os

from discovery_mission_radar.pipeline.data_sources.utils import S3CacheManager
from discovery_utils.utils.s3 import _download_obj, upload_obj

logger = logging.getLogger(__name__)


def get_argilla_client():
    """Get configured Argilla client."""
    import argilla as rg
    
    api_url = os.environ.get('ARGILLA_API_URL')
    api_key = os.environ.get('ARGILLA_API_KEY')
    
    if not api_url or not api_key:
        raise ValueError("Argilla API URL or API key not found in environment variables")
    
    return rg.Argilla(api_url=api_url, api_key=api_key)


def _has_valid_description(text) -> bool:
    """Check if text has valid description for review."""
    if pd.isna(text) or text is None:
        return False
    if isinstance(text, str):
        return len(text.strip()) > 10
    return False


def _clean_text_field(value) -> str:
    """Clean text field, handling NaN and empty values."""
    if pd.isna(value) or value is None:
        return "No description available"
    if isinstance(value, str):
        return value.strip() if value.strip() else "No description available"
    return str(value)


def _get_title_field(row: pd.Series, source_name: str) -> str:
    """Get appropriate title field based on source."""
    field_mapping = {
        'crunchbase': 'name',
        'gtr': 'title', 
        'hansard': 'major_heading'
    }
    field = field_mapping.get(source_name, 'name')
    return row.get(field, 'No title available')


def _registry_s3_key(mission: str, source_name: str) -> str:
    return f"data/mission_radar/argilla/{mission}/sampling_registry/{source_name}.json"


def _labels_s3_key(mission: str, source_name: str) -> str:
    return f"data/mission_radar/argilla/{mission}/labels/{source_name}.parquet"


async def get_entities_to_sample(
    topic: str, 
    quarter: str, 
    mission: str, 
    entities_df: pd.DataFrame, 
    sample_size: int, 
    cache_dir: Path, 
    s3_cache: S3CacheManager,
    source_name: str
) -> List[str]:
    """Deterministic sampling excluding previously sampled entities."""
    # Local registry cache
    registry_file = cache_dir / f"sampling_registry_{source_name}.json"

    registry: Dict[str, Dict] = {}

    if s3_cache.enabled:
        try:
            key = _registry_s3_key(mission, source_name)
            registry = _download_obj(s3_cache.s3, s3_cache.bucket_name, key, download_as='dict')
            logger.info(f"Downloaded Argilla sampling registry from S3: s3://{s3_cache.bucket_name}/{key}")
            # Persist local copy
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        except Exception as e:
            logger.info(f"No remote sampling registry found yet for {source_name}: {e}")
            # Fallback to local file if present
            if registry_file.exists():
                try:
                    registry = json.loads(registry_file.read_text(encoding="utf-8"))
                except Exception:
                    registry = {}
    else:
        if registry_file.exists():
            try:
                registry = json.loads(registry_file.read_text(encoding="utf-8"))
            except Exception:
                registry = {}
        else:
            logger.info(f"No existing Argilla sampling registry found for {mission} {source_name}")
    
    previously_sampled: Set[str] = set()
    key_prefix = f"{topic}_{quarter}"
    for k, sample_data in registry.items():
        if k == key_prefix:
            previously_sampled.update(sample_data.get('entity_ids', []))
    
    logger.info(f"Argilla sampling status for {source_name} {topic} in {quarter}:")
    logger.info(f"  Available entities: {len(entities_df)}")
    logger.info(f"  Previously sampled: {len(previously_sampled)}")
    
    available = entities_df[~entities_df['id'].isin(previously_sampled)].copy()
    logger.info(f"  New entities available: {len(available)}")
    
    if len(available) == 0:
        logger.info(f"No new entities available for sampling in {source_name} {topic}")
        return []
    
    if 'text' in available.columns:
        available = available[available['text'].apply(_has_valid_description)]
    else:
        logger.warning(f"No 'text' column found in entities_df for {source_name} {topic}")
    logger.info(f"  Entities with valid descriptions: {len(available)}")
    
    if len(available) == 0:
        logger.info(f"No entities with valid descriptions available for sampling in {source_name} {topic}")
        return []
    
    available['hash'] = available['id'].apply(lambda x: hash(f"{x}_{quarter}_{topic}_{source_name}"))

    # Determine remaining capacity for this topic/quarter to avoid oversampling across runs
    existing_ids = []
    if key_prefix in registry:
        existing_ids = registry[key_prefix].get('entity_ids', [])
    existing_count = len(existing_ids)
    remaining_to_sample = max(0, sample_size - existing_count)
    if remaining_to_sample == 0:
        logger.info(f"Sampling cap reached for {source_name} {topic} {quarter} (existing={existing_count}, cap={sample_size}); skipping selection")
        return []

    # Balanced sampling by LLM decision where available
    sampled = None
    if 'is_relevant' in available.columns and available['is_relevant'].notna().any():
        try:
            positive = available[available['is_relevant'] == 'yes'].copy()
            negative = available[available['is_relevant'] == 'no'].copy()

            target_pos = remaining_to_sample // 2
            target_neg = remaining_to_sample - target_pos

            pos_take = min(target_pos, len(positive))
            neg_take = min(target_neg, len(negative))

            # If one side is short, top up from the other side deterministically
            remaining = remaining_to_sample - (pos_take + neg_take)
            if remaining > 0:
                if pos_take < target_pos and len(negative) - neg_take > 0:
                    extra_neg = min(remaining, len(negative) - neg_take)
                    neg_take += extra_neg
                    remaining -= extra_neg
                if remaining > 0 and len(positive) - pos_take > 0:
                    extra_pos = min(remaining, len(positive) - pos_take)
                    pos_take += extra_pos

            pos_sample = positive.nsmallest(pos_take, 'hash') if pos_take > 0 else positive.iloc[0:0]
            neg_sample = negative.nsmallest(neg_take, 'hash') if neg_take > 0 else negative.iloc[0:0]

            sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)
            # Ensure deterministic overall ordering
            sampled = sampled.nsmallest(len(sampled), 'hash')
        except Exception:
            # Safety fallback to deterministic sampling
            sampled = available.nsmallest(min(remaining_to_sample, len(available)), 'hash')
    else:
        sampled = available.nsmallest(min(remaining_to_sample, len(available)), 'hash')
    
    new_ids = sampled['id'].tolist()
    current_time = datetime.now().isoformat()
    
    # Update registry
    if key_prefix not in registry:
        registry[key_prefix] = {
            'entity_ids': [],
            'exported_at': current_time,
            'counts': {'total': 0, 'reviewed': 0}
        }
    registry[key_prefix]['entity_ids'] = list(sorted(set(registry[key_prefix]['entity_ids']).union(new_ids)))
    registry[key_prefix]['last_updated'] = current_time
    registry[key_prefix]['counts']['total'] = len(registry[key_prefix]['entity_ids'])
    
    try:
        registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        if s3_cache.enabled:
            key = _registry_s3_key(mission, source_name)
            upload_obj(registry, s3_cache.bucket_name, key, kwargs_writing={'indent': 2})
            logger.info(f"Uploaded Argilla sampling registry to s3://{s3_cache.bucket_name}/{key}")
    except Exception as e:
        logger.error(f"Failed to save Argilla sampling registry: {e}")
        return []
    
    logger.info(f"Sampled {len(new_ids)} entities for Argilla review in {source_name} {topic}")
    return new_ids


def reset_sampling_registry(mission: str, source_name: str, cache_dir: Path, delete_remote: bool = True) -> bool:
    """Reset the Argilla sampling registry for a source.

    Removes the local registry file and, if configured, deletes the remote S3
    registry object. This fully resets sampling history for the specified
    `source_name` and `mission`.

    Args:
        mission: Mission identifier (e.g., 'AHL', 'ASF').
        source_name: Data source name, e.g., 'crunchbase', 'gtr', 'hansard'.
        cache_dir: Base cache directory used by the pipeline for this mission.
        delete_remote: Whether to delete the S3 registry object if S3 is enabled.

    Returns:
        True if the reset completed without critical errors, False otherwise.
    """
    ok = True
    try:
        registry_file = cache_dir / f"sampling_registry_{source_name}.json"
        if registry_file.exists():
            try:
                registry_file.unlink()
                logger.info(f"Deleted local Argilla sampling registry: {registry_file}")
            except Exception as e:
                logger.warning(f"Failed to delete local registry {registry_file}: {e}")
                ok = False

        if delete_remote:
            s3_cache = S3CacheManager(mission)
            if s3_cache.enabled:
                try:
                    key = _registry_s3_key(mission, source_name)
                    s3_cache.s3.delete_object(Bucket=s3_cache.bucket_name, Key=key)
                    logger.info(f"Deleted remote Argilla sampling registry: s3://{s3_cache.bucket_name}/{key}")
                except Exception as e:
                    logger.warning(f"Failed to delete remote registry for {source_name}: {e}")
                    ok = False
    except Exception as e:
        logger.error(f"Unexpected error resetting sampling registry for {source_name}: {e}")
        ok = False
    return ok


def delete_argilla_datasets_for_source(mission: str, source_name: str, workspace: str = "admin") -> int:
    """Delete all Argilla datasets for a mission/source across topics.

    Args:
        mission: Mission identifier (e.g., 'AHL', 'ASF').
        source_name: Data source name, e.g., 'crunchbase', 'gtr', 'hansard'.
        workspace: Argilla workspace name.

    Returns:
        Number of datasets successfully deleted.
    """
    try:
        client = get_argilla_client()
    except Exception as e:
        logger.error(f"Cannot initialise Argilla client: {e}")
        return 0

    deleted = 0
    try:
        prefix = f"{mission}_{source_name}_"
        try:
            datasets = [ds for ds in client.datasets.list() if hasattr(ds, 'workspace') and ds.workspace and getattr(ds.workspace, 'name', None) == workspace]
        except Exception as e:
            logger.error(f"Failed listing datasets in workspace '{workspace}': {e}")
            return 0

        for ds in datasets:
            try:
                if getattr(ds, 'name', '').startswith(prefix):
                    # Try dataset-bound delete first, then client-based
                    if hasattr(ds, 'delete'):
                        ds.delete()
                    else:
                        try:
                            client.datasets.delete(ds)
                        except Exception:
                            # Fallback: retrieve by name and delete
                            try:
                                resolved = client.datasets(ds.name, workspace=workspace)
                                if hasattr(resolved, 'delete'):
                                    resolved.delete()
                            except Exception:
                                pass
                    deleted += 1
                    logger.info(f"Deleted Argilla dataset: {ds.name} (workspace={workspace})")
            except Exception as e_del:
                logger.warning(f"Failed to delete dataset {getattr(ds, 'name', 'unknown')}: {e_del}")
                continue
    except Exception as e:
        logger.error(f"Unexpected error deleting Argilla datasets for {mission}/{source_name}: {e}")
        return deleted

    return deleted


def export_to_argilla(
    entities: List[str], 
    entities_df: pd.DataFrame, 
    topic: str, 
    quarter: str, 
    mission: str, 
    config: Dict,
    source_name: str
) -> bool:
    """Export entities to Argilla for manual review."""
    try:
        import argilla as rg
        
        client = get_argilla_client()
        workspace = config.get('workspace', 'argilla')
        
        export_df = entities_df[entities_df['id'].isin(entities)].copy()
        
        if len(export_df) == 0:
            logger.warning(f"No entities found for export to Argilla in {topic}")
            return False
        
        if 'text' in export_df.columns:
            export_df['clean_text'] = export_df['text'].apply(_clean_text_field)
        else:
            export_df['clean_text'] = "No description available"
            logger.warning(f"No 'text' column found in export_df for {source_name} {topic}")
        
        export_df = export_df[export_df['clean_text'] != "No description available"]
        export_df = export_df[export_df['clean_text'].str.len() > 10]
        
        if len(export_df) == 0:
            logger.warning(f"No entities with valid descriptions found for export to Argilla in {topic}")
            return False
        
        logger.info(f"Filtered to {len(export_df)} entities with valid descriptions for Argilla export")
        
        dataset_name = f"{mission}_{source_name}_{topic}"
        
        try:
            dataset = None
            try:
                dataset = client.datasets(dataset_name, workspace=workspace)
                if not (dataset and hasattr(dataset, 'settings') and dataset.settings):
                    dataset = None
                    raise Exception("Dataset exists but has no settings")
            except Exception:
                logger.info(f"Creating new Argilla dataset: {dataset_name}")
                
                settings = rg.Settings(
                    fields=[
                        rg.TextField(name="entity_title", title="Entity Name/Title"),
                        rg.TextField(name="text", title="Entity Description"),
                        rg.TextField(name="llm_explanation", title="LLM Reasoning")
                    ],
                    questions=[
                        rg.LabelQuestion(
                            name="relevance", 
                            title="Is this entity relevant to the topic?", 
                            description="Select 'yes' or 'no'",
                            labels=["yes", "no"]
                        ),
                        rg.TextQuestion(
                            name="notes",
                            title="Reviewer Notes (Optional)",
                            description="Provide any additional reasoning, context, or comments about your decision",
                            required=False
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="entity_id", title="Entity ID"),
                        rg.TermsMetadataProperty(name="source", title="Data Source"),
                        rg.TermsMetadataProperty(name="topic", title="Topic"),
                        rg.TermsMetadataProperty(name="quarter", title="Quarter"),
                        rg.TermsMetadataProperty(name="mission", title="Mission"),
                        rg.TermsMetadataProperty(name="llm_decision", title="LLM Decision")
                    ]
                )
                
                dataset = rg.Dataset(
                    name=dataset_name,
                    workspace=workspace,
                    settings=settings
                )
                
                dataset = client.datasets.add(dataset)
            
            if not dataset or not hasattr(dataset, 'settings') or not dataset.settings:
                logger.error(f"Dataset {dataset_name} still has no settings after creation/retrieval")
                return False
            
            records = []
            for _, row in export_df.iterrows():
                suggestions = []
                try:
                    suggestion = rg.Suggestion(
                        question_name="relevance",
                        value=str(row.get('is_relevant', 'no')),
                        agent="llm"
                    )
                    suggestions.append(suggestion)
                except Exception:
                    pass
                
                title_field = _get_title_field(row, source_name)
                
                record = rg.Record(
                    fields={
                        "entity_title": _clean_text_field(title_field),
                        "text": row['clean_text'],
                        "llm_explanation": _clean_text_field(row.get('explanation', 'No explanation provided'))
                    },
                    metadata={
                        'entity_id': str(row['id']),
                        'source': source_name,
                        'topic': topic,
                        'quarter': quarter,
                        'mission': mission,
                        'llm_decision': str(row.get('is_relevant', 'no'))
                    },
                    suggestions=suggestions
                )
                
                records.append(record)
            
            dataset.records.log(records)
            
            logger.info(f"Exported {len(export_df)} entities to Argilla dataset: {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update Argilla dataset {dataset_name}: {e}")
            return False
            
    except ImportError:
        logger.error("Argilla package not installed. Install with: pip install argilla")
        return False
    except Exception as e:
        logger.error(f"Failed to export to Argilla: {e}")
        return False


def import_from_argilla(
    topic: str, 
    quarter: str, 
    mission: str, 
    config: Dict,
    source_name: str
) -> List[Dict]:
    """Fetch completed manual reviews from Argilla and persist label store per source."""
    try:
        client = get_argilla_client()
        workspace = config.get('workspace', 'argilla')
        
        dataset_name = f"{mission}_{source_name}_{topic}"
        
        try:
            try:
                dataset = client.datasets(dataset_name, workspace=workspace)
            except Exception:
                return []
            
            manual_reviews = []
            
            for record in dataset.records:
                if (hasattr(record, 'metadata') and record.metadata and 
                    record.metadata.get('quarter') == quarter):
                    
                    if hasattr(record, 'responses') and record.responses:
                        for response in record.responses:
                            if hasattr(response, 'status') and response.status == "submitted":
                                relevance_value = "no"
                                reviewer_notes = ""
                                
                                if hasattr(response, 'values') and response.values:
                                    relevance_value = response.values.get('relevance', 'no')
                                    reviewer_notes = response.values.get('notes', '')
                                
                                explanation_parts = [f"Manual review: {relevance_value}"]
                                if reviewer_notes and reviewer_notes.strip():
                                    explanation_parts.append(f"Notes: {reviewer_notes.strip()}")
                                explanation = " | ".join(explanation_parts)
                                
                                manual_reviews.append({
                                    'entity_id': record.metadata.get('entity_id') if record.metadata else None,
                                    'manual_decision': relevance_value,
                                    'explanation': explanation,
                                    'llm_explanation': record.fields.get('llm_explanation') if hasattr(record, 'fields') and record.fields else '',
                                    'reviewer_notes': reviewer_notes,
                                    'reviewer_id': response.user_id if hasattr(response, 'user_id') else 'unknown',
                                    'review_timestamp': response.inserted_at.isoformat() if hasattr(response, 'inserted_at') and response.inserted_at else datetime.now().isoformat(),
                                    'review_quarter': quarter
                                })
            
            if len(manual_reviews) > 0:
                logger.info(f"Imported {len(manual_reviews)} completed reviews from Argilla for {topic} ({quarter})")
            
            try:
                s3_cache = S3CacheManager(mission)
                if s3_cache.enabled and manual_reviews:
                    import pandas as pd
                    
                    labels_key = _labels_s3_key(mission, source_name)
                    
                    new_df = pd.DataFrame([
                        {
                            'id': r['entity_id'],
                            'source': source_name,
                            'topic': topic,
                            'mission': mission,
                            'quarter': r['review_quarter'],
                            'is_relevant': r['manual_decision'],
                            'llm_explanation': r.get('llm_explanation', ''),
                            'reviewer_notes': r.get('reviewer_notes', ''),
                            'reviewer_id': r.get('reviewer_id', ''),
                            'review_timestamp': pd.to_datetime(r['review_timestamp']),
                            'llm_decision': r['manual_decision'],
                            'schema_version': '1'
                        }
                        for r in manual_reviews if r.get('entity_id')
                    ])
                    
                    # Download existing labels if present
                    existing_df = None
                    try:
                        existing_df = _download_obj(s3_cache.s3, s3_cache.bucket_name, labels_key, download_as='dataframe')
                    except Exception:
                        existing_df = None
                    
                    if existing_df is not None and not existing_df.empty:
                        combined = pd.concat([existing_df, new_df], ignore_index=True)
                        combined.sort_values('review_timestamp', inplace=True)
                        combined = combined.drop_duplicates(subset=['id', 'topic', 'quarter'], keep='last')
                    else:
                        combined = new_df
                    
                    upload_obj(combined, s3_cache.bucket_name, labels_key)
                    logger.info(f"Updated label store: s3://{s3_cache.bucket_name}/{labels_key}")
            except Exception as e:
                logger.warning(f"Failed to update label store parquet: {e}")
            
            return manual_reviews
            
        except Exception as e:
            logger.warning(f"Could not load Argilla dataset {dataset_name}: {e}")
            return []
            
    except ImportError:
        logger.error("Argilla package not installed. Install with: pip install argilla")
        return []
    except Exception as e:
        logger.error(f"Failed to import from Argilla: {e}")
        return []


def apply_manual_overrides(cache_file: Path, manual_reviews: List[Dict]) -> bool:
    """Update existing JSONL cache with manual decisions."""
    if not manual_reviews:
        return True
    
    try:
        cache_df = pd.read_json(cache_file, lines=True)
        
        override_count = 0
        for review in manual_reviews:
            entity_id = review['entity_id']
            mask = cache_df['id'] == entity_id
            
            if mask.any():
                cache_df.loc[mask, 'is_relevant'] = review['manual_decision']
                cache_df.loc[mask, 'manual_decision'] = review['manual_decision']
                cache_df.loc[mask, 'review_quarter'] = review['review_quarter']
                
                if 'explanation' in cache_df.columns:
                    cache_df.loc[mask, 'explanation'] = review['explanation']
                
                if 'reviewer_notes' in review and review['reviewer_notes']:
                    if 'reviewer_notes' not in cache_df.columns:
                        cache_df['reviewer_notes'] = ''
                    cache_df.loc[mask, 'reviewer_notes'] = review['reviewer_notes']
                
                override_count += 1
        
        cache_df.to_json(cache_file, orient='records', lines=True)
        
        logger.info(f"Applied {override_count} manual overrides to cache file: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply manual overrides: {e}")
        return False 


def _is_owner(client) -> bool:
    try:
        me = client.me
        return getattr(me, 'role', None) == 'owner'
    except Exception:
        return False


def ensure_users_from_s3(mission: str, key: str = "credentials/argilla_users.json") -> Dict[str, int]:
    """Create missing Argilla users from S3 JSON list at the given key.
    The S3 bucket is taken from the pipeline S3 configuration (S3CacheManager).
    """
    import argilla as rg
    
    s3_cache = S3CacheManager(mission)
    if not s3_cache.enabled:
        logger.info("S3 is not configured (S3_BUCKET missing). Skipping Argilla user provisioning.")
        return {"created": 0, "existing": 0, "errors": 0}
    
    try:
        users_data = _download_obj(s3_cache.s3, s3_cache.bucket_name, key, download_as='list')
    except Exception as e:
        logger.info(f"No Argilla users credentials found in S3 at '{key}': {e}")
        return {"created": 0, "existing": 0, "errors": 0}
    
    if not isinstance(users_data, list):
        logger.error("Credentials JSON must be a list of user objects")
        return {"created": 0, "existing": 0, "errors": 1}
    
    client = get_argilla_client()
    if not _is_owner(client):
        logger.warning("Current Argilla user is not 'owner'. Skipping user provisioning.")
        return {"created": 0, "existing": 0, "errors": 0}
    
    created = 0
    existing = 0
    errors = 0
    
    for entry in users_data:
        try:
            username = entry["username"]
            password = entry.get("password")
            if not password:
                raise ValueError(f"Password missing for user {username}")
            first_name = entry.get("first_name", username)
            last_name = entry.get("last_name", "")
            role = entry.get("role", "annotator")
            workspace_names = entry.get("workspaces", [])
            
            user_obj = rg.User(
                username=username,
                first_name=first_name,
                last_name=last_name,
                role=role,
                password=password,
                client=client,
            )
            try:
                user_obj.create()
                created += 1
                logger.info(f"Created Argilla user: {username}")
            except Exception as ce:
                msg = str(ce).lower()
                if "exist" in msg or "already" in msg or "409" in msg:
                    existing += 1
                    logger.info(f"User already exists: {username}")
                    try:
                        user_obj = client.users(username)
                    except Exception:
                        pass
                else:
                    errors += 1
                    logger.error(f"Failed creating user {username}: {ce}")
                    continue
            
            for ws_name in workspace_names:
                try:
                    workspace = client.workspaces(ws_name)
                    if workspace and hasattr(user_obj, 'add_to_workspace'):
                        user_obj.add_to_workspace(workspace)
                        logger.info(f"Added {username} to workspace: {ws_name}")
                except Exception as e_ws:
                    logger.warning(f"Could not add {username} to workspace {ws_name}: {e_ws}")
        except Exception as e:
            errors += 1
            logger.error(f"Error provisioning user from entry {entry}: {e}")
    
    summary = {"created": created, "existing": existing, "errors": errors}
    logger.info(f"Argilla user provisioning summary: {summary}")
    return summary 