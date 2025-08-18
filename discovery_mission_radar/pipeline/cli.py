#!/usr/bin/env python3
"""
Mission Radar Pipeline CLI

Orchestrates data sources + analysis functions
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

import argilla as rg
import click
import pandas as pd

from discovery_mission_radar.pipeline.runner import MissionRadarRunner
from discovery_mission_radar.pipeline.config_manager import get_pipeline_config
from discovery_mission_radar.pipeline.data_sources.utils import S3CacheManager

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_single_topic(runner: MissionRadarRunner, topic_name: str, mission: str, output_dir: Path = None):
    """Run analysis for a single topic"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running single topic analysis for: {topic_name} (Mission: {mission})")
    
    try:
        results = runner.run_topic_end_to_end(topic_name)
        logger.info(f"Successfully completed analysis for {topic_name}")
        
        print(f"\n=== Analysis Complete for {topic_name} (Mission: {mission}) ===")
        
        if results.get('native_category'):
            print(f"Processed as Crunchbase native category")
        
        for source, data in results.items():
            if source not in ['topic_name', 'mission', 'native_category'] and isinstance(data, dict):
                stats = data.get('stats', {})
                print(f"{source.upper()}: {stats}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process topic {topic_name}: {e}")
        raise

def run_batch_analysis(runner: MissionRadarRunner, topic_list: List[str], mission: str, output_dir: Path = None):
    """Run batch analysis for multiple topics"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running batch analysis for topics: {topic_list} (Mission: {mission})")
    
    try:
        results = runner.run_batch_analysis(topic_list)
        logger.info("Successfully completed batch analysis")
        
        print(f"\n=== Batch Analysis Complete (Mission: {mission}) ===")
        print(f"Processed {len(topic_list)} topics")
        
        if 'consolidated_files' in results:
            print(f"\nConsolidated CSV files generated:")
            for file_name, file_path in results['consolidated_files'].items():
                print(f"  - {file_name}: {file_path}")
        
        config = get_pipeline_config(mission)
        if config.google_sheets_enabled and config.upload_aggregated_data:
            print(f"\nGoogle Sheets Upload:")
            print(f"  - Sheet ID: {config.google_sheets_id}")
            print(f"  - Status: Enabled")
        elif config.google_sheets_enabled:
            print(f"\nGoogle Sheets: Enabled but upload_aggregated_data is disabled")
        else:
            print(f"\nGoogle Sheets: Disabled")
        
        if 'radar_charts' in results:
            print(f"\nCross-topic analysis generated:")
            cross_topic = results['radar_charts']
            if 'comparison_charts' in cross_topic:
                print(f"  - {len(cross_topic['comparison_charts'])} comparison charts")
            if 'aggregated_csv' in cross_topic:
                print(f"  - Aggregated data: {cross_topic['aggregated_csv']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process batch analysis: {e}")
        raise

def run_comprehensive_analysis(runner: MissionRadarRunner, mission: str, output_dir: Path = None):
    """Run comprehensive analysis for current mission"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running comprehensive analysis for mission: {mission}")
    
    try:
        results = runner.run_comprehensive_analysis()
        logger.info("Successfully completed comprehensive analysis")
        
        print(f"\n=== Comprehensive Analysis Complete (Mission: {mission}) ===")
        
        if 'topics_processed' in results:
            print(f"Processed {len(results['topics_processed'])} topics")
        
        if 'consolidated_files' in results:
            print(f"\nConsolidated CSV files generated:")
            for file_name, file_path in results['consolidated_files'].items():
                print(f"  - {file_name}: {file_path}")
        
        config = get_pipeline_config(mission)
        if config.google_sheets_enabled and config.upload_aggregated_data:
            print(f"\nGoogle Sheets Upload:")
            print(f"  - Sheet ID: {config.google_sheets_id}")
            print(f"  - Status: Enabled")
        elif config.google_sheets_enabled:
            print(f"\nGoogle Sheets: Enabled but upload_aggregated_data is disabled")
        else:
            print(f"\nGoogle Sheets: Disabled")
        
        if 'radar_charts' in results:
            print(f"\nCross-topic analysis generated:")
            cross_topic = results['radar_charts']
            if 'comparison_charts' in cross_topic:
                print(f"  - {len(cross_topic['comparison_charts'])} comparison charts")
            if 'aggregated_csv' in cross_topic:
                print(f"  - Aggregated data: {cross_topic['aggregated_csv']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process comprehensive analysis: {e}")
        raise

def run_radar_charts_only(runner: MissionRadarRunner, mission: str, output_dir: Path = None):
    """Generate radar charts from existing analysis results"""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating radar charts for mission: {mission}")
    
    try:
        results = runner.run_radar_charts_only()
        logger.info("Successfully generated radar charts")
        
        print(f"\n=== Radar Charts Generated (Mission: {mission}) ===")
        
        if 'radar_charts' in results:
            cross_topic = results['radar_charts']
            if 'comparison_charts' in cross_topic:
                print(f"  - {len(cross_topic['comparison_charts'])} comparison charts")
            if 'aggregated_csv' in cross_topic:
                print(f"  - Aggregated data: {cross_topic['aggregated_csv']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to generate radar charts: {e}")
        raise

def list_topics(runner: MissionRadarRunner, mission: str):
    """List available topics for the specified mission"""
    logger = logging.getLogger(__name__)
    logger.info(f"Listing topics for mission: {mission}")
    
    try:
        topics = runner.list_available_topics()
        
        print(f"\n=== Available Topics for {mission} Mission ===")
        
        if not topics:
            print("No topics found for this mission.")
            return
        
        # Group topics by type
        config_topics = [t for t in topics if not t.get('native_category')]
        native_topics = [t for t in topics if t.get('native_category')]
        
        if config_topics:
            print(f"\nConfiguration-based Topics ({len(config_topics)}):")
            for topic in config_topics:
                print(f"  - {topic['name']}")
        
        if native_topics:
            print(f"\nCrunchbase Native Categories ({len(native_topics)}):")
            for topic in native_topics:
                print(f"  - {topic['name']} (native: {topic['native_category']})")
        
        print(f"\nTotal: {len(topics)} topics available")
        
        return topics
        
    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise

def list_all_missions(runner: MissionRadarRunner):
    """List all available missions"""
    logger = logging.getLogger(__name__)
    logger.info("Listing all available missions")
    
    try:
        missions = runner.list_all_missions()
        
        print(f"\n=== Available Missions ===")
        
        for mission in missions:
            print(f"  - {mission['name']}: {mission['description']}")
            print(f"    Topics: {mission['topic_count']}")
        
        return missions
        
    except Exception as e:
        logger.error(f"Failed to list missions: {e}")
        raise

def validate_topic(runner: MissionRadarRunner, topic_name: str, mission: str):
    """Validate topic configuration"""
    logger = logging.getLogger(__name__)
    logger.info(f"Validating topic configuration for: {topic_name} (Mission: {mission})")
    
    try:
        validation_result = runner.validate_topic_config(topic_name)
        
        print(f"\n=== Topic Validation Results for {topic_name} (Mission: {mission}) ===")
        
        if validation_result.get('is_valid'):
            print("✅ Topic configuration is valid")
            
            config = validation_result.get('config', {})
            if config:
                print(f"\nConfiguration Summary:")
                print(f"  - Topic: {config.get('name', 'N/A')}")
                print(f"  - Description: {config.get('description', 'N/A')}")
                
                # Data source configuration
                sources = config.get('data_sources', {})
                if sources:
                    print(f"  - Data Sources:")
                    for source, enabled in sources.items():
                        status = "✅ Enabled" if enabled else "❌ Disabled"
                        print(f"    - {source}: {status}")
                
                # Keywords
                keywords = config.get('keywords', [])
                if keywords:
                    print(f"  - Keywords: {len(keywords)} defined")
                
        else:
            print("❌ Topic configuration has issues:")
            errors = validation_result.get('errors', [])
            for error in errors:
                print(f"  - {error}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate topic {topic_name}: {e}")
        raise

def show_config(mission: str):
    """Show pipeline configuration for the specified mission"""
    logger = logging.getLogger(__name__)
    logger.info(f"Showing configuration for mission: {mission}")
    
    try:
        config = get_pipeline_config(mission)
        
        print(f"\n=== Pipeline Configuration for {mission} Mission ===")
        
        # Mission configuration
        print(f"\nMission Settings:")
        print(f"  - Current Mission: {mission}")
        print(f"  - Current Quarter: {config.current_quarter}")
        print(f"  - Current Year: {config.current_year}")
        
        # Date ranges
        print(f"\nDate Ranges:")
        print(f"  - Data Start: {config.data_start_date}")
        print(f"  - Data End: {config.data_end_date}")
        print(f"  - Analysis Start Quarter: {config.quarterly_start_quarter}")
        print(f"  - Analysis End Quarter: {config.quarterly_end_quarter}")
        
        # Argilla configuration
        argilla_config = config.to_dict().get('argilla', {})
        print(f"\nArgilla Integration:")
        print(f"  - Enabled: {argilla_config.get('enabled', False)}")
        if argilla_config.get('enabled', False):
            print(f"  - Workspace: {argilla_config.get('connection', {}).get('workspace', 'N/A')}")
            print(f"  - Base Sample Size: {argilla_config.get('sampling', {}).get('base_sample_size', 'N/A')}")
        else:
            print(f"  - Status: Manual review disabled")
        
        # Data sources
        print(f"\nData Sources:")
        for source_name in ['crunchbase', 'gtr', 'hansard']:
            enabled = "Enabled" if config.is_source_enabled(source_name) else "Disabled"
            llm_check = "Enabled" if config.should_run_llm_check(source_name) else "Disabled"
            print(f"  - {source_name.upper()}: {enabled} (LLM Check: {llm_check})")
        
        # Output settings
        print(f"\nOutput Settings:")
        print(f"  - Base Directory: {config.output_base_dir}")
        print(f"  - Cache Enabled: {'Yes' if config.cache_enabled else 'No'}")
        
        # Google Sheets
        if config.google_sheets_enabled:
            print(f"  - Google Sheets: Enabled")
            print(f"    - Sheet ID: {config.google_sheets_id}")
            print(f"    - Upload Aggregated Data: {'Yes' if config.upload_aggregated_data else 'No'}")
        else:
            print(f"  - Google Sheets: Disabled")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        raise

@click.group()
@click.option('--mission', required=True, type=click.Choice(['AHL', 'ASF'], case_sensitive=False), 
              help='Mission to process (REQUIRED). AHL: A Healthy Life, ASF: A Sustainable Future.')
@click.option('--config-dir', type=click.Path(exists=True), 
              default=lambda: Path(__file__).parent.parent / "config",
              help='Configuration directory (default: discovery_mission_radar/config)')
@click.option('--output-dir', type=click.Path(), default='outputs',
              help='Output directory (default: outputs)')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO',
              help='Logging level (default: INFO)')
@click.pass_context
def cli(ctx, mission, config_dir, output_dir, log_level):
    """Mission Radar Pipeline CLI
    
    Orchestrates data sources + analysis functions with support for manual review via Argilla.
    
    Examples:
      python -m discovery_mission_radar.pipeline.cli --mission AHL comprehensive
      python -m discovery_mission_radar.pipeline.cli --mission ASF run hydrogen_energy
      python -m discovery_mission_radar.pipeline.cli --mission ahl batch heat_pumps solar
      
      # List topics for specific mission
      python -m discovery_mission_radar.pipeline.cli --mission AHL list
      
      # Show configuration with mission context
      python -m discovery_mission_radar.pipeline.cli --mission ASF config
      
      # Argilla manual review commands
      python -m discovery_mission_radar.pipeline.cli --mission ASF argilla-status
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store configuration in context
    ctx.obj['mission'] = mission.upper()
    ctx.obj['config_dir'] = Path(config_dir)
    ctx.obj['output_dir'] = Path(output_dir)
    
    # Setup logging
    setup_logging(log_level)
    
    # Validate config directory
    if not ctx.obj['config_dir'].exists():
        click.echo(f"Error: Configuration directory not found: {ctx.obj['config_dir']}", err=True)
        sys.exit(1)
    
    # Initialize runner
    try:
        ctx.obj['runner'] = MissionRadarRunner(
            ctx.obj['mission'], 
            ctx.obj['config_dir'], 
            ctx.obj['output_dir']
        )
    except Exception as e:
        click.echo(f"Error: Failed to initialize runner: {e}", err=True)
        sys.exit(1)

    try:
        from discovery_mission_radar.pipeline.data_sources.argilla import ensure_users_from_s3
        click.echo("Provisioning Argilla users from S3: credentials/argilla_users.json")
        summary = ensure_users_from_s3(ctx.obj['mission'], key='credentials/argilla_users.json')
        click.echo(f"Argilla users: created={summary['created']} existing={summary['existing']} errors={summary['errors']}")
    except Exception as e:
        click.echo(f"Warning: Argilla user provisioning skipped: {e}")

@cli.command()
@click.argument('topic_name')
@click.pass_context
def run(ctx, topic_name):
    """Run analysis for a single topic"""
    try:
        run_single_topic(ctx.obj['runner'], topic_name, ctx.obj['mission'], ctx.obj['output_dir'])
    except ValueError as e:
        click.echo(f"Error: Invalid topic '{topic_name}': {e}", err=True)
        click.echo(f"\nTo see all available topics, run:")
        click.echo(f"  python -m discovery_mission_radar.pipeline.cli --mission {ctx.obj['mission']} list")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('topics', nargs=-1, required=True)
@click.pass_context
def batch(ctx, topics):
    """Run batch analysis for multiple topics"""
    try:
        run_batch_analysis(ctx.obj['runner'], list(topics), ctx.obj['mission'], ctx.obj['output_dir'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def comprehensive(ctx):
    """Run comprehensive analysis for current mission"""
    try:
        run_comprehensive_analysis(ctx.obj['runner'], ctx.obj['mission'], ctx.obj['output_dir'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def radar_charts_only(ctx):
    """Generate radar charts from existing analysis results"""
    try:
        run_radar_charts_only(ctx.obj['runner'], ctx.obj['mission'], ctx.obj['output_dir'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def list(ctx):
    """List available topics for the specified mission"""
    try:
        list_topics(ctx.obj['runner'], ctx.obj['mission'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def missions(ctx):
    """List all available missions"""
    try:
        list_all_missions(ctx.obj['runner'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('topic_name')
@click.pass_context
def validate(ctx, topic_name):
    """Validate topic configuration"""
    try:
        validate_topic(ctx.obj['runner'], topic_name, ctx.obj['mission'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def config(ctx):
    """Show pipeline configuration for the specified mission"""
    try:
        show_config(ctx.obj['mission'])
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--quarter', help='Specific quarter (optional, defaults to current)')
@click.option('--topic', help='Specific topic (optional, shows all if not specified)')
@click.pass_context
def argilla_status(ctx, quarter, topic):
    """Show Argilla review status for mission/quarter"""
    try:
        config = get_pipeline_config(ctx.obj['mission'])
        
        argilla_config = config.to_dict().get('argilla', {})
        if not argilla_config.get('enabled', False):
            click.echo("Argilla is not enabled in the configuration.")
            return
        
        if not quarter:
            quarter = config.current_quarter
        
        click.echo(f"\nArgilla Review Status for {ctx.obj['mission']} Mission")
        click.echo(f"Quarter: {quarter}")
        
        try:
            from discovery_mission_radar.pipeline.data_sources.argilla import get_argilla_client
            client = get_argilla_client()
        except ValueError as e:
            click.echo(str(e))
            return
        
        workspace = argilla_config.get('workspace', 'argilla')
        
        try:
            all_datasets = []
            for dataset in client.datasets.list():
                if hasattr(dataset, 'workspace') and dataset.workspace.name == workspace:
                    all_datasets.append(dataset)
        except Exception as e:
            click.echo(f"Error accessing workspace: {e}")
            return
        
        mission_datasets = [
            dataset for dataset in all_datasets 
            if dataset.name.startswith(f"{ctx.obj['mission']}_")
        ]
        
        if not mission_datasets:
            click.echo(f"No datasets found for {ctx.obj['mission']} mission")
            return
        
        click.echo(f"\nDataset Status:")
        click.echo("-" * 60)
        
        total_records = 0
        total_reviewed = 0
        
        for dataset in mission_datasets:
            try:
                parts = dataset.name.split('_', 2)
                if len(parts) < 3:
                    continue
                    
                source = parts[1]
                topic_name = parts[2]
                
                if topic and topic != topic_name:
                    continue
                
                records_count = 0
                reviewed_count = 0
                
                for record in dataset.records:
                    if (hasattr(record, 'metadata') and record.metadata and 
                        record.metadata.get('quarter') == quarter):
                        records_count += 1
                        
                        if hasattr(record, 'responses') and record.responses:
                            for response in record.responses:
                                if hasattr(response, 'status') and response.status == "submitted":
                                    reviewed_count += 1
                                    break
                
                total_records += records_count
                total_reviewed += reviewed_count
                
                if records_count > 0:
                    completion_rate = (reviewed_count / records_count) * 100
                    status = "COMPLETE" if completion_rate == 100 else "IN PROGRESS" if completion_rate > 0 else "NOT STARTED"
                    click.echo(f"{status}: {topic_name}/{source}: {reviewed_count}/{records_count} ({completion_rate:.1f}%)")
                else:
                    click.echo(f"NO DATA: {topic_name}/{source}: No records for {quarter}")
            
            except Exception as e:
                click.echo(f"ERROR: {dataset.name}: {e}")
        
        click.echo("-" * 60)
        click.echo(f"Summary:")
        click.echo(f"  Datasets found: {len(mission_datasets)}")
        click.echo(f"  Total records: {total_records}")
        click.echo(f"  Reviews completed: {total_reviewed}")
        if total_records > 0:
            overall_completion = (total_reviewed / total_records) * 100
            click.echo(f"  Overall completion: {overall_completion:.1f}%")
        else:
            click.echo(f"  Overall completion: No records found for {quarter}")
        
    except ImportError:
        click.echo("Argilla package not installed. Install with: pip install argilla")
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)


@cli.command(name='argilla-reset-registry')
@click.option('--source', 'source_name', required=True, type=click.Choice(['crunchbase', 'gtr', 'hansard'], case_sensitive=False),
              help='Data source to reset sampling registry for')
@click.option('--no-remote', is_flag=True, default=False, help='Do not delete remote S3 registry object')
@click.option('--delete-datasets/--keep-datasets', default=True, help='Also delete Argilla datasets for this source')
@click.option('--workspace', default='admin', show_default=True, help='Argilla workspace to target when deleting datasets')
@click.pass_context
def argilla_reset_registry(ctx, source_name, no_remote, delete_datasets, workspace):
    """Reset Argilla sampling registry for a source (local, S3, and datasets)."""
    try:
        source_name = source_name.lower()
        mission = ctx.obj['mission']
        cache_dir = ctx.obj['runner'].cache_dir

        from discovery_mission_radar.pipeline.data_sources.argilla import reset_sampling_registry, delete_argilla_datasets_for_source

        ok = reset_sampling_registry(mission, source_name, cache_dir, delete_remote=not no_remote)
        if ok:
            click.echo(f"Sampling registry reset completed for {mission}/{source_name}")
        else:
            click.echo(f"Sampling registry reset completed with warnings for {mission}/{source_name}")

        if delete_datasets:
            deleted = delete_argilla_datasets_for_source(mission, source_name, workspace=workspace)
            click.echo(f"Deleted {deleted} Argilla dataset(s) for {mission}/{source_name} in workspace '{workspace}'")
    except Exception as e:
        click.echo(f"Error: Command failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli() 