#!/usr/bin/env python3
"""
Mission Radar Pipeline CLI

Orchestrates data sources + analysis functions
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from discovery_mission_radar.pipeline.runner import SimpleRunner
from discovery_mission_radar.pipeline.config_manager import get_pipeline_config

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_single_topic(runner: SimpleRunner, topic_name: str, output_dir: Path = None):
    """Run analysis for a single topic"""
    logger = logging.getLogger(__name__)
    config = get_pipeline_config()
    logger.info(f"Running single topic analysis for: {topic_name} (Mission: {config.current_mission})")
    
    try:
        results = runner.run_topic_end_to_end(topic_name)
        logger.info(f"Successfully completed analysis for {topic_name}")
        
        # Print summary
        print(f"\n=== Analysis Complete for {topic_name} (Mission: {config.current_mission}) ===")
        
        # Check if it's a native category
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

def run_batch_analysis(runner: SimpleRunner, topic_list: List[str], output_dir: Path = None):
    """Run batch analysis for multiple topics"""
    logger = logging.getLogger(__name__)
    config = get_pipeline_config()
    logger.info(f"Running batch analysis for topics: {topic_list} (Mission: {config.current_mission})")
    
    try:
        results = runner.run_batch_analysis(topic_list)
        logger.info("Successfully completed batch analysis")
        
        # Print summary
        print(f"\n=== Batch Analysis Complete (Mission: {config.current_mission}) ===")
        print(f"Processed {len(topic_list)} topics")
        
        if 'consolidated_files' in results:
            print(f"\nConsolidated CSV files generated:")
            for file_name, file_path in results['consolidated_files'].items():
                print(f"  - {file_name}: {file_path}")
        
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

def run_comprehensive_analysis(runner: SimpleRunner, output_dir: Path = None):
    """Run comprehensive analysis for current mission"""
    logger = logging.getLogger(__name__)
    config = get_pipeline_config()
    logger.info(f"Running comprehensive analysis for mission: {config.current_mission}")
    
    try:
        results = runner.run_comprehensive_analysis()
        logger.info("Successfully completed comprehensive analysis")
        
        # Print summary
        print(f"\n=== Comprehensive Analysis Complete (Mission: {config.current_mission}) ===")
        stats = results.get('stats', {})
        print(f"Config-based topics: {stats.get('config_topics', 0)}")
        print(f"Native categories: {stats.get('native_categories', 0)}")
        print(f"Total processed: {stats.get('total_processed', 0)}")
        
        if 'consolidated_files' in results:
            print(f"\nConsolidated CSV files generated:")
            for file_name, file_path in results['consolidated_files'].items():
                print(f"  - {file_name}: {file_path}")
        
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

def list_topics(runner: SimpleRunner):
    """List all available topics for the current mission"""
    config = get_pipeline_config()
    topics = runner.list_available_topics()
    native_categories = config.get_crunchbase_native_categories()
    
    if not topics and not native_categories:
        print(f"No topics found for mission: {config.current_mission}")
        return
    
    print(f"Available topics for mission {config.current_mission}:")
    
    if topics:
        print(f"\nConfig-based topics ({len(topics)}):")
        for i, topic in enumerate(topics, 1):
            excluded_sources = []
            for source in ['crunchbase', 'gtr', 'hansard']:
                if config.is_topic_excluded_for_source(topic, source):
                    excluded_sources.append(source)
            
            exclusion_info = f" (excludes: {', '.join(excluded_sources)})" if excluded_sources else ""
            print(f"  {i}. {topic}{exclusion_info}")
    
    if native_categories:
        print(f"\nCrunchbase native categories ({len(native_categories)}):")
        for i, (topic, category) in enumerate(native_categories.items(), 1):
            print(f"  {i}. {topic} → {category}")
    
    total = len(topics) + len(native_categories)
    print(f"\nTotal: {total} topics")

def list_all_missions(runner: SimpleRunner):
    """List all missions and their topics"""
    config = get_pipeline_config()
    missions = config.get_available_missions()
    
    print("Available missions and their topics:")
    for mission in missions:
        try:
            topics = runner.list_topics_by_mission(mission)
            status = " (current)" if mission == config.current_mission else ""
            print(f"\n{mission}{status}:")
            if topics:
                for i, topic in enumerate(topics, 1):
                    print(f"  {i}. {topic}")
                print(f"  Total: {len(topics)} topics")
            else:
                print("  No topics found")
        except Exception as e:
            print(f"  Error listing topics: {e}")

def validate_topic(runner: SimpleRunner, topic_name: str):
    """Validate a topic configuration"""
    config = get_pipeline_config()
    errors = runner.validate_topic_config(topic_name)
    
    if not errors:
        topic_type = "native category" if config.is_crunchbase_native_category(topic_name) else "config-based"
        print(f"✓ Topic '{topic_name}' configuration is valid ({topic_type}, Mission: {config.current_mission})")
    else:
        print(f"✗ Topic '{topic_name}' configuration has errors (Mission: {config.current_mission}):")
        for error in errors:
            print(f"  - {error}")

def show_config():
    """Show current pipeline configuration"""
    config = get_pipeline_config()
    print("Current Pipeline Configuration:")
    print(f"  Mission: {config.current_mission}")
    print(f"  Available missions: {', '.join(config.get_available_missions())}")
    print(f"  Data start date: {config.data_start_date}")
    print(f"  Data end date: {config.data_end_date}")
    print(f"  Cache enabled: {config.cache_enabled}")
    print(f"  Cache TTL: {config.cache_ttl_hours} hours")
    print(f"  Log level: {config.log_level}")
    
    # Show exclusions for current mission
    current_mission = config.current_mission
    print(f"\nData source exclusions for {current_mission}:")
    for source in ['crunchbase', 'gtr', 'hansard']:
        excluded = config.get_excluded_topics(source, current_mission)
        status = f"{len(excluded)} excluded" if excluded else "none excluded"
        print(f"  {source}: {status}")
        if excluded:
            for topic in excluded:
                print(f"    - {topic}")
    
    # Show native categories for current mission
    native_categories = config.get_crunchbase_native_categories()
    print(f"\nCrunchbase native categories for {current_mission}: {len(native_categories)}")
    for topic, category in native_categories.items():
        print(f"  {topic} → {category}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Mission Radar Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single topic
  python -m discovery_mission_radar.pipeline.cli_new run hydrogen_energy
  
  # Run batch analysis (generates consolidated CSV files)
  python -m discovery_mission_radar.pipeline.cli_new batch hydrogen_energy heat_pumps solar
  
  # Run comprehensive analysis (all topics + native categories)
  python -m discovery_mission_radar.pipeline.cli_new comprehensive
  
  # List available topics for current mission
  python -m discovery_mission_radar.pipeline.cli_new list
  
  # List all missions and their topics
  python -m discovery_mission_radar.pipeline.cli_new missions
  
  # Show current configuration
  python -m discovery_mission_radar.pipeline.cli_new config
  
  # Validate topic configuration
  python -m discovery_mission_radar.pipeline.cli_new validate hydrogen_energy
  
Mission Configuration:
  The current mission is set in config/pipeline.yaml under 'mission.current_mission'.
  Only topics from the current mission's directory will be processed.
  Available missions: AHL, ASF
  
Selective Processing:
  Data sources can be excluded per topic via 'data_sources.<source>.excluded_topics'
  Crunchbase native categories are configured in 'crunchbase_native_categories'
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'batch', 'comprehensive', 'list', 'missions', 'validate', 'config'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'topics',
        nargs='*',
        help='Topic names to process (for run/batch commands)'
    )
    
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path(__file__).parent.parent / "config",
        help='Configuration directory (default: discovery_mission_radar/config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("outputs"),
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate config directory
    if not args.config_dir.exists():
        logger.error(f"Configuration directory not found: {args.config_dir}")
        sys.exit(1)
    
    # Initialize runner
    try:
        runner = SimpleRunner(args.config_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'list':
            list_topics(runner)
            
        elif args.command == 'missions':
            list_all_missions(runner)
            
        elif args.command == 'config':
            show_config()
            
        elif args.command == 'comprehensive':
            run_comprehensive_analysis(runner, args.output_dir)
            
        elif args.command == 'validate':
            if not args.topics:
                logger.error("Topic name required for validate command")
                sys.exit(1)
            validate_topic(runner, args.topics[0])
            
        elif args.command == 'run':
            if not args.topics:
                logger.error("Topic name required for run command")
                sys.exit(1)
            run_single_topic(runner, args.topics[0], args.output_dir)
            
        elif args.command == 'batch':
            if not args.topics:
                logger.error("At least one topic name required for batch command")
                sys.exit(1)
            run_batch_analysis(runner, args.topics, args.output_dir)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 