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

from discovery_mission_radar.pipeline.runner import MissionRadarRunner
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
        stats = results.get('stats', {})
        print(f"Config-based topics: {stats.get('config_topics', 0)}")
        print(f"Native categories: {stats.get('native_categories', 0)}")
        print(f"Total processed: {stats.get('total_processed', 0)}")
        
        if 'consolidated_files' in results:
            print(f"\nConsolidated CSV files generated:")
            for file_name, file_path in results['consolidated_files'].items():
                print(f"  - {file_name}: {file_path}")
        
        # Show Google Sheets upload status
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
    """Generate cross-topic radar charts from existing analysis results"""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating radar charts for mission: {mission}")
    
    try:
        results = runner.run_radar_charts_only()
        logger.info("Successfully completed radar charts generation")
        
        print(f"\n=== Radar Charts Generation Complete (Mission: {mission}) ===")
        print(f"Topics processed: {results.get('topics_processed', 0)}")
        
        if 'radar_charts' in results:
            cross_topic = results['radar_charts']
            if 'comparison_charts' in cross_topic:
                print(f"\nCross-topic charts generated:")
                for chart_file in cross_topic['comparison_charts']:
                    print(f"  - {chart_file}")
                print(f"\nTotal charts: {len(cross_topic['comparison_charts'])}")
            if 'aggregated_csv' in cross_topic:
                print(f"Aggregated data: {cross_topic['aggregated_csv']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to generate radar charts: {e}")
        raise

def list_topics(runner: MissionRadarRunner, mission: str):
    """List all available topics for the current mission"""
    config = get_pipeline_config(mission)
    topics = runner.list_available_topics()
    native_categories = config.get_crunchbase_native_categories()
    
    if not topics and not native_categories:
        print(f"No topics found for mission: {mission}")
        return
    
    print(f"Available topics for mission {mission}:")
    
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

def list_all_missions(runner: MissionRadarRunner):
    """List all missions and their topics"""
    missions = ['AHL', 'ASF']
    
    print("Available missions and their topics:")
    for mission in missions:
        try:
            topics = runner.list_topics_by_mission(mission)
            print(f"\n{mission}")
            if topics:
                for i, topic in enumerate(topics, 1):
                    print(f"  {i}. {topic}")
                print(f"  Total: {len(topics)} topics")
            else:
                print("  No topics found")
        except Exception as e:
            print(f"  Error listing topics: {e}")

def validate_topic(runner: MissionRadarRunner, topic_name: str, mission: str):
    """Validate a topic configuration"""
    config = get_pipeline_config(mission)
    logger = logging.getLogger(__name__)
    
    print(f"Validating topic: {topic_name} for mission: {mission}")
    
    try:
        available_topics = runner.list_available_topics()
        native_categories = list(config.get_crunchbase_native_categories().keys())

        has_config = False
        try:
            topic_config = runner._load_topic_config(topic_name)
            has_config = True
            print(f"✅ Topic config file found")
        except ValueError:
            print(f"❌ Topic config file not found")
        
        # Check if it's a native category
        is_native = config.is_crunchbase_native_category(topic_name)
        if is_native:
            category = config.get_crunchbase_category_for_topic(topic_name)
            print(f"Crunchbase native category: {category}")
        else:
            print(f"Not a Crunchbase native category")
        
        # Must have either config file or be native category
        if not has_config and not is_native:
            print(f"Topic invalid: no config file and not a native category")
            
            print(f"\nAvailable topics for mission {mission}:")
            if available_topics:
                print(f"  Config-based topics ({len(available_topics)}): {', '.join(available_topics[:5])}{'...' if len(available_topics) > 5 else ''}")
            if native_categories:
                print(f"  Native categories ({len(native_categories)}): {', '.join(native_categories)}")
            
            return
        
        # Check data source availability
        print(f"\nData source availability:")
        for source in ['crunchbase', 'gtr', 'hansard']:
            should_run = config.should_run_source_for_topic(topic_name, source)
            excluded = config.is_topic_excluded_for_source(topic_name, source)
            enabled = config.is_source_enabled(source)
            
            status = "✅" if should_run else "❌"
            reason = ""
            if not enabled:
                reason = "(source disabled)"
            elif excluded:
                reason = "(topic excluded)"
            elif source == 'crunchbase' and not has_config and is_native:
                reason = "(native category - config not required)"
            elif source != 'crunchbase' and not has_config:
                reason = "(no config file)"
            
            print(f"  {status} {source}: {'enabled' if should_run else 'disabled'} {reason}")
        
        print(f"\n✅ Topic validation complete")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        
        try:
            available_topics = runner.list_available_topics()
            native_categories = list(config.get_crunchbase_native_categories().keys())
            
            print(f"\nAvailable topics for mission {mission}:")
            if available_topics:
                print(f"  Config-based topics: {', '.join(available_topics)}")
            if native_categories:
                print(f"  Native categories: {', '.join(native_categories)}")
        except:
            pass
        
        errors = str(e).split('\n')
        for error in errors:
            if error.strip():
                print(f"  - {error}")

def show_config(mission: str):
    """Show current pipeline configuration"""
    config = get_pipeline_config(mission)
    print("Current Pipeline Configuration:")
    print(f"  Mission: {mission}")
    print(f"  Available missions: AHL, ASF")
    print(f"  Data start date: {config.data_start_date}")
    print(f"  Data end date: {config.data_end_date}")
    print(f"  Cache enabled: {config.cache_enabled}")
    print(f"  Cache TTL: {config.cache_ttl_hours} hours")
    print(f"  Log level: {config.log_level}")
    
    # Show Google Sheets configuration
    print(f"\nGoogle Sheets Integration:")
    print(f"  Enabled: {config.google_sheets_enabled}")
    if config.google_sheets_enabled:
        print(f"  Sheet ID for {mission}: {config.google_sheets_id}")
        print(f"  Upload aggregated data: {config.upload_aggregated_data}")
    
    # Show included topics for current mission
    try:
        config_dir = Path(__file__).parent.parent / "config"
        runner = MissionRadarRunner(mission, config_dir, Path("outputs"))
        
        available_topics = runner.list_available_topics()
        native_categories = config.get_crunchbase_native_categories()
        all_topics = set(available_topics + list(native_categories.keys()))
        
        print(f"\nData source coverage for {mission}:")
        for source in ['crunchbase', 'gtr', 'hansard']:
            if not config.is_source_enabled(source):
                print(f"  {source}: disabled")
                continue
                
            included_topics = []
            for topic in all_topics:
                if config.should_run_source_for_topic(topic, source):
                    included_topics.append(topic)
            
            status = f"{len(included_topics)} topics" if included_topics else "no topics"
            print(f"  {source}: {status}")
            if included_topics:
                for topic in included_topics:
                    print(f"    - {topic}")
    
    except Exception as e:
        # Fallback to exclusions if runner creation fails
        print(f"\nData source exclusions for {mission}:")
        for source in ['crunchbase', 'gtr', 'hansard']:
            excluded = config.get_excluded_topics(source, mission)
            status = f"{len(excluded)} excluded" if excluded else "none excluded"
            print(f"  {source}: {status}")
            if excluded:
                for topic in excluded:
                    print(f"    - {topic}")

    # Show native categories for current mission
    native_categories = config.get_crunchbase_native_categories()
    print(f"\nCrunchbase native categories for {mission}: {len(native_categories)}")
    for topic, category in native_categories.items():
        print(f"  {topic} → {category}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Mission Radar Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m discovery_mission_radar.pipeline.cli --mission AHL comprehensive
  python -m discovery_mission_radar.pipeline.cli --mission ASF run hydrogen_energy
  python -m discovery_mission_radar.pipeline.cli --mission ahl batch heat_pumps solar
  
  # List topics for specific mission
  python -m discovery_mission_radar.pipeline.cli --mission AHL list
  
  # Show configuration with mission context
  python -m discovery_mission_radar.pipeline.cli --mission ASF config

Mission Configuration:
  Mission is specified as a required CLI argument
  Mission parameter is case-insensitive (ahl, AHL, asf, ASF all accepted).
  
Selective Processing:
  Data sources can be excluded per topic via 'data_sources.<source>.excluded_topics'
  Crunchbase native categories are configured in 'crunchbase_native_categories'
        """
    )
    
    # REQUIRED
    parser.add_argument(
        '--mission',
        required=True,
        choices=['AHL', 'ASF', 'ahl', 'asf'],
        help='Mission to process (REQUIRED). AHL: A Healthy Life, ASF: A Sustainable Future. Case-insensitive.'
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'batch', 'comprehensive', 'radar_charts_only', 'list', 'missions', 'validate', 'config'],
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
    
    # Normalize mission to uppercase for internal use
    mission = args.mission.upper()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate config directory
    if not args.config_dir.exists():
        logger.error(f"Configuration directory not found: {args.config_dir}")
        sys.exit(1)
    
    # Initialise runner
    try:
        runner = MissionRadarRunner(mission, args.config_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'list':
            list_topics(runner, mission)
            
        elif args.command == 'missions':
            list_all_missions(runner)
            
        elif args.command == 'config':
            show_config(mission)
            
        elif args.command == 'comprehensive':
            run_comprehensive_analysis(runner, mission, args.output_dir)
            
        elif args.command == 'radar_charts_only':
            run_radar_charts_only(runner, mission, args.output_dir)
            
        elif args.command == 'validate':
            if not args.topics:
                logger.error("Topic name required for validate command")
                sys.exit(1)
            validate_topic(runner, args.topics[0], mission)
            
        elif args.command == 'run':
            if not args.topics:
                logger.error("Topic name required for run command")
                sys.exit(1)
            
            topic_name = args.topics[0]
            try:
                run_single_topic(runner, topic_name, mission, args.output_dir)
            except ValueError as e:
                logger.error(f"Invalid topic '{topic_name}': {e}")
                print(f"\nTo see all available topics, run:")
                print(f"  python -m discovery_mission_radar.pipeline.cli --mission {mission} list")
                sys.exit(1)
            
        elif args.command == 'batch':
            if not args.topics:
                logger.error("At least one topic name required for batch command")
                sys.exit(1)
            
            # Validate all topics before processing
            try:
                config = get_pipeline_config(mission)
                available_topics = runner.list_available_topics()
                native_categories = list(config.get_crunchbase_native_categories().keys())
                all_valid_topics = set(available_topics + native_categories)
                
                invalid_topics = [topic for topic in args.topics if topic not in all_valid_topics]
                
                if invalid_topics:
                    logger.error(f"Invalid topics found: {', '.join(invalid_topics)}")
                    print(f"\nTo see all available topics, run:")
                    print(f"  python -m discovery_mission_radar.pipeline.cli --mission {mission} list")
                    sys.exit(1)
                
                run_batch_analysis(runner, args.topics, mission, args.output_dir)
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 