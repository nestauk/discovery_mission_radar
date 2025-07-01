"""
Mission Radar Pipeline runner

Orchestrates data sources + analysis functions
"""
from pathlib import Path
from typing import Dict, Any, List
import logging
import yaml

from discovery_utils.getters import crunchbase as cb_getters, gtr as gtr_getters
from discovery_utils.synthesis.policy import policy_update
from discovery_mission_radar.pipeline.data_sources import CrunchbaseDataSource, GtrDataSource, HansardDataSource
from discovery_mission_radar.pipeline.analysis import crunchbase_analysis, gtr_analysis, hansard_analysis, aggregation, consolidation
from discovery_mission_radar.pipeline.config_manager import get_pipeline_config

logger = logging.getLogger(__name__)

class MissionRadarRunner:
    """
    Orchestrator that for the Mission Radar Pipeline.
    Data Sources → Analysis → Charts
    """
    
    def __init__(self, config_dir, output_dir=None):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.cache_dir = self.output_dir / ".cache"
        
        self.pipeline_config = get_pipeline_config()
        
        # Get mission-specific topics directory
        self.topics_dir = self.pipeline_config.get_topics_directory(self.config_dir)
        
        # Initialise getters once to avoid redownloading parquet files
        self._initialize_getters()
        
        # Initialise data sources
        self._initialize_data_sources()
        
        logger.info(f"Initialized MissionRadarRunner with config_dir={config_dir}, output_dir={self.output_dir}")
        logger.info(f"Current mission: {self.pipeline_config.current_mission}")
        logger.info(f"Topics directory: {self.topics_dir}")
    
    def _initialize_getters(self):
        """Initialise getters once to avoid redownloading parquet files"""
        logger.info("Initialising data getters (this may take a few minutes for first run)")
        
        self.getters = {}
        
        if self.pipeline_config.is_source_enabled('crunchbase'):
            logger.info("Initialising Crunchbase getter")
            self.getters['crunchbase'] = cb_getters.CrunchbaseGetter()
            
        if self.pipeline_config.is_source_enabled('gtr'):
            logger.info("Initialising GTR getter")
            self.getters['gtr'] = gtr_getters.GtrGetter()
            
        if self.pipeline_config.is_source_enabled('hansard'):
            logger.info("Initialising Hansard getter")
            self.getters['hansard'] = policy_update.HansardData()
        
        logger.info("All getters initialized successfully")
    
    def _initialize_data_sources(self):
        """Initialise data source instances"""
        logger.info("Initialising data sources")
        
        self.data_sources = {}
        
        if self.pipeline_config.is_source_enabled('crunchbase'):
            self.data_sources['crunchbase'] = CrunchbaseDataSource()
            
        if self.pipeline_config.is_source_enabled('gtr'):
            self.data_sources['gtr'] = GtrDataSource()
            
        if self.pipeline_config.is_source_enabled('hansard'):
            self.data_sources['hansard'] = HansardDataSource()
        
        logger.info("All data sources initialized successfully")
    
    def run_topic_end_to_end(self, topic_name: str) -> Dict[str, Any]:
        """
        Process single topic through complete pipeline.
        Follows Mission Radar process: Data Sources → Analysis → Charts
        
        Handles topics with Crunchbase native categories by:
        - Using native category for Crunchbase if available and not excluded
        - Using config-based processing for GTR/Hansard if topic config exists and not excluded
        """
        logger.info(f"Processing topic: {topic_name} (Mission: {self.pipeline_config.current_mission})")
        
        # Try to load topic config first (needed for GTR/Hansard even if Crunchbase uses native category)
        topic_config = None
        has_config = False
        try:
            topic_config = self._load_topic_config(topic_name)
            has_config = True
        except ValueError:
            # No config file exists - check if it's a pure native category
            if not self.pipeline_config.is_crunchbase_native_category(topic_name):
                # Neither config nor native category - invalid topic
                raise ValueError(f"Topic {topic_name} has no config file and is not a native category")
            # Pure native category - only Crunchbase processing possible
            has_config = False
        
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {'topic_name': topic_name, 'mission': self.pipeline_config.current_mission}
        
        # Data Sources Phase - handle each source appropriately
        
        # Crunchbase: Use native category if available, otherwise config-based
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'crunchbase'):
            if self.pipeline_config.is_crunchbase_native_category(topic_name):
                logger.info("Processing Crunchbase native category")
                cb_results = self._process_crunchbase_native_category(topic_name, topic_output_dir)
                results['crunchbase'] = cb_results
            elif has_config:
                logger.info("Processing Crunchbase data source")
                cb_data = self.data_sources['crunchbase'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['crunchbase'], 
                    mission=self.pipeline_config.current_mission
                )
                cb_results = crunchbase_analysis.produce_cb_stats(cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase'])
                results['crunchbase'] = cb_results
            else:
                logger.info("Skipping Crunchbase data source (no config available)")
        else:
            logger.info("Skipping Crunchbase data source (excluded or disabled)")
            
        # GTR: Always use config-based processing if config exists
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'gtr'):
            if has_config:
                logger.info("Processing GTR data source")
                gtr_data = self.data_sources['gtr'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['gtr'],
                    mission=self.pipeline_config.current_mission
                )
                gtr_results = gtr_analysis.produce_gtr_stats(gtr_data, topic_output_dir / "gtr", self.getters['gtr'])
                results['gtr'] = gtr_results
            else:
                logger.info("Skipping GTR data source (no config available)")
        else:
            logger.info("Skipping GTR data source (excluded or disabled)")
            
        # Hansard: Always use config-based processing if config exists  
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'hansard'):
            if has_config:
                logger.info("Processing Hansard data source")
                hansard_data = self.data_sources['hansard'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['hansard'], 
                    use_llm_check=False, mission=self.pipeline_config.current_mission
                )
                hansard_results = hansard_analysis.produce_hansard_stats(hansard_data, topic_output_dir / "hansard", self.getters['hansard'])
                results['hansard'] = hansard_results
            else:
                logger.info("Skipping Hansard data source (no config available)")
        else:
            logger.info("Skipping Hansard data source (excluded or disabled)")
        
        # Add config to results if available
        if has_config:
            results['config'] = topic_config
        
        logger.info(f"Completed processing: {topic_name}")
        return results
    
    def _process_crunchbase_native_category(self, topic_name: str, topic_output_dir: Path) -> Dict[str, Any]:
        """Process Crunchbase using native categories"""
        cb_category = self.pipeline_config.get_crunchbase_category_for_topic(topic_name)
        logger.info(f"Processing Crunchbase native category: {topic_name} -> {cb_category}")
        
        # Create a mock config for native categories
        mock_config = {
            'search_recipe': {
                'category_name': topic_name,
                'description': f'Companies in {cb_category} category'
            }
        }
        
        # Get companies using native Crunchbase categories
        CB = self.getters['crunchbase']
        company_ids = CB.get_companies_in_categories([cb_category], "narrow").id.tolist()
        
        # Create mock data structure
        cb_data = {
            'ids': company_ids,
            'topic_name': topic_name,
            'config': mock_config,
            'companies_count': len(company_ids),
            'relevant_count': len(company_ids)  # No LLM filtering for native categories
        }
        
        return crunchbase_analysis.produce_cb_stats(
            cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase']
        )
    
    def run_batch_analysis(self, topics: List[str]) -> Dict[str, Any]:
        """Run analysis on multiple topics and consolidate results"""
        logger.info(f"Starting batch analysis for {len(topics)} topics: {topics} (Mission: {self.pipeline_config.current_mission})")
        
        topic_results = []
        
        for topic in topics:
            logger.info(f"Processing topic: {topic}")
            try:
                result = self.run_topic_end_to_end(topic)
                topic_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {topic}: {e}")
                continue
        
        # Create mission-specific output directory
        mission_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}_batch"
        mission_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all results
        consolidated_files = consolidation.consolidate_all_topics(topic_results, mission_output_dir)
        
        # Generate cross-topic radar charts
        radar_charts = aggregation.produce_radar_charts(topic_results, mission_output_dir)
        
        return {
            'mission': self.pipeline_config.current_mission,
            'topic_results': topic_results,
            'consolidated_files': consolidated_files,
            'radar_charts': radar_charts
        }

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis for current mission including:
        - All available topics (with selective data source processing based on exclusions)
        - Crunchbase native categories
        """
        logger.info(f"Starting comprehensive analysis for mission: {self.pipeline_config.current_mission}")
        
        available_topics = self.list_available_topics()
        
        native_categories = self.pipeline_config.get_crunchbase_native_categories()
        
        all_topics = list(set(available_topics + list(native_categories.keys())))
        
        logger.info(f"Processing {len(available_topics)} config-based topics and {len(native_categories)} native categories")
        logger.info(f"Total topics to process: {len(all_topics)}")
        
        topic_results = []
        
        # Process all topics with selective data source processing
        for topic in all_topics:
            logger.info(f"Processing topic: {topic}")
            try:
                result = self.run_topic_end_to_end(topic)
                topic_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {topic}: {e}")
                continue
        
        # Create mission-specific output directory
        mission_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}"
        mission_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all results
        consolidated_files = consolidation.consolidate_all_topics(topic_results, mission_output_dir)
        
        # Generate cross-topic radar charts
        radar_charts = aggregation.produce_radar_charts(topic_results, mission_output_dir)
        
        logger.info(f"✅ Comprehensive analysis completed for mission: {self.pipeline_config.current_mission}!")
        return {
            'mission': self.pipeline_config.current_mission,
            'topic_results': topic_results,
            'consolidated_files': consolidated_files,
            'radar_charts': radar_charts,
            'stats': {
                'config_topics': len(available_topics),
                'native_categories': len(native_categories),
                'total_processed': len(topic_results)
            }
        }

    def run_prototype_analysis(self) -> Dict[str, Any]:
        """
        Legacy method - now delegates to comprehensive analysis
        """
        logger.info("run_prototype_analysis is deprecated, using run_comprehensive_analysis instead")
        return self.run_comprehensive_analysis()

    def run_topic_with_selective_sources(self, topic_name: str, run_crunchbase: bool, run_gtr: bool, run_hansard: bool) -> Dict[str, Any]:
        """Run topic analysis with selective data sources"""
        logger.info(f"Processing {topic_name}: CB={run_crunchbase}, GTR={run_gtr}, Hansard={run_hansard}")
        
        # Try to load topic config first (needed for GTR/Hansard even if Crunchbase uses native category)
        topic_config = None
        has_config = False
        try:
            topic_config = self._load_topic_config(topic_name)
            has_config = True
        except ValueError:
            # No config file exists - check if it's a pure native category
            if not self.pipeline_config.is_crunchbase_native_category(topic_name):
                # Neither config nor native category - invalid topic
                raise ValueError(f"Topic {topic_name} has no config file and is not a native category")
            # Pure native category - only Crunchbase processing possible
            has_config = False
        
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        result = {'topic_name': topic_name, 'mission': self.pipeline_config.current_mission}
        
        # Add config to results if available
        if has_config:
            result['config'] = topic_config
        
        # Run Crunchbase if selected
        if run_crunchbase:
            if self.pipeline_config.is_crunchbase_native_category(topic_name):
                logger.info("Processing Crunchbase native category")
                cb_results = self._process_crunchbase_native_category(topic_name, topic_output_dir)
                result['crunchbase'] = cb_results
            elif has_config:
                logger.info("Processing Crunchbase data source")
                cb_data = self.data_sources['crunchbase'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['crunchbase'],
                    mission=self.pipeline_config.current_mission
                )
                cb_analysis = crunchbase_analysis.produce_cb_stats(
                    cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase']
                )
                result['crunchbase'] = cb_analysis
            else:
                logger.info("Skipping Crunchbase data source (no config available)")
        
        # Run GTR if selected
        if run_gtr:
            if has_config:
                logger.info("Processing GTR data source")
                gtr_data = self.data_sources['gtr'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['gtr'],
                    mission=self.pipeline_config.current_mission
                )
                gtr_analysis_result = gtr_analysis.produce_gtr_stats(
                    gtr_data, topic_output_dir / "gtr", self.getters['gtr']
                )
                result['gtr'] = gtr_analysis_result
            else:
                logger.info("Skipping GTR data source (no config available)")
        
        # Run Hansard if selected
        if run_hansard:
            if has_config:
                logger.info("Processing Hansard data source") 
                hansard_data = self.data_sources['hansard'].get_data(
                    topic_name, self.cache_dir, topic_config, self.getters['hansard'], 
                    use_llm_check=False, mission=self.pipeline_config.current_mission
                )
                hansard_analysis_result = hansard_analysis.produce_hansard_stats(
                    hansard_data, topic_output_dir / "hansard", self.getters['hansard']
                )
                result['hansard'] = hansard_analysis_result
            else:
                logger.info("Skipping Hansard data source (no config available)")
        
        return result

    def run_crunchbase_native_category(self, topic_name: str, cb_category: str) -> Dict[str, Any]:
        """Run Crunchbase analysis using native Crunchbase categories (like Solar, Wind) - legacy method"""
        logger.warning("run_crunchbase_native_category is deprecated, use run_topic_end_to_end with native category configured in pipeline.yaml")
        
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a mock config for native categories
        mock_config = {
            'search_recipe': {
                'category_name': topic_name,
                'description': f'Companies in {cb_category} category'
            }
        }
        
        # Get companies using native Crunchbase categories
        CB = self.getters['crunchbase']
        company_ids = CB.get_companies_in_categories([cb_category], "narrow").id.tolist()
        
        # Create mock data structure
        cb_data = {
            'ids': company_ids,
            'topic_name': topic_name,
            'config': mock_config,
            'companies_count': len(company_ids),
            'relevant_count': len(company_ids)  # No LLM filtering for native categories
        }
        
        cb_analysis = crunchbase_analysis.produce_cb_stats(
            cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase']
        )
        
        return {
            'topic_name': topic_name,
            'mission': self.pipeline_config.current_mission,
            'config': mock_config,
            'crunchbase': cb_analysis
        }
    
    def list_available_topics(self) -> List[str]:
        """List all available topic configurations for the current mission"""
        if not self.topics_dir.exists():
            logger.warning(f"Topics directory does not exist: {self.topics_dir}")
            return []
        
        topics = []
        for config_file in self.topics_dir.glob("config_*.yaml"):
            topic_name = config_file.stem.replace("config_", "")
            topics.append(topic_name)
        
        logger.info(f"Found {len(topics)} topics for mission {self.pipeline_config.current_mission}")
        return sorted(topics)
    
    def list_topics_by_mission(self, mission: str) -> List[str]:
        """List all available topic configurations for a specific mission"""
        if not self.pipeline_config.is_valid_mission(mission):
            raise ValueError(f"Invalid mission: {mission}. Valid missions: {self.pipeline_config.get_available_missions()}")
        
        mission_topics_dir = self.config_dir / "topics" / mission
        if not mission_topics_dir.exists():
            logger.warning(f"Topics directory does not exist for mission {mission}: {mission_topics_dir}")
            return []
        
        topics = []
        for config_file in mission_topics_dir.glob("config_*.yaml"):
            topic_name = config_file.stem.replace("config_", "")
            topics.append(topic_name)
        
        return sorted(topics)
    
    def validate_topic_config(self, topic_name: str) -> List[str]:
        """Validate a topic configuration and return any errors"""
        errors = []
        
        # Check if it's a native category first
        if self.pipeline_config.is_crunchbase_native_category(topic_name):
            category = self.pipeline_config.get_crunchbase_category_for_topic(topic_name)
            if not category:
                errors.append(f"Native category not found for topic: {topic_name}")
            return errors
        
        try:
            topic_config = self._load_topic_config(topic_name)
            
            if 'search_recipe' not in topic_config:
                errors.append("Missing 'search_recipe' section")
            else:
                if 'category_name' not in topic_config['search_recipe']:
                    errors.append("Missing 'search_recipe.category_name'")
            
            if self.pipeline_config.should_run_source_for_topic(topic_name, 'crunchbase'):
                try:
                    self.data_sources['crunchbase'].validate_config(topic_config)
                except ValueError as e:
                    errors.append(f"Crunchbase validation: {e}")
            
            if self.pipeline_config.should_run_source_for_topic(topic_name, 'gtr'):
                try:
                    self.data_sources['gtr'].validate_config(topic_config)
                except ValueError as e:
                    errors.append(f"GTR validation: {e}")
            
            if self.pipeline_config.should_run_source_for_topic(topic_name, 'hansard'):
                try:
                    self.data_sources['hansard'].validate_config(topic_config)
                except ValueError as e:
                    errors.append(f"Hansard validation: {e}")
                    
        except Exception as e:
            errors.append(f"Failed to load topic config: {e}")
        
        return errors
    
    def _load_topic_config(self, topic_name: str) -> Dict[str, Any]:
        """Load topic configuration with basic validation from mission-specific directory"""
        config_file = self.topics_dir / f"config_{topic_name}.yaml"
        if not config_file.exists():
            raise ValueError(f"Topic config not found: {config_file} (Mission: {self.pipeline_config.current_mission})")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_fields = ['search_recipe']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in {topic_name} config: {field}")
        
        return config
    
    def get_topic_summary(self, topic_name: str) -> Dict[str, Any]:
        """Get a summary of outputs for a topic"""
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        if not topic_output_dir.exists():
            return {'status': 'not_processed', 'files': []}
        
        # Count files in each source directory
        summary = {'status': 'processed', 'sources': {}}
        
        for source in ['crunchbase', 'gtr', 'hansard']:
            source_dir = topic_output_dir / source
            if source_dir.exists():
                csv_files = list((source_dir / "csv").glob("*.csv")) if (source_dir / "csv").exists() else []
                chart_files = list((source_dir / "charts").glob("*.png")) if (source_dir / "charts").exists() else []
                
                summary['sources'][source] = {
                    'csv_files': len(csv_files),
                    'chart_files': len(chart_files),
                    'last_modified': max(
                        [f.stat().st_mtime for f in csv_files + chart_files] or [0]
                    )
                }
        
        return summary 