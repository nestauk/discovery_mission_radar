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
from discovery_mission_radar.pipeline.data_sources import crunchbase, gtr, hansard
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
        
        logger.info(f"Initialized SimpleRunner with config_dir={config_dir}, output_dir={self.output_dir}")
        logger.info(f"Current mission: {self.pipeline_config.current_mission}")
        logger.info(f"Topics directory: {self.topics_dir}")
    
    def _initialize_getters(self):
        """Initialise getters once to avoid redownloading parquet files"""
        logger.info("Initialising data getters (this may take a few minutes for first run)")
        
        self.getters = {}
        
        if self.pipeline_config['sources']['crunchbase']['enabled']:
            logger.info("Initialising Crunchbase getter")
            self.getters['crunchbase'] = cb_getters.CrunchbaseGetter()
            
        if self.pipeline_config['sources']['gtr']['enabled']:
            logger.info("Initialising GTR getter")
            self.getters['gtr'] = gtr_getters.GtrGetter()
            
        if self.pipeline_config['sources']['hansard']['enabled']:
            logger.info("Initialising Hansard getter")
            self.getters['hansard'] = policy_update.HansardData()
        
        logger.info("All getters initialized successfully")
    
    def run_topic_end_to_end(self, topic_name: str) -> Dict[str, Any]:
        """
        Process single topic through complete pipeline.
        Follows Mission Radar process: Data Sources → Analysis → Charts
        """
        logger.info(f"Processing topic: {topic_name} (Mission: {self.pipeline_config.current_mission})")
        
        # Check if this is a Crunchbase native category
        if self.pipeline_config.is_crunchbase_native_category(topic_name):
            return self._run_crunchbase_native_category(topic_name)
        
        topic_config = self._load_topic_config(topic_name)
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {'topic_name': topic_name, 'mission': self.pipeline_config.current_mission}
        
        # Data Sources Phase (with caching) - using pre-initialized getters and exclusion logic
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'crunchbase'):
            logger.info("Processing Crunchbase data source")
            cb_data = crunchbase.get_cb_data(topic_name, self.cache_dir, topic_config, self.getters['crunchbase'])
            cb_results = crunchbase_analysis.produce_cb_stats(cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase'])
            results['crunchbase'] = cb_results
        else:
            logger.info("Skipping Crunchbase data source (excluded or disabled)")
            
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'gtr'):
            logger.info("Processing GTR data source")
            gtr_data = gtr.get_gtr_data(topic_name, self.cache_dir, topic_config, self.getters['gtr'])
            gtr_results = gtr_analysis.produce_gtr_stats(gtr_data, topic_output_dir / "gtr", self.getters['gtr'])
            results['gtr'] = gtr_results
        else:
            logger.info("Skipping GTR data source (excluded or disabled)")
            
        if self.pipeline_config.should_run_source_for_topic(topic_name, 'hansard'):
            logger.info("Processing Hansard data source")
            hansard_data = hansard.get_hansard_data(topic_name, self.cache_dir, topic_config, self.getters['hansard'], use_llm_check=False)
            hansard_results = hansard_analysis.produce_hansard_stats(hansard_data, topic_output_dir / "hansard", self.getters['hansard'])
            results['hansard'] = hansard_results
        else:
            logger.info("Skipping Hansard data source (excluded or disabled)")
        
        logger.info(f"Completed processing: {topic_name}")
        return results
    
    def _run_crunchbase_native_category(self, topic_name: str) -> Dict[str, Any]:
        """Process a topic using Crunchbase native categories"""
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
        
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        
        cb_analysis = crunchbase_analysis.produce_cb_stats(
            cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase']
        )
        
        return {
            'topic_name': topic_name,
            'mission': self.pipeline_config.current_mission,
            'config': mock_config,
            'crunchbase': cb_analysis,
            'native_category': True
        }
    
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
        
        # Check if this is a Crunchbase native category
        if self.pipeline_config.is_crunchbase_native_category(topic_name) and run_crunchbase:
            return self._run_crunchbase_native_category(topic_name)
        
        topic_config = self._load_topic_config(topic_name)
        topic_output_dir = self.output_dir / f"{self.pipeline_config.current_mission}" / topic_name
        topic_output_dir.mkdir(parents=True, exist_ok=True)
        result = {'topic_name': topic_name, 'mission': self.pipeline_config.current_mission, 'config': topic_config}
        
        # Run Crunchbase if selected
        if run_crunchbase:
            logger.info("Processing Crunchbase data source")
            cb_data = crunchbase.get_cb_data(topic_name, self.cache_dir, topic_config, self.getters['crunchbase'])
            cb_analysis = crunchbase_analysis.produce_cb_stats(
                cb_data, topic_output_dir / "crunchbase", self.getters['crunchbase']
            )
            result['crunchbase'] = cb_analysis
        
        # Run GTR if selected
        if run_gtr:
            logger.info("Processing GTR data source")
            gtr_data = gtr.get_gtr_data(topic_name, self.cache_dir, topic_config, self.getters['gtr'])
            gtr_analysis_result = gtr_analysis.produce_gtr_stats(
                gtr_data, topic_output_dir / "gtr", self.getters['gtr']
            )
            result['gtr'] = gtr_analysis_result
        
        # Run Hansard if selected
        if run_hansard:
            logger.info("Processing Hansard data source") 
            hansard_data = hansard.get_hansard_data(topic_name, self.cache_dir, topic_config, self.getters['hansard'], use_llm_check=False)
            hansard_analysis_result = hansard_analysis.produce_hansard_stats(
                hansard_data, topic_output_dir / "hansard", self.getters['hansard']
            )
            result['hansard'] = hansard_analysis_result
        
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
                    crunchbase.validate_cb_config(topic_config)
                except ValueError as e:
                    errors.append(f"Crunchbase validation: {e}")
            
            if self.pipeline_config.should_run_source_for_topic(topic_name, 'gtr'):
                try:
                    gtr.validate_gtr_config(topic_config)
                except ValueError as e:
                    errors.append(f"GTR validation: {e}")
            
            if self.pipeline_config.should_run_source_for_topic(topic_name, 'hansard'):
                try:
                    hansard.validate_hansard_config(topic_config)
                except ValueError as e:
                    errors.append(f"Hansard validation: {e}")
                    
        except Exception as e:
            errors.append(f"Failed to load topic config: {e}")
        
        return errors
    
    def _load_topic_config(self, topic_name: str) -> Dict[str, Any]:
        """Load topic configuration with basic validation from mission-specific directory"""
        # Skip loading for native categories
        if self.pipeline_config.is_crunchbase_native_category(topic_name):
            raise ValueError(f"Topic {topic_name} is a native category, no config file needed")
        
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