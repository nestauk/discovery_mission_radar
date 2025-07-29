"""
Pipeline Configuration Manager

Handles loading and providing configuration settings throughout the pipeline,
including parameterized date ranges and current period settings.
"""
import yaml
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class PipelineConfig:
    """Manages pipeline configuration with smart defaults and date handling"""
    
    def __init__(self, config_file: Optional[Path] = None, mission: str = "ASF"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to pipeline.yaml config file. If None, uses default location.
            mission: Required mission parameter.
        """
        mission = mission.upper()
        if mission not in ['AHL', 'ASF']:
            raise ValueError(f"Invalid mission: {mission}. Must be one of: AHL, ASF")
        
        if config_file is None:
            config_file = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        
        self.config_file = config_file
        self._mission = mission
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback defaults"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded pipeline config from {self.config_file}")
                return config
            else:
                logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _validate_config(self, mission: Optional[str] = None):
        """Validate configuration for required fields"""
        required_sections = ['date_ranges', 'current_period', 'data_sources']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
        
        self._validate_google_sheets_config()
        
        self._validate_mission_exclusions()
        
        self._validate_data_sources()
        
        self._validate_date_ranges()
    
    def _validate_google_sheets_config(self):
        """Validate Google Sheets configuration for current mission"""
        google_sheets = self._config.get('output', {}).get('google_sheets', {})
        
        if google_sheets.get('enabled'):
            mission_key = f"{self._mission.lower()}_sheet_id"
            sheet_id = google_sheets.get(mission_key)
            
            if not sheet_id:
                logger.warning(f"Google Sheets enabled but {mission_key} not configured for mission {self._mission}")
            elif len(sheet_id) < 10:
                logger.warning(f"Google Sheets {mission_key} appears invalid (too short): {sheet_id}")
            else:
                logger.info(f"Google Sheets configured for {self._mission}: {sheet_id}")
        else:
            logger.info(f"Google Sheets integration disabled")
    
    def _validate_mission_exclusions(self):
        """Validate mission-specific exclusion lists exist and warn about coverage"""
        data_sources = self._config.get('data_sources', {})
        
        for source_name, source_config in data_sources.items():
            if 'excluded_topics' in source_config:
                excluded_topics = source_config['excluded_topics']
                
                if self._mission not in excluded_topics:
                    logger.warning(f"No exclusion list found for mission {self._mission} in {source_name} data source")
                else:
                    excluded_count = len(excluded_topics[self._mission])
                    if excluded_count > 0:
                        logger.info(f"{source_name}: {excluded_count} topics excluded for mission {self._mission}")
    
    def _validate_data_sources(self):
        """Validate data source configuration and warn about potential issues"""
        data_sources = self._config.get('data_sources', {})
        required_sources = ['crunchbase', 'gtr', 'hansard']
        
        for source in required_sources:
            if source not in data_sources:
                logger.warning(f"Data source '{source}' not configured")
            elif not data_sources[source].get('enabled', True):
                logger.warning(f"Data source '{source}' is disabled")
        
        # Validate native categories for mission
        crunchbase_config = data_sources.get('crunchbase', {})
        native_categories = crunchbase_config.get('native_categories', {})
        
        if self._mission in native_categories:
            native_count = len(native_categories[self._mission])
            if native_count > 0:
                logger.info(f"Crunchbase native categories configured for {self._mission}: {native_count} categories")
        else:
            logger.info(f"No Crunchbase native categories configured for mission {self._mission}")
    
    def _validate_date_ranges(self):
        """Validate date range configuration"""
        date_ranges = self._config.get('date_ranges', {})
        
        try:
            start_date = date_ranges.get('data_start_date')
            end_date = date_ranges.get('data_end_date')
            
            if start_date and end_date:
                from datetime import datetime
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start >= end:
                    logger.warning(f"Data start date ({start_date}) is not before end date ({end_date})")
                
                # Check if end date is in the future (which might be intentional)
                now = datetime.now()
                if end > now:
                    logger.info(f"Data end date ({end_date}) is in the future - this may be intentional for pipeline scheduling")
                    
        except ValueError as e:
            logger.warning(f"Invalid date format in configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Provide sensible defaults if config file is missing"""
        current_year = datetime.now().year
        current_quarter = f"{current_year}-Q{((datetime.now().month-1)//3) + 1}"
        
        return {
            'date_ranges': {
                'data_start_date': '2014-01-01',
                'data_end_date': f'{current_year}-06-30',
                'yearly_analysis': {
                    'start_year': 2014,
                    'end_year': current_year
                },
                'quarterly_analysis': {
                    'start_quarter': '2023-Q1',
                    'end_quarter': current_quarter
                },
                'growth_analysis': {
                    'base_year': 2020,
                    'comparison_year': current_year - 1
                }
            },
            'current_period': {
                'quarter': current_quarter,
                'year': current_year
            },
            'data_sources': {
                'crunchbase': {
                    'enabled': True, 
                    'run_llm_check': True,
                    'excluded_topics': {'ASF': [], 'AHL': []},
                    'native_categories': {
                        'ASF': {
                            'energy_storage': 'Battery',
                            'renewables_general': 'Renewable Energy',
                            'solar': 'Solar',
                            'wind': 'Wind Energy'
                        },
                        'AHL': {}
                    }
                },
                'gtr': {
                    'enabled': True, 
                    'run_llm_check': True,
                    'excluded_topics': {'ASF': [], 'AHL': []}
                },
                'hansard': {
                    'enabled': True, 
                    'run_llm_check': False,
                    'excluded_topics': {'ASF': [], 'AHL': []}
                }
            },
            'output': {
                'base_dir': './outputs',
                'cache_enabled': True,
                'google_sheets': {
                    'enabled': False,
                    'ahl_sheet_id': '',  # Mission-specific sheet IDs
                    'asf_sheet_id': '',
                    'upload_aggregated_data': False
                }
            },
            'execution': {
                'log_level': 'INFO',
                'fail_fast': True
            }
        }
    
    # Mission properties
    @property
    def current_mission(self) -> str:
        """Get the current mission"""
        return self._mission
    
    @property
    def google_sheets_id(self) -> str:
        """Get Google Sheets ID for current mission"""
        mission_key = f"{self._mission.lower()}_sheet_id"
        return self._config.get('output', {}).get('google_sheets', {}).get(mission_key, '')
    
    @property
    def google_sheets_enabled(self) -> bool:
        """Check if Google Sheets integration is enabled"""
        return self._config.get('output', {}).get('google_sheets', {}).get('enabled', False)
    
    @property
    def upload_aggregated_data(self) -> bool:
        """Check if aggregated data upload is enabled"""
        return self._config.get('output', {}).get('google_sheets', {}).get('upload_aggregated_data', False)
    
    @property
    def categories_to_show(self) -> Dict[str, List[str]]:
        """Get categories to show in cross-topic charts by mission"""
        return self._config.get('categories_to_show', {
            'ASF': ["heat_pumps", "biomass_heating", "wind", "solar", "hydrogen_energy", "energy_efficiency", "ccus"],  
            'AHL': ["reformulation_fiber", "reformulation_salt", "reformulation_fat", "reformulation_general", "reformulation_sugar"]
        })
    
    def get_available_missions(self) -> List[str]:
        """Get list of available missions"""
        return ['AHL', 'ASF']
    
    def get_topics_directory(self, base_config_dir: Path) -> Path:
        """Get the topics directory for the current mission"""
        return base_config_dir / "topics" / self.current_mission
        
    def is_valid_mission(self, mission: str) -> bool:
        """Check if a mission name is valid"""
        return mission in self.get_available_missions()

    # Data source exclusion methods
    def get_excluded_topics(self, source: str, mission: str = None) -> List[str]:
        """Get list of topics excluded for a specific data source and mission"""
        if mission is None:
            mission = self.current_mission
        
        return self._config.get('data_sources', {}).get(source, {}).get('excluded_topics', {}).get(mission, [])
    
    def is_topic_excluded_for_source(self, topic_name: str, source: str, mission: str = None) -> bool:
        """Check if a topic is excluded for a specific data source"""
        excluded_topics = self.get_excluded_topics(source, mission)
        return topic_name in excluded_topics
    
    def should_run_source_for_topic(self, topic_name: str, source: str, mission: str = None) -> bool:
        """Check if a data source should be run for a specific topic (not excluded and source enabled)"""
        return self.is_source_enabled(source) and not self.is_topic_excluded_for_source(topic_name, source, mission)

    # Crunchbase native categories (now mission-specific)
    def get_crunchbase_native_categories(self, mission: str = None) -> Dict[str, str]:
        """Get mapping of topic names to Crunchbase native category names for a specific mission"""
        if mission is None:
            mission = self.current_mission
        
        return self._config.get('data_sources', {}).get('crunchbase', {}).get('native_categories', {}).get(mission, {})
    
    def is_crunchbase_native_category(self, topic_name: str, mission: str = None) -> bool:
        """Check if a topic uses Crunchbase native categories"""
        return topic_name in self.get_crunchbase_native_categories(mission)
    
    def get_crunchbase_category_for_topic(self, topic_name: str, mission: str = None) -> Optional[str]:
        """Get the Crunchbase native category name for a topic"""
        return self.get_crunchbase_native_categories(mission).get(topic_name)

    # Date range properties
    @property
    def data_start_date(self) -> str:
        """Start date for data fetching (YYYY-MM-DD format)"""
        return self._config['date_ranges']['data_start_date']
    
    @property 
    def data_end_date(self) -> str:
        """End date for data fetching (YYYY-MM-DD format)"""
        return self._config['date_ranges']['data_end_date']
    
    @property
    def yearly_start_year(self) -> int:
        """Start year for yearly analysis"""
        return self._config['date_ranges']['yearly_analysis']['start_year']
    
    @property
    def yearly_end_year(self) -> int:
        """End year for yearly analysis"""
        return self._config['date_ranges']['yearly_analysis']['end_year']
    
    @property
    def quarterly_start_quarter(self) -> str:
        """Start quarter for quarterly analysis (YYYY-QX format)"""
        return self._config['date_ranges']['quarterly_analysis']['start_quarter']
    
    @property
    def quarterly_end_quarter(self) -> str:
        """End quarter for quarterly analysis (YYYY-QX format)"""
        return self._config['date_ranges']['quarterly_analysis']['end_quarter']
    
    @property
    def current_quarter(self) -> str:
        """Current quarter (YYYY-QX format)"""
        return self._config['current_period']['quarter']
    
    @property
    def current_year(self) -> int:
        """Current year"""
        return self._config['current_period']['year']
    
    @property
    def growth_base_year(self) -> int:
        """Base year for growth analysis"""
        return self._config['date_ranges']['growth_analysis']['base_year']
    
    @property
    def growth_comparison_year(self) -> int:
        """Comparison year for growth analysis"""
        return self._config['date_ranges']['growth_analysis']['comparison_year']
    
    # Data source properties
    def is_source_enabled(self, source: str) -> bool:
        """Check if a data source is enabled"""
        return self._config['data_sources'].get(source, {}).get('enabled', False)
    
    def should_run_llm_check(self, source: str) -> bool:
        """Check if LLM check should be run for a data source"""
        return self._config['data_sources'].get(source, {}).get('run_llm_check', False)
    
    # Output properties
    @property
    def output_base_dir(self) -> str:
        """Base directory for outputs"""
        return self._config['output']['base_dir']
    
    @property
    def cache_enabled(self) -> bool:
        """Whether caching is enabled"""
        return self._config['output']['cache_enabled']
    
    # Execution properties
    @property
    def log_level(self) -> str:
        """Logging level"""
        return self._config['execution']['log_level']
    
    @property
    def fail_fast(self) -> bool:
        """Whether to stop on first error"""
        return self._config['execution']['fail_fast']
    
    # Utility methods
    def get_date_query_filter(self) -> str:
        """Get pandas query filter for date range"""
        return f"date >= '{self.data_start_date}' & date <= '{self.data_end_date}'"
    
    def get_yearly_analysis_filter(self) -> str:
        """Get pandas query filter for yearly analysis"""
        return f"date >= '{self.yearly_start_year}-01-01'"
    
    def get_quarterly_analysis_filter(self) -> str:
        """Get pandas query filter for quarterly analysis"""
        start_year = int(self.quarterly_start_quarter.split('-')[0])
        return f"date >= '{start_year}-01-01'"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._config.copy()

# Global config functions - mission now required
def get_pipeline_config(mission: str) -> PipelineConfig:
    """Get pipeline configuration instance for specific mission"""
    return PipelineConfig(mission=mission)

def reload_config(config_file: Optional[Path] = None, mission: str = "ASF") -> PipelineConfig:
    """Reload configuration from file with mission parameter"""
    return PipelineConfig(config_file, mission) 