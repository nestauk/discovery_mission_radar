# Mission Radar Pipeline

The Mission Radar Pipeline is a process for analysing innovation data from three sources: Crunchbase, Gateway to Research (GtR), and Hansard.


## 🏗️ Architecture

```
📁 discovery_mission_radar/
├── 📁 config/
│   ├── pipeline.yaml                # Main pipeline config
│   └── 📁 topics/
│       ├── 📁 AHL/                  # A Healthy Life topics
│       └── 📁 ASF/                  # A Sustainable Future topics
├── 📁 pipeline/
│   ├── 📁 data_sources/              # Data fetching and relevence checking
│   ├── 📁 analysis/                 # Analysis functions
│   ├── runner.py                    # MissionRadarRunner orchestration
│   ├── cli.py                       # Command-line interface
│   └── config_manager.py            # Configuration handling & validation
├── 📁 outputs/
│   ├── 📁 .cache/
│   │   ├── 📁 ahl/                  # Mission-specific cache (AHL)
│   │   └── 📁 asf/                  # Mission-specific cache (ASF)
│   ├── 📁 AHL/                      # AHL analysis outputs
│   └── 📁 ASF/                      # ASF analysis outputs
```

## 🚀 Quick Start

All commands require the `--mission` parameter to specify which mission to process:

```bash
# Activate environment
source .venv/bin/activate

# View available topics for a mission
python -m discovery_mission_radar.pipeline.cli --mission AHL list
python -m discovery_mission_radar.pipeline.cli --mission ASF list

# Show configuration for a specific mission
python -m discovery_mission_radar.pipeline.cli --mission AHL config

# Run a single topic
python -m discovery_mission_radar.pipeline.cli --mission AHL run reformulation_sugar
python -m discovery_mission_radar.pipeline.cli --mission ASF run hydrogen_energy
```

## 📖 Usage

### Mission Management

```bash
# Show mission-specific configuration
python -m discovery_mission_radar.pipeline.cli --mission AHL config
python -m discovery_mission_radar.pipeline.cli --mission ASF config

# List topics for a specific mission
python -m discovery_mission_radar.pipeline.cli --mission AHL list
python -m discovery_mission_radar.pipeline.cli --mission ASF list
```

### Running Analysis

```bash
# Single topic analysis
python -m discovery_mission_radar.pipeline.cli --mission AHL run reformulation_sugar
python -m discovery_mission_radar.pipeline.cli --mission ASF run heat_pumps

# Batch analysis for multiple topics
python -m discovery_mission_radar.pipeline.cli --mission AHL batch reformulation_sugar reformulation_salt
python -m discovery_mission_radar.pipeline.cli --mission ASF batch heat_pumps solar hydrogen_energy

# Comprehensive analysis (all topics for mission)
python -m discovery_mission_radar.pipeline.cli --mission AHL comprehensive
python -m discovery_mission_radar.pipeline.cli --mission ASF comprehensive

# Generate cross-topic radar charts only
python -m discovery_mission_radar.pipeline.cli --mission AHL radar_charts_only
```

### Topic Validation

```bash
# Validate topic configuration and data source availability
python -m discovery_mission_radar.pipeline.cli --mission AHL validate reformulation_sugar
python -m discovery_mission_radar.pipeline.cli --mission ASF validate hydrogen_energy
```



## ⚙️ Configuration

### Mission-Specific Settings

The pipeline automatically handles mission-specific configuration:

- **Google Sheets**: Each mission has its own sheet ID for data export
- **Caching**: Mission-specific cache directories prevent data conflicts
- **Topic Exclusions**: Data sources can exclude specific topics per mission
- **Native Categories**: Crunchbase native categories mapped per mission

### Data Source Coverage

Each mission has different data source coverage:

**AHL Mission**: 7 active topics across all sources
- Focused on food tech, nutrition, and health innovation
- Selective processing based on exclusion lists

**ASF Mission**: Variable coverage (11-18 topics per source)
- Comprehensive energy and sustainability coverage
- Includes 4 Crunchbase native categories

### Pipeline Configuration File

The main configuration is in `config/pipeline.yaml`:

```yaml
# No mission setting - specified at runtime via CLI
date_ranges:
  data_start_date: '2014-01-01'
  data_end_date: '2025-06-30'

data_sources:
  crunchbase:
    enabled: true
    excluded_topics:
      AHL: [alt_protein_general, cloud_kitchen, ...]  # 16 excluded
      ASF: [bioenergy, decarbonisation_general, ...]  # 9 excluded
    native_categories:
      ASF:
        energy_storage: 'Battery'
        renewables_general: 'Renewable Energy'
        solar: 'Solar'
        wind: 'Wind Energy'
      AHL: {}

output:
  google_sheets:
    enabled: true
    ahl_sheet_id: 'your-ahl-sheet-id'
    asf_sheet_id: 'your-asf-sheet-id'
```

## 🎯 Mission Topics

### AHL (A Healthy Life) - 23 Topics
- Food technology: `reformulation_sugar`, `reformulation_salt`, `reformulation_fat`, `reformulation_fiber`
- Alternative proteins: `alt_protein_general`, `lab_meat`, `plant_based`, `insects`
- Food systems: `meal_kits`, `delivery_apps`, `restaurants`, `supply_chain`
- Health & nutrition: `personalised_nutrition`, `weight_loss_drugs`, `health_general`

### ASF (A Sustainable Future) - 21 Topics  
- Energy generation: `solar`, `wind`, `hydrogen_energy`, `geothermal_energy`
- Energy systems: `energy_storage`, `energy_grid`, `energy_efficiency`
- Heating & cooling: `heat_pumps`, `district_heating`, `biomass_heating`
- Carbon management: `ccus`, `renewables_general`

## 🔍 Data Sources

### Crunchbase
- Company data, funding rounds, acquisitions
- LLM-based relevance filtering
- Native category support for ASF mission

### Gateway to Research (GtR)
- UK research project data
- UKRI funding information
- Category-based filtering with LLM validation

### Hansard
- UK Parliamentary debate transcripts
- Policy discussion analysis
- Keyword and topic-based filtering

### Overton
- Government publications
- UK and international searches
- Keyword search with LLM validation

## 📊 Outputs

### Generated Files
- **CSV Data**: Topic-specific data exports for each source
- **Charts**: Visualisation outputs for trends and analysis
- **Consolidated Data**: Cross-topic aggregated datasets
- **Radar Charts**: Cross-topic comparison visualizations

### Google Sheets Integration
- Automatic upload of aggregated data to mission-specific sheets
- Configurable upload settings per mission
- Separate sheet IDs for AHL and ASF missions

## 🛠️ Development

### Requirements
- Python 3.8+
- Virtual environment with dependencies from `requirements.txt`
- AWS credentials for S3
- OpenAI API key for LLM relevance checking

### Installation
```bash
# Clone repository
git clone [repository-url]
cd discovery_mission_radar

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Copy `.env.example` to `.env` and configure

## 📋 Examples

### Basic Workflow
```bash
# 1. Check mission configuration
python -m discovery_mission_radar.pipeline.cli --mission AHL config

# 2. List available topics
python -m discovery_mission_radar.pipeline.cli --mission AHL list

# 3. Validate a topic
python -m discovery_mission_radar.pipeline.cli --mission AHL validate reformulation_sugar

# 4. Run single topic analysis
python -m discovery_mission_radar.pipeline.cli --mission AHL run reformulation_sugar

# 5. Run comprehensive analysis
python -m discovery_mission_radar.pipeline.cli --mission AHL comprehensive
```

### Cross-Mission Analysis
```bash
# Compare the same topic across missions (if available)
python -m discovery_mission_radar.pipeline.cli --mission AHL run health_general
python -m discovery_mission_radar.pipeline.cli --mission ASF run green_skills

# Process both missions comprehensively
python -m discovery_mission_radar.pipeline.cli --mission AHL comprehensive
python -m discovery_mission_radar.pipeline.cli --mission ASF comprehensive
```

## 🔧 Troubleshooting

### Common Issues

**Mission argument missing**:
```bash
# ❌ This will fail
python -m discovery_mission_radar.pipeline.cli list

# ✅ This will work
python -m discovery_mission_radar.pipeline.cli --mission AHL list
```

**Invalid topic name**:
```bash
# Use validate command to check topic configuration
python -m discovery_mission_radar.pipeline.cli --mission AHL validate invalid_topic_name
```