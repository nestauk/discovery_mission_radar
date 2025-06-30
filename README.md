# Mission Radar Pipeline

The Mission Radar Pipeline is a process for analysing innovation data from three sources: Crunchbase, Gateway to Research (GtR), and Hansard. It is designed to support mission-specific topic processing. It organises and processes data based on selected mission (AHL and ASF) to output the data required to produce the Mission Radar report. 

## 🏗️ Architecture

```
📁 discovery_mission_radar/
├── 📁 config/
│   ├── pipeline.yaml                # Main pipeline config with mission settings
│   └── 📁 topics/
│       ├── 📁 AHL/                  # Aiding Healthy Lives topics
│       └── 📁 ASF/                  # Accelerating Science & Innovation topics
├── 📁 pipeline/
│   ├── 📁 data_sources/              # Data fetching with caching
│   ├── 📁 analysis/                 # Analysis functions
│   ├── runner.py                    # MissionRadarRunner orchestration
│   ├── cli.py                       # Command-line interface
│   └── config_manager.py            # Configuration handling & validation
```

## Usage

The pipeline is controlled via a command-line interface.

### Listing Missions and Topics

```bash
# List all missions and their topics
python -m discovery_mission_radar.pipeline.cli missions

# List topics for the current mission
python -m discovery_mission_radar.pipeline.cli list
```

### Running the Pipeline

```bash
# Run a single topic analysis
python -m discovery_mission_radar.pipeline.cli run <topic_name>

# Run a batch analysis for multiple topics
python -m discovery_mission_radar.pipeline.cli batch <topic_name_1> <topic_name_2>

# Run all topics
python -m discovery_mission_radar.pipeline.cli comprehensive
```

### Configuration

```bash
# Show the current pipeline configuration
python -m discovery_mission_radar.pipeline.cli config

# Validate a topic's configuration
python -m discovery_mission_radar.pipeline.cli validate <topic_name>
```

## Configuration

The primary configuration file for the pipeline is `config/pipeline.yaml`. Here you can set the current mission:

```yaml
mission:
  current_mission: "ASF"  # or "AHL"
```
