import graphviz
from pathlib import Path

# Create the flow diagram in Graphviz DOT format
dot = graphviz.Digraph(name="MissionRadarPipeline", format="png")
dot.attr(rankdir="LR", fontname="Sans")

# Subgraph: User entry-point
with dot.subgraph(name="cluster_user") as c:
    c.attr(label="User entry‑point")
    c.node("CLI", "CLI\n(pipeline/cli.py)", shape="box")

# Subgraph: Initialisation
with dot.subgraph(name="cluster_init") as c:
    c.attr(label="Initialisation")
    c.node("CFG", "config/pipeline.yaml", shape="box")
    c.node("RUNNER", "MissionRadarRunner\n(pipeline/runner.py)", shape="box")

# Subgraph: For each topic
with dot.subgraph(name="cluster_topic") as c:
    c.attr(label="For each topic")
    c.node("DS_CR", "CrunchbaseDataSource", shape="box")
    c.node("DS_GTR", "GtrDataSource", shape="box")
    c.node("DS_HAN", "HansardDataSource", shape="box")
    c.node("AN_CR", "crunchbase_analysis", shape="box")
    c.node("AN_GTR", "gtr_analysis", shape="box")
    c.node("AN_HAN", "hansard_analysis", shape="box")

# Subgraph: Cross-topic aggregation
with dot.subgraph(name="cluster_agg") as c:
    c.attr(label="Cross-topic aggregation")
    c.node("CONSOL", "consolidate_all_topics", shape="box")
    c.node("AGGR", "produce_radar_charts", shape="box")

# Output node
dot.node("OUTPUT", "Outputs\n(CSVs & radar PNGs)", shape="box")

# Edges
dot.edges([
    ("CLI", "RUNNER"),
    ("CFG", "RUNNER"),
    ("RUNNER", "DS_CR"),
    ("RUNNER", "DS_GTR"),
    ("RUNNER", "DS_HAN"),
    ("DS_CR", "AN_CR"),
    ("DS_GTR", "AN_GTR"),
    ("DS_HAN", "AN_HAN"),
    ("AN_CR", "CONSOL"),
    ("AN_GTR", "CONSOL"),
    ("AN_HAN", "CONSOL"),
    ("CONSOL", "AGGR"),
    ("AGGR", "OUTPUT")
])

# Render the PNG to /mnt/data
output_path = Path("./mission_radar_pipeline")
dot.render(filename=output_path.name, directory=str(output_path.parent), cleanup=True)

output_path_str = str(output_path.with_suffix(".png"))
output_path_str
