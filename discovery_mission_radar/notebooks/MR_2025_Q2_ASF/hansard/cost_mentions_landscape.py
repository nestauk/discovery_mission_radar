#!/usr/bin/env python3
"""
Script to create BERTopic landscape visualization of Hansard cost mentions.
Uses discovery_utils.viz_landscape utilities for topic modeling and visualization.
Loads processed speech data from cost_mentions_analysis.py output.

Usage:
    python cost_mentions_landscape.py              # Use all verified speeches
    python cost_mentions_landscape.py --reduction  # Use only cost reduction speeches
"""

import pandas as pd
import altair as alt
import numpy as np
from discovery_utils.utils.viz_landscape import (
    generate_bertopic,
    generate_reduced_embeddings,
    create_viz_dataframe,
    generate_landscape_keywords,
    chart_keywords,
    scatter_keyword_chart,
)
from discovery_mission_radar import PROJECT_DIR

OUTPUT_DIR = PROJECT_DIR / "data/2025_Q2_ASF/hansard"

# Visualization parameters
MIN_CLUSTER_SIZE = 20  # Smaller than default since we have fewer documents
NR_TOPICS = 8  # Number of topics to generate
N_KEYWORD_CLUSTERS = 30  # Number of keyword clusters for overlay
RANDOM_STATE = 42

def load_hansard_cost_data(use_reduction_only=False):
    """Load processed speech data from the analysis script output."""
    print("Loading processed Hansard speech data...")
    
    # Determine which file to load
    if use_reduction_only:
        file_path = OUTPUT_DIR / "reduction_speeches.csv"
        data_type = "cost reduction"
    else:
        file_path = OUTPUT_DIR / "all_verified_speeches.csv"
        data_type = "all verified"
    
    try:
        hansard_df = pd.read_csv(file_path)
        print(f"Loaded {len(hansard_df)} {data_type} speeches from {file_path}")
        return hansard_df
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        print("Please run cost_mentions_analysis.py first to generate the processed data.")
        return pd.DataFrame()

def prepare_text_data(hansard_df: pd.DataFrame):
    """Prepare text data for BERTopic analysis."""
    print("Preparing text data for analysis...")
    
    # Clean and prepare the text
    text_df = (
        hansard_df
        .assign(
            # Use marked sentence as primary text
            text_clean=lambda df: df.marked_sentence.str.replace(r'\*([^*]+)\*', r'\1', regex=True)  # Remove markdown
        )
        .query("text_clean.str.len() > 10")  # Filter out very short texts
        .drop_duplicates(subset=['text_clean'])  # Remove duplicates
        .reset_index(drop=True)
    )
    
    print(f"Prepared {len(text_df)} unique text entries for analysis")
    return text_df

def create_vectors_dataframe(text_df: pd.DataFrame):
    """Create vectors dataframe with proper embeddings."""
    from sentence_transformers import SentenceTransformer
    
    # Use a sentence transformer to generate embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Generating embeddings for text data...")
    texts = text_df['text_clean'].tolist()
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    # Create the vectors dataframe
    vectors_df = pd.DataFrame({
        'text': texts,
        'vector': [emb.tolist() for emb in embeddings]
    })
    
    return vectors_df, text_df

def create_hansard_viz_dataframe(viz_df: pd.DataFrame, original_df: pd.DataFrame):
    """Create the final visualization dataframe for Hansard data."""
    
    # Map back to original data using text matching
    merged_df = (
        viz_df
        .reset_index()
        .rename(columns={'index': 'text_index'})
        .merge(
            original_df.reset_index().rename(columns={'index': 'orig_index', 'text_clean': 'text'}),
            on='text',
            how='left'
        )
    )
    
    # Add relevant columns for visualization
    final_df = (
        merged_df
        .assign(
            # Create display title
            title=lambda df: df.speaker + " (" + df.date + ")",
            # Short description
            description=lambda df: df.text.str[:200] + "...",
            # Category is the topic keyword
            category=lambda df: df.topic_keyword,
        )
        .fillna({
            'speaker': 'Unknown',
            'topic_keyword': 'Unknown',
            'major_heading': 'Unknown',
            'minor_heading': 'Unknown'
        })
    )
    
    return final_df

def chart_hansard_landscape(
    viz_df: pd.DataFrame,
    width: int = 900,
    height: int = 750,
    opacity: float = 0.6,
) -> alt.Chart:
    """Generate the Hansard landscape visualization."""
    
    # Dropdown menus for filtering
    topic_dropdown = alt.binding_select(
        options=[None] + list(sorted(viz_df["category"].unique())), 
        name="Topic:"
    )
    topic_selection = alt.selection_point(fields=["category"], bind=topic_dropdown, name="SelectTopic")
    
    year_dropdown = alt.binding_select(
        options=[None] + sorted(list(sorted(viz_df["year"].unique().tolist()))), 
        name="Year:"
    )
    year_selection = alt.selection_point(fields=["year"], bind=year_dropdown, name="SelectYear")
    
    # Create the scatter plot
    scatter = (
        alt.Chart(viz_df, width=width, height=height)
        .mark_point(size=40, opacity=opacity)
        .encode(
            x=alt.X("umap_x:Q", axis=None),
            y=alt.Y("umap_y:Q", axis=None),
            tooltip=[
                "title:N",
                "description:N", 
                "category:N",
                "date:N",
                "major_heading:N",
                "minor_heading:N",
                "Name:N",  # Topic name from BERTopic
            ],
            color=alt.Color(
                "Name:N", 
                legend=alt.Legend(title="BERTopic Cluster", labelLimit=300),
                scale=alt.Scale(scheme='category20')
            ),
            opacity=alt.condition(
                topic_selection & year_selection, 
                alt.value(opacity), 
                alt.value(0.0)
            ),
        )
        .add_params(
            topic_selection,
            year_selection,
        )
        .interactive()
    )
    
    return scatter

def generate_hansard_landscape(
    text_df: pd.DataFrame,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    nr_topics: int = NR_TOPICS,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
    n_keyword_clusters: int = N_KEYWORD_CLUSTERS,
):
    """Generate the complete Hansard cost mentions landscape."""
    
    print("Creating vectors dataframe...")
    vectors_df, original_df = create_vectors_dataframe(text_df)
    
    print("Generating BERTopic model...")
    topic_vectors_df, topic_model = generate_bertopic(
        vectors_df,
        min_cluster_size=min_cluster_size,
        nr_topics=nr_topics,
        random_state=random_state,
        verbose=verbose,
    )
    
    print("Generating reduced embeddings for visualization...")
    reduced_embeddings = generate_reduced_embeddings(topic_vectors_df, random_state)
    
    print("Creating visualization dataframe...")
    viz_df = create_viz_dataframe(topic_vectors_df, reduced_embeddings, topic_model)
    
    print("Generating landscape keywords...")
    viz_df, centroids_df = generate_landscape_keywords(
        viz_df, 
        n_keyword_clusters=n_keyword_clusters
    )
    
    print("Preparing final Hansard visualization data...")
    hansard_viz_df = create_hansard_viz_dataframe(viz_df, original_df)
    
    print("Creating visualization...")
    scatter_chart = chart_hansard_landscape(hansard_viz_df)
    keyword_chart = chart_keywords(centroids_df)
    
    final_chart = scatter_keyword_chart(
        scatter_chart=scatter_chart,
        keyword_chart=keyword_chart,
        title="Hansard Cost Mentions Landscape",
        subtitle=[
            "Parliamentary speeches mentioning costs and prices across energy topics",
            "Each point represents a speech sentence. Colors show topic keywords, shapes show timing."
        ]
    )
    
    return final_chart, hansard_viz_df, topic_model

def save_results(chart: alt.Chart, viz_df: pd.DataFrame, topic_model, use_reduction_only: bool = False):
    """Save the visualization and data."""
    
    # Determine file suffix based on analysis type
    suffix = "reduction" if use_reduction_only else "all"
    
    # Save the chart
    chart_path = OUTPUT_DIR / "charts" / f"cost_mentions_landscape_{suffix}.html"
    chart_path.parent.mkdir(exist_ok=True)
    chart.save(str(chart_path))
    print(f"Saved landscape chart to {chart_path}")
    
    # Save the visualization data
    viz_data_path = OUTPUT_DIR / f"cost_mentions_landscape_data_{suffix}.csv"
    viz_df.to_csv(viz_data_path, index=False)
    print(f"Saved visualization data to {viz_data_path}")
    
    # Save topic information
    topic_info = topic_model.get_topic_info()
    topic_info_path = OUTPUT_DIR / f"cost_mentions_topic_info_{suffix}.csv"
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Saved topic information to {topic_info_path}")
    
    print(f"\nGenerated landscape with {len(viz_df)} points and {len(topic_info)} topics")
    print("Top topics:")
    for i, row in topic_info.head(10).iterrows():
        print(f"  Topic {row['Topic']}: {row['Name']} ({row['Count']} speeches)")

if __name__ == "__main__":
    print("="*60)
    print("HANSARD COST MENTIONS LANDSCAPE ANALYSIS")
    print("="*60)
    
    # Allow choosing between all verified speeches or cost reduction only
    import sys
    use_reduction_only = "--reduction" in sys.argv
    
    # Load the data
    hansard_df = load_hansard_cost_data(use_reduction_only=use_reduction_only)
    
    if hansard_df.empty:
        print("No data available. Please run cost_mentions_analysis.py first.")
        exit(1)
    
    # Prepare text data
    text_df = prepare_text_data(hansard_df)
    
    if len(text_df) < MIN_CLUSTER_SIZE:
        print(f"Not enough data for clustering (need at least {MIN_CLUSTER_SIZE}, got {len(text_df)})")
        print("Try reducing MIN_CLUSTER_SIZE or running on more data.")
        exit(1)
    
    # Generate the landscape
    try:
        chart, viz_df, topic_model = generate_hansard_landscape(text_df)
        
        # Save results
        save_results(chart, viz_df, topic_model, use_reduction_only)
        
        data_type = "cost reduction" if use_reduction_only else "all verified"
        suffix = "reduction" if use_reduction_only else "all"
        print("\n" + "="*60)
        print("LANDSCAPE GENERATION COMPLETE!")
        print("="*60)
        print(f"Generated landscape using {data_type} speeches")
        print(f"Open {OUTPUT_DIR}/charts/cost_mentions_landscape_{suffix}.html to view the interactive landscape.")
        
    except Exception as e:
        print(f"Error generating landscape: {e}")
        print("This might be due to insufficient data or API limits.")
        print("Try reducing the number of topics or check your OpenAI API key.")
        raise