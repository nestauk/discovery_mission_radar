#!/usr/bin/env python3
"""
Script to analyze combined Hansard speeches across all topics for mentions of costs/prices.
Creates both individual topic analysis and combined deduplication analysis.
Produces two versions: all LLM-verified speeches vs. only cost-reduction mentions.
"""

import datetime
import pandas as pd
import re
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils import (
    analysis,
    charts,
    google,
)
from discovery_mission_radar import PROJECT_DIR
from discovery_mission_radar.notebooks.MR_2025_Q2_ASF.hansard.cost_mentions import (
    TOPIC_KEYWORDS,
)

OUTPUT_DIR = PROJECT_DIR / "data/2025_Q2_ASF/hansard"
CHARTS_DIR = OUTPUT_DIR / "charts"
PRESENT_QUARTER = "2025-Q2"
UPLOAD_TO_GSHEET = True


def impute_missing_quarters(df, date_col="quarter", value_col="speeches", min_quarter=None, max_quarter=None):
    # Convert quarter strings to Period type for easy handling
    df[date_col] = pd.PeriodIndex(df[date_col], freq='Q')
    
    # Determine min and max quarters
    if min_quarter is None:
        min_quarter = df[date_col].min()
    if max_quarter is None:
        max_quarter = df[date_col].max()
    
    # Generate full range of quarters
    full_range = pd.period_range(start=min_quarter, end=max_quarter, freq='Q')
    
    # Create a complete DataFrame with all quarters
    full_df = pd.DataFrame({date_col: full_range})
    
    # Merge with the original DataFrame
    df = full_df.merge(df, on=date_col, how='left')
    
    # Fill missing values in # speeches column with 0
    df[value_col] = df[value_col].fillna(0).astype(int)
    
    # Convert period back to string if necessary
    df[date_col] = df[date_col].astype(str)
    # Add a dash between year and quarter
    df[date_col] = df[date_col].str.replace("Q", "-Q")
    return df

def impute_missing_years(df, year_col="year", value_col="speeches", min_year=None, max_year=None):
    # Ensure year column is integer
    df[year_col] = df[year_col].astype(int)

    # Determine min and max years
    if min_year is None:
        min_year = df[year_col].min()
    if max_year is None:
        max_year = df[year_col].max()

    # Generate a complete range of years
    full_range = pd.DataFrame({year_col: range(min_year, max_year + 1)})

    # Merge with the existing DataFrame
    df = full_range.merge(df, on=year_col, how='left')

    # Fill missing values in # speeches column with 0
    df[value_col] = df[value_col].fillna(0).astype(int)
    return df

def process_topic_data(topic_keyword: str, all_speeches_df: pd.DataFrame, analysis_type: str = "all"):
    """
    Process data for a single topic.
    
    Parameters:
    - topic_keyword: The topic to analyze
    - all_speeches_df: Combined DataFrame with all speeches
    - analysis_type: "all" for all verified speeches, "reduction" for cost-reduction only
    """
    # Filter for this topic
    topic_df = all_speeches_df.query("topic_keyword == @topic_keyword").copy()
    
    # Further filter based on analysis type
    if analysis_type == "reduction":
        topic_df = topic_df.query("mentions_reducing_costs_or_prices == 'yes'")
    
    # Calculate yearly time series
    ts_df = (
        topic_df
        .query("date >= '2014-01-01'")
        .groupby("year")
        .agg(speeches=("speech_id", "count"))
        .reset_index()
        .pipe(impute_missing_years, min_year=2014, max_year=2025)
        .assign(topic_keyword=topic_keyword)
    )
    
    # Calculate quarterly time series
    ts_quarterly_df = (
        topic_df
        .query("date >= '2020-01-01'")
        .groupby("quarter")
        .agg(speeches=("speech_id", "count"))
        .reset_index()
        .pipe(impute_missing_quarters, min_quarter="2020Q1", max_quarter="2025Q2")
        .assign(topic_keyword=topic_keyword)
    )
    
    # Calculate growth and magnitude statistics
    growth_magnitude_df = (
        analysis.magnitude_growth(ts_df.drop(columns=['topic_keyword']), year_start=2020, year_end=2024)
        .assign(topic_keyword=topic_keyword)
        .reset_index()
        .rename(columns={'index': 'variable'})
    )
    
    # Create charts for this topic
    create_charts(ts_df, ts_quarterly_df, topic_keyword, analysis_type)
    
    return ts_df, ts_quarterly_df, growth_magnitude_df, topic_df

def load_and_combine_data():
    """Load and combine all topic data from LLM output files."""
    print("Loading Hansard data...")
    HansardData = policy_update.HansardData()
    
    all_topics_data = []
    
    for topic_keyword in TOPIC_KEYWORDS:
        print(f"Processing {topic_keyword}...")
        
        # Load the cost mentions data
        try:
            keyword_data = pd.read_csv(OUTPUT_DIR / f"cost_mentions_{topic_keyword}.csv")
            print(f"  Loaded {len(keyword_data)} cost mentions for {topic_keyword}")
        except FileNotFoundError:
            print(f"  Warning: cost_mentions_{topic_keyword}.csv not found, skipping...")
            continue
        
        # Load the LLM verification results
        try:
            llm_results = pd.read_json(OUTPUT_DIR / f"output_{topic_keyword}.jsonl", lines=True)
            print(f"  Loaded {len(llm_results)} LLM results for {topic_keyword}")
        except FileNotFoundError:
            print(f"  Warning: output_{topic_keyword}.jsonl not found, skipping...")
            continue
        
        # Merge keyword data with speech details
        keyword_data_with_speech = (
            keyword_data
            .merge(HansardData.debates_df[["speech_id", "speech"]], on="speech_id", how="left")
        )
        
        # Create aggregated sentences per speech
        hit_sentences = (
            keyword_data_with_speech
            .groupby("speech_id")
            .agg(
                marked_sentence=("marked_sentence", "; ".join),
                speech=("speech", lambda x: x.iloc[0]),
            )
            .reset_index()
        )
        
        # Merge with LLM results
        hit_sentences_checked = (
            hit_sentences
            .merge(llm_results, left_on="speech_id", right_on="id", how="left")
            .merge(
                keyword_data.drop_duplicates(subset="speech_id")[
                    ['speech_id', 'date', 'quarter', 'speaker', 'major_heading', 'minor_heading']
                ], 
                on="speech_id", 
                how="left"
            )
            .assign(speech_text_norm=lambda df: df.speech.apply(lambda x: re.sub(r"\s+", " ", x)))
            .drop_duplicates(["speaker", "date", "speech_text_norm"])
            .assign(topic_keyword=topic_keyword)
            .assign(year=lambda df: df.date.apply(lambda x: x.split("-")[0]))
        )
        
        # Filter for verified speeches only
        verified_col = f"is_about_costs_or_prices"
        hit_sentences_checked = hit_sentences_checked.query(f"{verified_col} == 'yes'")
        
        print(f"  Found {len(hit_sentences_checked)} verified speeches for {topic_keyword}")
        
        all_topics_data.append(hit_sentences_checked)
    
    # Combine all topics
    if all_topics_data:
        combined_df = pd.concat(all_topics_data, ignore_index=True)
        print(f"\nCombined total: {len(combined_df)} speeches across all topics")
        return combined_df
    else:
        print("No data found for any topics!")
        return pd.DataFrame()

def create_combined_analysis(all_speeches_df: pd.DataFrame, analysis_type: str = "all"):
    """
    Create combined analysis across all topics with deduplication.
    
    Parameters:
    - all_speeches_df: Combined DataFrame with all speeches
    - analysis_type: "all" for all verified speeches, "reduction" for cost-reduction only
    """
    # Apply filtering based on analysis type
    if analysis_type == "reduction":
        # Filter for speeches mentioning cost reduction
        filtered_df = all_speeches_df.query("mentions_reducing_costs_or_prices == 'yes'").copy()
    else:
        filtered_df = all_speeches_df.copy()
    
    # Deduplicate by speech_id (keep first occurrence)
    deduplicated_df = filtered_df.drop_duplicates(subset="speech_id").copy()
    
    print(f"Combined analysis ({analysis_type}): {len(filtered_df)} total -> {len(deduplicated_df)} after deduplication")
    
    # Calculate yearly time series
    ts_df = (
        deduplicated_df
        .query("date >= '2014-01-01'")
        .groupby("year")
        .agg(speeches=("speech_id", "count"))
        .reset_index()
        .pipe(impute_missing_years, min_year=2014, max_year=2025)
        .assign(topic_keyword="combined")
    )
    
    # Calculate quarterly time series
    ts_quarterly_df = (
        deduplicated_df
        .query("date >= '2020-01-01'")
        .groupby("quarter")
        .agg(speeches=("speech_id", "count"))
        .reset_index()
        .pipe(impute_missing_quarters, min_quarter="2020Q1", max_quarter="2025Q2")
        .assign(topic_keyword="combined")
    )
    
    # Calculate growth and magnitude statistics
    growth_magnitude_df = (
        analysis.magnitude_growth(ts_df.drop(columns=['topic_keyword']), year_start=2020, year_end=2024)
        .assign(topic_keyword="combined")
        .reset_index()
        .rename(columns={'index': 'variable'})
    )
    
    # Create charts for combined analysis
    create_charts(ts_df, ts_quarterly_df, "combined", analysis_type)
    
    return ts_df, ts_quarterly_df, growth_magnitude_df, deduplicated_df

def pivot_time_series_to_wide(ts_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Convert time series data from long format to wide format.
    
    Args:
        ts_df: DataFrame with columns [time_col, 'speeches', 'topic_keyword']
        time_col: Name of the time column ('year' or 'quarter')
    
    Returns:
        DataFrame with time_col as rows, topic_keyword as columns, speeches as values
    """
    # Pivot the data
    wide_df = ts_df.pivot(index=time_col, columns='topic_keyword', values='speeches')
    
    # Fill NaN values with 0 (in case some topic/time combinations are missing)
    wide_df = wide_df.fillna(0)
    
    # Reset index to make the time column a regular column
    wide_df = wide_df.reset_index()
    
    # Clean up column names (remove the 'topic_keyword' name from columns)
    wide_df.columns.name = None
    
    return wide_df

def create_charts(ts_yearly_df: pd.DataFrame, ts_quarterly_df: pd.DataFrame, category_name: str, analysis_type: str = "all"):
    """
    Create and save charts for time series data.
    
    Args:
        ts_yearly_df: DataFrame with yearly time series data
        ts_quarterly_df: DataFrame with quarterly time series data
        category_name: Name of the category for chart titles and filenames
        analysis_type: "all" or "reduction" to distinguish chart types
    """
    import os
    
    # Ensure charts directory exists
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    # Chart configuration
    scale_factor = 2
    prefix = f"{category_name}_{analysis_type}"
    
    # Create yearly chart
    yearly_chart_title = f"Number of speeches for {category_name} ({analysis_type})"
    fig_yearly = charts.ts_bar(
        ts_yearly_df,
        variable="speeches",
        variable_title="Number of speeches",
        time_column="year",
    )
    fig_yearly = charts.configure_plots(fig_yearly, chart_title=yearly_chart_title)
    yearly_filename = CHARTS_DIR / f"{prefix}_yearly_speeches.png"
    fig_yearly.save(str(yearly_filename), scale_factor=scale_factor)
    print(f"  Saved yearly chart: {yearly_filename}")
    
    # Create quarterly chart
    quarterly_chart_title = f"Quarterly speeches for {category_name} ({analysis_type})"
    fig_quarterly = charts.ts_bar(
        ts_quarterly_df,
        variable="speeches", 
        variable_title="Number of speeches",
        time_column="quarter",
    )
    fig_quarterly = charts.configure_plots(fig_quarterly, chart_title=quarterly_chart_title)
    quarterly_filename = CHARTS_DIR / f"{prefix}_quarterly_speeches.png"
    fig_quarterly.save(str(quarterly_filename), scale_factor=scale_factor)
    print(f"  Saved quarterly chart: {quarterly_filename}")

def prepare_output_data(speeches_df: pd.DataFrame):
    """Prepare speeches data for output with consistent columns."""
    # Define output columns
    output_cols = [
        "speech_id", "date", "year", "quarter", "major_heading", "minor_heading", 
        "speaker", "marked_sentence", "topic_keyword",
        "is_about_costs_or_prices", "mentions_reducing_costs_or_prices", "reduction_mechanism",
        "model", "timestamp", "temperature", "speech"
    ]
    
    # Select only columns that exist
    available_cols = [col for col in output_cols if col in speeches_df.columns]
    
    return (
        speeches_df[available_cols]
        .copy()
        .assign(speech=lambda df: df.speech.str[:3000] if 'speech' in df.columns else "")
    )

if __name__ == "__main__":
    # Load and combine all data
    all_speeches_df = load_and_combine_data()
    
    if all_speeches_df.empty:
        print("No data to analyze. Please run the LLM check script first.")
        exit(1)
    
    # Initialize containers for results
    all_ts_yearly = []
    all_ts_quarterly = []
    all_stats = []
    all_speeches_list = []
    
    reduction_ts_yearly = []
    reduction_ts_quarterly = []
    reduction_stats = []
    reduction_speeches_list = []
    
    print("\n" + "="*50)
    print("PROCESSING INDIVIDUAL TOPICS")
    print("="*50)
    
    # Process each topic individually
    for topic_keyword in TOPIC_KEYWORDS:
        if topic_keyword not in all_speeches_df.topic_keyword.unique():
            print(f"Skipping {topic_keyword} - no data available")
            continue
            
        print(f"\nProcessing {topic_keyword}...")
        
        # All verified speeches
        ts_df, ts_quarterly_df, growth_df, topic_df = process_topic_data(
            topic_keyword, all_speeches_df, "all"
        )
        all_ts_yearly.append(ts_df)
        all_ts_quarterly.append(ts_quarterly_df)
        all_stats.append(growth_df)
        all_speeches_list.append(topic_df)
        print(f"  All verified: {len(topic_df)} speeches")
        
        # Cost reduction only
        ts_df_red, ts_quarterly_df_red, growth_df_red, topic_df_red = process_topic_data(
            topic_keyword, all_speeches_df, "reduction"
        )
        reduction_ts_yearly.append(ts_df_red)
        reduction_ts_quarterly.append(ts_quarterly_df_red)
        reduction_stats.append(growth_df_red)
        reduction_speeches_list.append(topic_df_red)
        print(f"  Cost reduction: {len(topic_df_red)} speeches")
    
    print("\n" + "="*50)
    print("PROCESSING COMBINED ANALYSIS")
    print("="*50)
    
    # Combined analysis - All verified
    combined_ts, combined_ts_q, combined_stats, combined_speeches = create_combined_analysis(
        all_speeches_df, "all"
    )
    all_ts_yearly.append(combined_ts)
    all_ts_quarterly.append(combined_ts_q)
    all_stats.append(combined_stats)
    
    # Combined analysis - Cost reduction only  
    combined_ts_red, combined_ts_q_red, combined_stats_red, combined_speeches_red = create_combined_analysis(
        all_speeches_df, "reduction"
    )
    reduction_ts_yearly.append(combined_ts_red)
    reduction_ts_quarterly.append(combined_ts_q_red)
    reduction_stats.append(combined_stats_red)
    
    print("\n" + "="*50)
    print("PREPARING OUTPUT DATA")
    print("="*50)
    
    # Combine all results
    final_all_ts_yearly = pd.concat(all_ts_yearly, ignore_index=True)
    final_all_ts_quarterly = pd.concat(all_ts_quarterly, ignore_index=True)
    final_all_stats = pd.concat(all_stats, ignore_index=True)
    final_all_speeches = pd.concat(all_speeches_list, ignore_index=True)
    
    final_reduction_ts_yearly = pd.concat(reduction_ts_yearly, ignore_index=True)
    final_reduction_ts_quarterly = pd.concat(reduction_ts_quarterly, ignore_index=True)
    final_reduction_stats = pd.concat(reduction_stats, ignore_index=True)
    final_reduction_speeches = pd.concat(reduction_speeches_list, ignore_index=True)
    
    # Prepare speeches data for output
    prepared_all_speeches = prepare_output_data(final_all_speeches)
    prepared_reduction_speeches = prepare_output_data(final_reduction_speeches)
    
    print(f"Final results:")
    print(f"  All verified speeches: {len(prepared_all_speeches)}")
    print(f"  Cost reduction speeches: {len(prepared_reduction_speeches)}")
    
    print("\n" + "="*50)
    print("SAVING CSV FILES")
    print("="*50)
    
    # Save CSV files for other scripts to use
    all_speeches_path = OUTPUT_DIR / "all_verified_speeches.csv"
    prepared_all_speeches.to_csv(all_speeches_path, index=False)
    print(f"Saved all verified speeches to {all_speeches_path}")
    
    reduction_speeches_path = OUTPUT_DIR / "reduction_speeches.csv"
    prepared_reduction_speeches.to_csv(reduction_speeches_path, index=False)
    print(f"Saved cost reduction speeches to {reduction_speeches_path}")
    
    print("\n" + "="*50)
    print("CONVERTING TIME SERIES TO WIDE FORMAT")
    print("="*50)
    
    # Convert time series data from long to wide format for easier visualization
    print("Converting yearly time series to wide format...")
    final_all_ts_yearly_wide = pivot_time_series_to_wide(final_all_ts_yearly, 'year')
    final_reduction_ts_yearly_wide = pivot_time_series_to_wide(final_reduction_ts_yearly, 'year')
    
    print("Converting quarterly time series to wide format...")
    final_all_ts_quarterly_wide = pivot_time_series_to_wide(final_all_ts_quarterly, 'quarter')
    final_reduction_ts_quarterly_wide = pivot_time_series_to_wide(final_reduction_ts_quarterly, 'quarter')
    
    print(f"Wide format yearly data shape: {final_all_ts_yearly_wide.shape}")
    print(f"Wide format quarterly data shape: {final_all_ts_quarterly_wide.shape}")
    
    if UPLOAD_TO_GSHEET:
        print("\n" + "="*50)
        print("UPLOADING TO GOOGLE SHEETS")
        print("="*50)
        
        # Upload to Google Sheets
        sheet_id = "1pTZLEbq1MxDOv8Y0JstkF9_kS8RAs6CoJQygiC8huZM"
        
        # All verified speeches version
        print("Uploading all verified speeches data...")
        google.upload_data_to_gsheet(sheet_id, {
            "all_speeches": prepared_all_speeches,
            "all_stats": final_all_stats,
            "all_chart_yearly": final_all_ts_yearly_wide,
            "all_chart_quarterly": final_all_ts_quarterly_wide,
        })
        
        # Cost reduction only version
        print("Uploading cost reduction data...")
        google.upload_data_to_gsheet(sheet_id, {
            "reduction_speeches": prepared_reduction_speeches,
            "reduction_stats": final_reduction_stats,
            "reduction_chart_yearly": final_reduction_ts_yearly_wide,
            "reduction_chart_quarterly": final_reduction_ts_quarterly_wide,
        })
        
        # Format sheets
        print("Formatting sheets...")
        for sheet_name in ["all_speeches", "reduction_speeches"]:
            google.format_gsheet(sheet_id, sheet_name, freeze_cols=2)
        
        for sheet_name in ["all_stats", "all_chart_yearly", "all_chart_quarterly", 
                        "reduction_stats", "reduction_chart_yearly", "reduction_chart_quarterly"]:
            google.format_gsheet(sheet_id, sheet_name, freeze_cols=0)
        
        print("\nAnalysis complete! Results uploaded to Google Sheets.")
        print(f"Sheet ID: {sheet_id}")