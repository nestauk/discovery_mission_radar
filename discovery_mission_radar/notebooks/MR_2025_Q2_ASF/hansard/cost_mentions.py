#!/usr/bin/env python3
"""
Script to analyze Hansard speeches for mentions of energy prices and costs.
"""

import datetime
import pandas as pd
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils import keywords as kw
from discovery_mission_radar import PROJECT_DIR

OUTPUT_DIR = PROJECT_DIR / "data/2025_Q2_ASF/hansard"
START_YEAR = 2014

TOPIC_KEYWORDS = [
    "electricity",
    "energy",
    "heating",
    " gas ",
    "heat pump",
]

COST_KEYWORDS = [
    ["price"],
    ["cost"],
    ["pricing"],
    ["costs"],
    ["bills"],
    ["tariff"],
    ["tariffs"],
    ["rate"],
    ["rates"],
    ["charge"],
    ["charges"],
    ["expense"],
    ["expenses"],
    ["payment"],
    ["payments"],
    ["spend"],
    ["spending"],
    ["expenditure"],
    ["affordability"],
    ["affordable"],
    ["expensive"],
    ["cheap"],
    ["cheaper"],
    ["expensive"],
    ["costly"],
    ["budget"],
    ["budgeting"],
    ["financial"],
    ["finance"],
    ["money"],
    ["pound"],
    ["pounds"],
]

def get_quarter_from_date(date: str) -> int:
    """Return the quarter number from a given YYYY-MM-DD date string."""
    _date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return (_date.month - 1) // 3 + 1


def create_cost_keywords(topic_keyword: str) -> dict:
    """Create keywords dictionary for price/cost analysis."""
    cost_keywords = [[topic_keyword] + cost_keyword for cost_keyword in COST_KEYWORDS]
    return {topic_keyword: cost_keywords}


def get_cost_mentions(topic_keyword: str, speeches_df: pd.DataFrame):
    """Main function to analyze price/cost mentions in Hansard speeches."""

    # Create keywords dictionary
    keywords_dict = create_cost_keywords(topic_keyword)
    
    # Process each speech to find electricity price mentions
    print(f"Searching for {topic_keyword} price mentions...")
    all_hits = []
    
    for idx, row in speeches_df.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx} speeches...")
        
        speech_text = row['speech']
        speech_id = row['speech_id']
        speaker = row['speakername']
        date = row['date']
        quarter = row['quarter']
        major_heading = row['major_heading']
        minor_heading = row['minor_heading']
        
        # Get keyword hits for this speech
        hits_df = kw.get_keyword_hits(speech_text, keywords_dict)
        
        if not hits_df.empty:
            # Add metadata to each hit
            hits_df['speech_id'] = speech_id
            hits_df['speaker'] = speaker
            hits_df['date'] = date
            hits_df['quarter'] = quarter
            hits_df['major_heading'] = major_heading
            hits_df['minor_heading'] = minor_heading
            
            all_hits.append(hits_df)
    
    if all_hits:
        # Combine all hits
        combined_hits = pd.concat(all_hits, ignore_index=True)
        
        print(f"\nFound {len(combined_hits)} sentences mentioning electricity prices/costs")
        
        # Save results
        output_file = OUTPUT_DIR / f"cost_mentions_{topic_keyword}.csv"
        combined_hits.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Display summary statistics
        print("\nSummary by quarter:")
        quarterly_summary = combined_hits.groupby('quarter').size().sort_index()
        print(quarterly_summary)
        
        print("\nTop speakers mentioning electricity prices:")
        speaker_summary = combined_hits.groupby('speaker').size().sort_values(ascending=False).head(10)
        print(speaker_summary)
        
        print("\nSample sentences:")
        sample_sentences = combined_hits[['date', 'speaker', 'sentence']].head(10)
        for _, row in sample_sentences.iterrows():
            print(f"{row['date']} - {row['speaker']}: {row['sentence'][:200]}...")
        
        return combined_hits
    else:
        print("No electricity price mentions found.")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Loading Hansard data...")
    HansardData = policy_update.HansardData()
    
    print("Processing speeches...")
    speeches_df = (
        HansardData.debates_df
        .query(f"date >= '{START_YEAR}-01-01' & date <= '2025-08-31'")
        .assign(quarter=lambda df: df.date.apply(get_quarter_from_date))
        .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
        .assign(speech=lambda df: df.speech.apply(lambda x: x.lower()))
        .reset_index(drop=True)
    )
    
    print(f"Loaded {len(speeches_df)} speeches from 2014 to 2025")
    for topic_keyword in TOPIC_KEYWORDS:
        results = get_cost_mentions(topic_keyword, speeches_df) 
