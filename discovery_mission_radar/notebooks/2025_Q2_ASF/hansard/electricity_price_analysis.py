#!/usr/bin/env python3
"""
Script to analyze Hansard speeches for mentions of electricity prices and costs.

This script searches through parliamentary speeches to find sentences that mention
both "electricity" and either "price" or "cost" in the same sentence.
"""

import datetime
import pandas as pd
from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils import keywords as kw


def get_quarter_from_date(date: str) -> int:
    """Return the quarter number from a given YYYY-MM-DD date string."""
    _date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return (_date.month - 1) // 3 + 1


def create_electricity_price_keywords():
    """Create keywords dictionary for electricity price/cost analysis."""
    return {
        "electricity_prices": [
            ["electricity", "price"],
            ["electricity", "cost"],
            ["electricity", "pricing"],
            ["electricity", "costs"],
            ["electricity", "bill"],
            ["electricity", "bills"],
            ["electricity", "tariff"],
            ["electricity", "tariffs"],
            ["electricity", "rate"],
            ["electricity", "rates"],
            ["electricity", "charge"],
            ["electricity", "charges"],
            ["electricity", "expense"],
            ["electricity", "expenses"],
            ["electricity", "payment"],
            ["electricity", "payments"],
            ["electricity", "spend"],
            ["electricity", "spending"],
            ["electricity", "expenditure"],
            ["electricity", "affordability"],
            ["electricity", "affordable"],
            ["electricity", "expensive"],
            ["electricity", "cheap"],
            ["electricity", "cheaper"],
            ["electricity", "expensive"],
            ["electricity", "costly"],
            ["electricity", "value"],
            ["electricity", "worth"],
            ["electricity", "budget"],
            ["electricity", "budgeting"],
            ["electricity", "financial"],
            ["electricity", "finance"],
            ["electricity", "economic"],
            ["electricity", "economy"],
            ["electricity", "money"],
            ["electricity", "pound"],
            ["electricity", "pounds"],
            ["electricity", "penny"],
            ["electricity", "pence"],
            ["electricity", "shilling"],
            ["electricity", "shillings"],
            ["electricity", "dollar"],
            ["electricity", "dollars"],
            ["electricity", "euro"],
            ["electricity", "euros"],
            ["electricity", "cent"],
            ["electricity", "cents"],
            ["electricity", "kwh"],
            ["electricity", "kilowatt"],
            ["electricity", "kilowatts"],
            ["electricity", "megawatt"],
            ["electricity", "megawatts"],
            ["electricity", "gigawatt"],
            ["electricity", "gigawatts"],
            ["electricity", "watt"],
            ["electricity", "watts"],
            ["electricity", "joule"],
            ["electricity", "joules"],
            ["electricity", "btu"],
            ["electricity", "btus"],
            ["electricity", "therm"],
            ["electricity", "therms"],
            ["electricity", "calorie"],
            ["electricity", "calories"],
            ["electricity", "joule"],
            ["electricity", "joules"],
            ["electricity", "watt-hour"],
            ["electricity", "watt-hours"],
            ["electricity", "kilowatt-hour"],
            ["electricity", "kilowatt-hours"],
            ["electricity", "megawatt-hour"],
            ["electricity", "megawatt-hours"],
            ["electricity", "gigawatt-hour"],
            ["electricity", "gigawatt-hours"],
            ["electricity", "terawatt-hour"],
            ["electricity", "terawatt-hours"],
            ["electricity", "twh"],
            ["electricity", "gwh"],
            ["electricity", "mwh"],
            ["electricity", "kwh"],
            ["electricity", "wh"],
            ["electricity", "j"],
            ["electricity", "kj"],
            ["electricity", "mj"],
            ["electricity", "gj"],
            ["electricity", "tj"],
            ["electricity", "pj"],
            ["electricity", "ej"],
            ["electricity", "zj"],
            ["electricity", "yj"],
            ["electricity", "cal"],
            ["electricity", "kcal"],
            ["electricity", "mcal"],
            ["electricity", "gcal"],
            ["electricity", "tcal"],
            ["electricity", "pcal"],
            ["electricity", "ecal"],
            ["electricity", "zcal"],
            ["electricity", "ycal"],
            ["electricity", "btu"],
            ["electricity", "mbtu"],
            ["electricity", "mmbtu"],
            ["electricity", "gbtu"],
            ["electricity", "tbtu"],
            ["electricity", "pbtu"],
            ["electricity", "ebtu"],
            ["electricity", "zbtu"],
            ["electricity", "ybtu"],
            ["electricity", "therm"],
            ["electricity", "mtherm"],
            ["electricity", "gtherm"],
            ["electricity", "ttherm"],
            ["electricity", "ptherm"],
            ["electricity", "etherm"],
            ["electricity", "ztherm"],
            ["electricity", "ytherm"],
        ]
    }


def analyze_electricity_prices():
    """Main function to analyze electricity price mentions in Hansard speeches."""
    
    print("Loading Hansard data...")
    HansardData = policy_update.HansardData()
    
    print("Processing speeches...")
    speeches_df = (
        HansardData.debates_df
        .query("date >= '2014-01-01' & date <= '2025-08-31'")
        .assign(quarter=lambda df: df.date.apply(get_quarter_from_date))
        .assign(quarter=lambda df: df.year.astype(str) + "-Q" + df.quarter.astype(str))
    )
    
    print(f"Loaded {len(speeches_df)} speeches from 2014 to 2025")
    
    # Create keywords dictionary
    keywords_dict = create_electricity_price_keywords()
    
    # Process each speech to find electricity price mentions
    print("Searching for electricity price mentions...")
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
        output_file = "electricity_price_mentions.csv"
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
    results = analyze_electricity_prices() 