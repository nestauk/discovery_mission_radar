import pandas as pd

from discovery_utils.synthesis.policy import policy_update
from discovery_utils.utils.llm import batch_check
from discovery_mission_radar import PROJECT_DIR
from discovery_mission_radar.notebooks.MR_2025_Q2_ASF.hansard.cost_mentions import (
    TOPIC_KEYWORDS,
)

OUTPUT_DIR = PROJECT_DIR / "data/2025_Q2_ASF/hansard"

FIELDS = [
    {"name": "is_about_costs_or_prices", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},
    {"name": "mentions_reducing_costs_or_prices", "type": "str", "description": "A one-word answer: 'yes' or 'no'."},
    {"name": "reduction_mechanism", "type": "str", "description": "Proposed mechanism to reduce prices or costs (1-3 word answer); if not proposed, answer 'n/a'"},
]


if __name__ == "__main__":
    print("Loading Hansard data...")
    HansardData = policy_update.HansardData()
    
    for topic_keyword in TOPIC_KEYWORDS:

        keyword_data = (
            pd.read_csv(OUTPUT_DIR / f"cost_mentions_{topic_keyword}.csv")
            .merge(HansardData.debates_df[["speech_id", "speech"]], on="speech_id", how="left")
        )

        hit_sentences = (
            keyword_data
            .groupby("speech_id")
            .agg(
                marked_sentence = ("marked_sentence", "; ".join),
                speech = ("speech", lambda x: x.iloc[0]),
            )
            .reset_index()
        )    
        list_of_speeches = []
        for index, row in hit_sentences.iterrows():
            list_of_speeches.append(f"Relevant sentences: {row.marked_sentence}\n\nFull speech: {row.speech}\n\n")
        
        check_data = dict(zip(hit_sentences.speech_id, list_of_speeches))            

        system_message = f"""
            Determine whether this speech (1) actually is about {topic_keyword} costs or price, and 
            (2) explicitly mentions the need to REDUCE the {topic_keyword} cost or price.
            """

        processor = batch_check.LLMProcessor(
            output_path=OUTPUT_DIR / f"output_{topic_keyword}.jsonl",
            system_message=system_message,
            session_name="testing",
            output_fields=FIELDS,
        )

        processor.run(check_data, batch_size=25, sleep_time=0.5)
        