import argilla as rg
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil 

argilla_url  = os.environ["ARGILLA_API_URL"]
argilla_api_key = os.environ["ARGILLA_API_KEY"]

GUIDELINES = (
    "Task: Decide if the item is relevant to *Low-carbon Heating — Green Finance*.\n\n"
    "Labels:\n"
    "- Relevant: Finance mechanisms tied explicitly to HEATING (e.g., heat pumps, district heat, boilers) — loans, green mortgages, guarantees, HaaS.\n"
    "- Borderline: Energy retrofit finance where heating is present but not explicit/salient.\n"
    "- Not relevant: Generic green finance without heating; finance for unrelated sectors (mobility, power-only).\n"
    "- Unclear: Insufficient info.\n"
)

accepted_dataset = pd.read_csv("/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF/green_finance/gtr/low_carbon_heating/filtered/relevant_low_carbon_heating_llm_filtered.csv")
rejected_dataset = pd.read_csv("/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF/green_finance/gtr/low_carbon_heating/filtered/relevant_low_carbon_heating_rejected.csv")


# Take 20 random samples from each (or all if fewer than 20)
accepted_sample = accepted_dataset.sample(n=min(20, len(accepted_dataset)), random_state=42)
rejected_sample = rejected_dataset.sample(n=min(20, len(rejected_dataset)), random_state=42)

# Combine and shuffle so groups are mixed
sampled = pd.concat([accepted_sample, rejected_sample], ignore_index=True)
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Build the text to label: title + abstract
sampled["text"] = sampled.get("title", "").fillna("").astype(str).str.strip() + \
                  "\n\n" + sampled.get("abstractText_x", "").fillna("").astype(str).str.strip()

print("Setting up and running Argilla labelling task...")

if __name__ == "__main__":
    client = rg.Argilla(
        api_url=argilla_url,
        api_key=argilla_api_key
    )

    settings = rg.Settings(
        guidelines=GUIDELINES,
        fields=[
            rg.TextField(
                name="text",
            ),
        ],
        questions=[
            rg.LabelQuestion(
                name="relevance",
                labels=["relevant", "borderline", "not relevant", "unclear"],
                title="Is this relevant to Low‑carbon Heating — Green Finance?",
            ),
        ],
        # NEW: declare metadata properties you want persisted & filterable
        metadata=[
            rg.TermsMetadataProperty(name="is_relevant", title="Is Relevant"),
            rg.TermsMetadataProperty(name="source", title="Source"),
            rg.TermsMetadataProperty(name="assignee", title="Assignee"),
            rg.TermsMetadataProperty(name="subset", title="Subset"), 
        ],
    )

    # An exception will be raised if this dataset already exists, so delete it
    dataset_name = "gtr_lch_gf"
    dataset = client.datasets(name=dataset_name)

    if dataset is not None:
        print(f"Deleting and recreating dataset {dataset_name}")
        delete_continue = input("Continue? (y/n): ")
        if delete_continue=='y':
            dataset.delete()
        else:
            print("Not deleting dataset, so rest of flow won't work")

    dataset = rg.Dataset(
        name="gtr_lch_gf",
        settings=settings,
    )
    # Be explicit about workspace if needed
    dataset.create()
    dataset = client.datasets(name=dataset_name)
    # ——————————————————————————————————————————————————————————————
    # 1. Choose which users you’ll assign to:
    list_of_desired_users = ["karlis", "will", "aidan"]
    users = client.users.list()
    filtered_users = [u for u in users if u.username in list_of_desired_users]
    # 2. Helper to build a single Argilla record with an assignee
    def row_to_record(row, assignee):
        return rg.Record(
            id=str(row.get("id") or f"{assignee}-{row['index']}"),
            fields={"text": row["text"]},
            metadata={"source": "gtr", "assignee": assignee},  # <- stored & filterable
        )
    # dataset.create()
    # records = sampled.to_dict(orient='records') # Needs to be in the format [{"text": "this is text"}, {"text": "this is another text"},]
    # ---- 50% overlap, 50% disjoint ----
    overlap_n = len(sampled) // 2
    overlap_df = sampled.iloc[:overlap_n].reset_index(drop=True)
    disjoint_df = sampled.iloc[overlap_n:].reset_index(drop=True)

    # 1) Log overlap once (all annotators will see it)
    overlap_records = [
        rg.Record(
            id=f"ov-{i}",
            fields={"text": row["text"]},
            metadata={
                "source": "gtr",
                "assignee": "all",
                "subset": "overlap",
                "is_relevant": row["is_relevant"],
            },
        )
        for i, row in overlap_df.iterrows()
    ]
    dataset.records.log(overlap_records)
    print(f"✓ Logged {len(overlap_records)} overlap records")
    # 3. Break your DataFrame into chunks of size = number of users
    from collections import defaultdict
    assignments = defaultdict(list)
    n = len(filtered_users)
    #chunked = [sampled[i : i + n] for i in range(0, len(sampled), n)]
    # 4. Zip each chunk with your users so each user gets one row
    for i, row in enumerate(disjoint_df.reset_index().to_dict(orient="records")):
        user = filtered_users[i % n]
        assignments[user.username].append(
            rg.Record(
                id=f"{user.username}-{row['index']}",
                fields={"text": row["text"]},
                metadata={
                    "source": "gtr",
                    "assignee": user.username,
                    "subset": "disjoint",
                    "is_relevant": row["is_relevant"],
                },
            )
        )
    #for chunk in chunked:
    #    rows = chunk.reset_index().to_dict(orient="records")
    #    for user, row in zip(filtered_users, rows):
    #        assignments[user.username].append(row_to_record(row, user.username))
    # 5. Log each user’s subset as its own batch
    for username, records in assignments.items():
        #ds_user = rg.Dataset(name=dataset_name)
        dataset.records.log(records)
        print(f"✓ Logged {len(records)} disjoint records for {username}")
        print("Logging records to Argilla...")
    #print(records)
    #dataset.records.log(records)
    #dataset.update()
    print(f"Argilla labelling task running on {argilla_url} - click on url to start labelling!")
    
    dataset.update()

    while True:
        save_now = input("When you have finished labelling save labels by typing 'S' or 's', if you aren't ready there is no need to type anything:")

        if save_now in ['S', 's']:

            # download the labelled data
            dataset_labelled = client.datasets(name="gtr_lch_gf")

            # Will download all data not just the status="completed" labelled ones
            # has a "user_id" field

            output_dir = "/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF/green_finance/gtr/low_carbon_heating/argilla_outputs/gtr_lch_gf/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                overwrite_file_name = os.path.join(output_dir, 'records.json')
                if os.path.isfile(overwrite_file_name):
                    print("Replacing existing download, old version will be kept in records_previous.json")
                    os.rename(overwrite_file_name, "/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF/green_finance/gtr/low_carbon_heating/argilla_outputs/gtr_lch_gf/records_previous.json")
                argilla_folder_name = os.path.join(output_dir, '.argilla/')
                if os.path.isdir(argilla_folder_name):
                    shutil.rmtree(argilla_folder_name)

            dataset_labelled.to_disk(path=output_dir)
            print(f"Dataset saved to {output_dir}")
    
        elif save_now == '':  # If the user just presses Enter
            print("Continuing labelling...")
            continue  # Go back to the beginning of the while loop
        else:
            print("Invalid input. Please type 'S' or 's' to save, or press Enter to continue labelling.")