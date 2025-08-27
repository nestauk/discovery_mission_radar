import argparse
import argilla as rg
import pandas as pd
import os
import shutil
from collections import defaultdict
from eval_config import DATASET_CONFIGS, LABELS, USERS

# Parse arguments
parser = argparse.ArgumentParser(
    description="Run Argilla labelling for multiple dataset types"
)
parser.add_argument(
    "dataset_type", 
    choices=list(DATASET_CONFIGS.keys()),
    help="Dataset type to process"
)
parser.add_argument(
    "--sample-size", 
    type=int, 
    default=30,
    help="Number of samples to take from each dataset (default: 30)"
)

args = parser.parse_args()
config = DATASET_CONFIGS[args.dataset_type]

print(f"Setting up Argilla labelling for: {config['name']}")

# Load environment variables
argilla_url = os.environ["ARGILLA_API_URL"]
argilla_api_key = os.environ["ARGILLA_API_KEY"]

# Load and sample data
print("Loading and sampling data...")
accepted_dataset = pd.read_csv(config["accepted_path"])
rejected_dataset = pd.read_csv(config["rejected_path"])

# Take samples from each
accepted_sample = accepted_dataset.sample(
    n=min(args.sample_size, len(accepted_dataset)), random_state=42
)
rejected_sample = rejected_dataset.sample(
    n=min(args.sample_size, len(rejected_dataset)), random_state=42
)

# Add relevance labels
accepted_sample["is_relevant"] = "accepted"
rejected_sample["is_relevant"] = "rejected"

# Combine and shuffle
sampled = pd.concat([accepted_sample, rejected_sample], ignore_index=True)
sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Build text field based on dataset type
def build_text_field(row):
    title = str(row.get(config["title_field"], "")).strip()
    text = str(row.get(config["text_field"], "")).strip()
    return title + "\n\n" + text

sampled["text"] = sampled.apply(build_text_field, axis=1)

print(f"Sampled {len(sampled)} records")
print("Setting up Argilla labelling task...")

if __name__ == "__main__":
    client = rg.Argilla(
        api_url=argilla_url,
        api_key=argilla_api_key
    )

    settings = rg.Settings(
        guidelines=config["guidelines"],
        fields=[
            rg.TextField(name="text"),
        ],
        questions=[
            rg.LabelQuestion(
                name="relevance",
                labels=LABELS,
                title=config["question_title"],
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="is_relevant", title="Is Relevant"),
            rg.TermsMetadataProperty(name="source", title="Source"),
            rg.TermsMetadataProperty(name="assignee", title="Assignee"),
            rg.TermsMetadataProperty(name="subset", title="Subset"), 
        ],
    )

    # Handle existing dataset
    dataset_name = config["argilla_dataset_name"]
    dataset = client.datasets(name=dataset_name)

    if dataset is not None:
        print(f"Deleting and recreating dataset {dataset_name}")
        delete_continue = input("Continue? (y/n): ")
        if delete_continue=='y':
            dataset.delete()
        else:
            print("Not deleting dataset, so rest of flow won't work")

    dataset = rg.Dataset(
        name=dataset_name,
        settings=settings,
    )
    dataset.create()
    dataset = client.datasets(name=dataset_name)

    # Get users
    list_of_desired_users = USERS
    users = client.users.list()
    filtered_users = [u for u in users if u.username in list_of_desired_users]
    print(f"Found {len(filtered_users)} users: {[u.username for u in filtered_users]}")

    # 50% overlap, 50% disjoint
    overlap_n = len(sampled) // 3
    overlap_df = sampled.iloc[:overlap_n].reset_index(drop=True)
    disjoint_df = sampled.iloc[overlap_n:].reset_index(drop=True)

    # Log overlap records (all annotators will see them)
    overlap_records = [
        rg.Record(
            id=f"ov-{i}",
            fields={"text": row["text"]},
            metadata={
                "source": config["source"],
                "assignee": "all",
                "subset": "overlap",
                "is_relevant": row["is_relevant"],
            },
        )
        for i, row in overlap_df.iterrows()
    ]
    dataset.records.log(overlap_records)
    print(f"✓ Logged {len(overlap_records)} overlap records")

    # Assign disjoint records to users
    assignments = defaultdict(list)
    n = len(filtered_users)

    for i, row in enumerate(disjoint_df.reset_index().to_dict(orient="records")):
        user = filtered_users[i % n]
        assignments[user.username].append(
            rg.Record(
                id=f"{user.username}-{row['index']}",
                fields={"text": row["text"]},
                metadata={
                    "source": config["source"],
                    "assignee": user.username,
                    "subset": "disjoint",
                    "is_relevant": row["is_relevant"],
                },
            )
        )

    # Log each user's subset as its own batch
    for username, records in assignments.items():
        dataset.records.log(records)
        print(f"✓ Logged {len(records)} disjoint records for {username}")

    dataset.update()
    print(f"Argilla labelling task running on {argilla_url}")
    print(f"Dataset: {dataset_name}")

    while True:
        save_now = input("When you have finished labelling save labels by typing 'S' or 's', if you aren't ready there is no need to type anything:")

        if save_now in ['S', 's']:
            # Download the labelled data
            dataset_labelled = client.datasets(name=dataset_name)
            
            output_dir = config["output_dir"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                overwrite_file_name = os.path.join(output_dir, 'records.json')
                if os.path.isfile(overwrite_file_name):
                    print("Replacing existing download, old version will be kept in records_previous.json")
                    os.rename(overwrite_file_name, os.path.join(output_dir, "records_previous.json"))
                argilla_folder_name = os.path.join(output_dir, '.argilla/')
                if os.path.isdir(argilla_folder_name):
                    shutil.rmtree(argilla_folder_name)

            dataset_labelled.to_disk(path=output_dir)
            print(f"Dataset saved to {output_dir}")
            break
    
        elif save_now == '':  # If the user just presses Enter
            print("Continuing labelling...")
            continue  # Go back to the beginning of the while loop
        else:
            print("Invalid input. Please type 'S' or 's' to save, or press Enter to continue labelling.")