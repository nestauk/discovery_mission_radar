import os

# Base paths
BASE_DATA_PATH = "/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF"
BASE_OUTPUT_PATH = "/Users/aidan.kelly/nesta/discovery/discovery_mission_radar/discovery_mission_radar/data/2025_Q2_ASF"

# Dataset configurations
DATASET_CONFIGS = {
    "gtr_lch_gf": {
        "name": "GtR Low Carbon Heating - Green Finance",
        "argilla_dataset_name": "gtr_lch_gf",
        "source": "gtr",
        "accepted_path": f"{BASE_DATA_PATH}/green_finance/gtr/low_carbon_heating/filtered/relevant_low_carbon_heating_llm_filtered.csv",
        "rejected_path": f"{BASE_DATA_PATH}/green_finance/gtr/low_carbon_heating/filtered/relevant_low_carbon_heating_rejected.csv",
        "output_dir": f"{BASE_OUTPUT_PATH}/green_finance/gtr/low_carbon_heating/argilla_outputs/gtr_lch_gf/",
        "title_field": "title",
        "text_field": "abstractText_x",
        "guidelines": (
            "Task: Decide if the item is relevant to *Low-carbon Heating — Green Finance*.\n\n"
            "Labels:\n"
            "- Relevant: Finance mechanisms tied explicitly to HEATING (e.g., heat pumps, district heat, boilers) — loans, green mortgages, guarantees, HaaS.\n"
            "- Borderline: Energy retrofit finance where heating is present but not explicit/salient.\n"
            "- Not relevant: Generic green finance without heating; finance for unrelated sectors (mobility, power-only).\n"
            "- Unclear: Insufficient info.\n"
        ),
        "question_title": "Is this relevant to Low‑carbon Heating — Green Finance?"
    },
    
    "cb_lch_gf": {
        "name": "CrunchBase Low Carbon Heating - Green Finance", 
        "argilla_dataset_name": "cb_lch_gf",
        "source": "crunchbase",
        "accepted_path": f"{BASE_DATA_PATH}/green_finance/cb/low_carbon_heating/filtered/relevant_low_carbon_heating_llm_filtered.csv",
        "rejected_path": f"{BASE_DATA_PATH}/green_finance/cb/low_carbon_heating/filtered/relevant_low_carbon_heating_rejected.csv",
        "output_dir": f"{BASE_OUTPUT_PATH}/green_finance/cb/low_carbon_heating/argilla_outputs/cb_lch_gf/",
        "title_field": "short_description",
        "text_field": "text",
        "guidelines": (
            "Task: Decide if the item is relevant to *Low-carbon Heating — Green Finance*.\n\n"
            "Labels:\n"
            "- Relevant: Finance mechanisms tied explicitly to HEATING (e.g., heat pumps, district heat, boilers) — loans, green mortgages, guarantees, HaaS.\n"
            "- Borderline: Energy retrofit finance where heating is present but not explicit/salient.\n"
            "- Not relevant: Generic green finance without heating; finance for unrelated sectors (mobility, power-only).\n"
            "- Unclear: Insufficient info.\n"
        ),
        "question_title": "Is this relevant to Low‑carbon Heating — Green Finance?"
    },
    
    "gtr_lch_opt": {
        "name": "GtR Low Carbon Heating - Optimisation",
        "argilla_dataset_name": "gtr_lch_opt", 
        "source": "gtr",
        "accepted_path": f"{BASE_DATA_PATH}/lch_optimisation/gtr/filtered/relevant_lch_optimisation_llm_filtered.csv",
        "rejected_path": f"{BASE_DATA_PATH}/lch_optimisation/gtr/filtered/relevant_lch_optimisation_rejected.csv",
        "output_dir": f"{BASE_OUTPUT_PATH}/lch_optimisation/gtr/argilla_outputs/gtr_lch_opt/",
        "title_field": "title",
        "text_field": "abstractText_x",
        "guidelines": (
            "Task: Decide if the item is relevant to Low‑carbon Heating — Optimisation (Upfront cost and Performance).\n\n"
            "Labels:\n"
            "- Relevant: EITHER (a) Upfront cost/deployment optimisation for residential low‑carbon heating (modular/prefab, standardisation, installation automation, streamlined workflows/commissioning, manufacturing scale/lean, supply‑chain/bulk procurement, business models/financing that lower upfront outlay); OR (b) Performance/operational optimisation that measurably improves efficiency, COP, comfort, or running costs via heating‑focused smart controls, optimisation algorithms, tariffs/ToU/flex for heating, sensors/IoT, diagnostics/fault detection, predictive maintenance, system tuning/balancing, digital twins/analytics, thermal storage optimisation — for heat pumps, heat networks/district heating, solar thermal, thermal storage, etc.\n"
            "- Borderline: General home/building optimisation or grid‑flex products where heating is present but not central; analytics/monitoring without clear optimisation action; finance that affects ongoing costs only (not upfront) unless tied to performance optimisation of heating.\n"
            "- Not relevant: No low‑carbon heating, or no clear link to upfront or performance optimisation (power‑only, mobility, generic green tech/AI/IoT, unrelated finance, pure building ops unrelated to heating).\n"
            "- Unclear: Insufficient detail to judge.\n"
        ),
        "question_title": "Is this relevant to Low‑carbon Heating — Optimisation?"
    }
}

# Common configuration
LABELS = ["relevant", "borderline", "not relevant", "unclear"]
USERS = ["karlis", "will", "aidan"]