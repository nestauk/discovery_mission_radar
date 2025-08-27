from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import os, json
from collections import Counter

def map_human_label(lbl: str, mode: str = "strict") -> int | None:
    """
    Map a human-provided relevance label to a binary value for evaluation.

    Args:
        lbl (str): The human label, expected to be one of "relevant", "borderline", "not relevant", or "unclear".
        mode (str): Mapping mode. "strict" maps only "relevant" to 1, all others to 0.
                    "lenient" maps "relevant" and "borderline" to 1, "not relevant" and "unclear" to 0.

    Returns:
        int | None: 1 for relevant, 0 for not relevant, or None if the label is missing or unrecognized.
    """
    if lbl is None: return None
    s = str(lbl).strip().lower()
    if mode == "strict":
        return 1 if s == "relevant" else (0 if s in {"borderline", "not relevant", "unclear"} else None)
    # lenient
    return 1 if s in {"relevant", "borderline"} else (0 if s in {"not relevant", "unclear"} else None)

def map_llm(llm: str) -> int | None:
    """
    Map a language model (LLM) response to a binary value for evaluation.

    Args:
        llm (str): The LLM response, expected to be one of "yes", "no", "true", "false", "1", or "0".

    Returns:
        int | None: 1 for positive responses, 0 for negative responses, or None if the response is missing or unrecognized.
    """
    if llm is None: return None
    s = str(llm).strip().lower()
    if s in {"yes", "true", "1"}: return 1
    if s in {"no", "false", "0"}: return 0
    return None

def _load_records(records_path: str):
    """
    Load records from a JSON or JSONL file.

    Args:
        records_path (str): The file path to the records.

    Returns:
        list: A list of records, or an empty list if loading fails.
    """
    if not os.path.isfile(records_path):
        return []

    with open(records_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # JSONL fallback
        try:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        except Exception:
            return []
    # Argilla may export a JSON string containing a JSON array
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return []
    if isinstance(payload, dict):
        for k in ("records", "items", "data"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
        return []
    if isinstance(payload, list):
        return payload
    return []

def _disjoint_extract_human_label(rec: dict) -> str | None:
    """
    Extract the human relevance label from an Argilla record.

    Args:
        rec (dict): Argilla record.

    Returns:
        str | None: Human label or None.
    """
    # Newer Argilla export: responses is a dict keyed by question name
    resp = rec.get("responses")
    if isinstance(resp, dict):
        answers = resp.get("relevance")
        if isinstance(answers, list) and answers:
            val = answers[-1].get("value")
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            return val
    return None

def _overlap_extract_human_label(rec: dict) -> str | None:
    """
    Extract the human relevance label from an Argilla record.

    Args:
        rec (dict): Argilla record.

    Returns:
        str | None: Human label or None.
    """
    # Newer Argilla export: responses is a dict keyed by question name
    labels: list[str] = []
    resp = rec.get("responses")
    if isinstance(resp, dict):
        answers = resp.get("relevance")
        if isinstance(answers, list):
            for ans in answers:
                val = ans.get("value")
                labels.append(str(val))
    return labels

def _majority_label_winner(labels: list[str]) -> str | None:
    """
    Determine the majority label from a list of labels.

    Args:
        labels (list[str]): List of labels.

    Returns:
        str | None: The majority label or None if no labels are present.
    """
    if not labels:
        return None

    allowed = {"relevant", "borderline", "not relevant", "unclear"}
    norm = [str(x).strip().lower() for x in labels if str(x).strip()]
    norm = [x for x in norm if x in allowed]
    if not norm:
        return None

    counts = Counter(norm)
    top_two = counts.most_common(2)

    if len(top_two) == 1:
        return top_two[0][0]

    (lbl1, c1), (lbl2, c2) = top_two
    if c1 == c2:
        return "unclear"
    return lbl1

def print_binary_classification_metrics(y_true: list[int], y_pred: list[int], mode: str, subset: str):
    """
    Print confusion matrix, precision, recall, F1, and classification report for binary labels.
    """
    if not y_true:
        print("⚠️  No comparable labeled records found.")
        return
    print(f"\n=== Set ({subset.upper()}) ===")
    print(f"\n=== Metrics ({mode.upper()}) ===")
    print("Confusion matrix [tn fp; fn tp]:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["not relevant", "relevant"], zero_division=0))

def disjoint_compute_metrics_from_export(export_dir: str, mode: str = "strict"):
    """
    Compute and print classification metrics comparing human and LLM relevance labels from Argilla export.

    Loads records from 'records.json' in the specified export directory, extracts human and LLM labels,
    maps them to binary values, and computes confusion matrix, precision, recall, F1 score, and a full
    classification report. Supports both "strict" and "lenient" mapping modes for human labels.

    Args:
        export_dir (str): Path to the directory containing Argilla export files.
        mode (str): Label mapping mode, either "strict" or "lenient".

    Returns:
        None
    """
    records_path = os.path.join(export_dir, "records.json")
    if not os.path.isfile(records_path):
        print(f"⚠️  No records.json at {records_path}")
        return
    records = _load_records(records_path)
    if not records:
        print("⚠️  No records found in export.")
        return
    y_true, y_pred = [], []
    for r in records:
        human = _disjoint_extract_human_label(r)
        meta = r.get("metadata") or {}
        if meta.get("subset") == "disjoint":
            # Prefer 'is_relevant' as per your export, but accept common fallbacks
            llm = meta.get("is_relevant")
            t, p = map_human_label(human, mode), map_llm(llm)
            if t is None or p is None:
                continue
            y_true.append(t); y_pred.append(p)
    return y_true, y_pred



def overlap_compute_metrics_from_export(export_dir: str, mode: str = "strict"):
    """
    Compute and print classification metrics comparing human and LLM relevance labels from Argilla export.

    Loads records from 'records.json' in the specified export directory, extracts human and LLM labels,
    maps them to binary values, and computes confusion matrix, precision, recall, F1 score, and a full
    classification report. Supports both "strict" and "lenient" mapping modes for human labels.

    Args:
        export_dir (str): Path to the directory containing Argilla export files.
        mode (str): Label mapping mode, either "strict" or "lenient".

    Returns:
        None
    """
    records_path = os.path.join(export_dir, "records.json")
    if not os.path.isfile(records_path):
        print(f"⚠️  No records.json at {records_path}")
        return
    records = _load_records(records_path)
    if not records:
        print("⚠️  No records found in export.")
        return

    y_true, y_pred = [], []
    for r in records:
        human = _overlap_extract_human_label(r)
        meta = r.get("metadata") or {}
        if meta.get("subset") == "overlap":
            # Prefer 'is_relevant' as per your export, but accept common fallbacks
            llm = meta.get("is_relevant")
            human_majority_winner = _majority_label_winner(human)
            t, p = map_human_label(human_majority_winner, mode), map_llm(llm)
            if t is None or p is None:
                continue
            y_true.append(t); y_pred.append(p)

    if not y_true:
        print("⚠️  No comparable labeled records found.")
        return
    return y_true, y_pred

def combined_compute_metrics_from_export(export_dir: str, mode: str = "strict") -> tuple[list[int], list[int]]:
    """
    Combine overlap (majority) and disjoint into one y_true/y_pred.
    """
    y_true_o, y_pred_o = overlap_compute_metrics_from_export(export_dir, mode)
    y_true_d, y_pred_d = disjoint_compute_metrics_from_export(export_dir, mode)
    return (y_true_o + y_true_d), (y_pred_o + y_pred_d)