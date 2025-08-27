# eval.py
import argparse
from metrics_utils import (
    overlap_compute_metrics_from_export,
    disjoint_compute_metrics_from_export,
    combined_compute_metrics_from_export,
    print_binary_classification_metrics,
    compute_simple_overlap_agreement,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--mode", choices=["strict","lenient","both"], default="both")
    args = ap.parse_args()

    # Compute overlap agreement (mode-independent)
    compute_simple_overlap_agreement(args.export_dir)

    if args.mode in ("strict","both"):
        y_disjoint_true, y_disjoint_pred = disjoint_compute_metrics_from_export(args.export_dir, "strict")
        print_binary_classification_metrics(y_disjoint_true, y_disjoint_pred, "strict", "Disjoint")
        y_overlap_true, y_overlap_pred = overlap_compute_metrics_from_export(args.export_dir, "strict")
        print_binary_classification_metrics(y_overlap_true, y_overlap_pred, "strict", "Overlap")
        y_combined_true, y_combined_pred = combined_compute_metrics_from_export(args.export_dir, "strict")
        print_binary_classification_metrics(y_combined_true, y_combined_pred, "strict", "Combined")
    if args.mode in ("lenient","both"):
        y_disjoint_true, y_disjoint_pred = disjoint_compute_metrics_from_export(args.export_dir, "lenient")
        print_binary_classification_metrics(y_disjoint_true, y_disjoint_pred, "lenient", "Disjoint")
        y_overlap_true, y_overlap_pred = overlap_compute_metrics_from_export(args.export_dir, "lenient")
        print_binary_classification_metrics(y_overlap_true, y_overlap_pred, "lenient", "Overlap")
        y_combined_true, y_combined_pred = combined_compute_metrics_from_export(args.export_dir, "lenient")
        print_binary_classification_metrics(y_combined_true, y_combined_pred, "lenient", "Combined")
