import os
import argparse
import pandas as pd
from sklearn.metrics import f1_score


DATA_DIR = "data"
SUBMISSION_DIR = "submissions"


def evaluate(submission_file, truth_file):
    sub = pd.read_csv(submission_file)
    truth = pd.read_csv(truth_file)

    merged = sub.merge(truth, on="graph_index")

    y_true = merged["target"]
    y_pred = merged["prediction"] if "prediction" in merged else merged["target_y"]

    return f1_score(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description='Score submission files')
    parser.add_argument('submission_path', nargs='?', help='Path to submission file (optional)')
    parser.add_argument('--require-metadata', action='store_true', help='Require metadata (ignored for compatibility)')
    
    args = parser.parse_args()
    
    # If submission_path is provided, use it to determine which files to score
    if args.submission_path:
        print(f"Processing submission: {args.submission_path}")
        # You can implement logic here to handle single submission files
        # For now, we'll just note it and continue with the default behavior
    
    ideal_sub = os.path.join(SUBMISSION_DIR, "ideal_submission.csv")
    perturbed_sub = os.path.join(SUBMISSION_DIR, "perturbed_submission.csv")
    truth = os.path.join(DATA_DIR, "test_labels_hidden.csv")

    if not os.path.exists(ideal_sub):
        raise ValueError("ideal_submission.csv missing")

    if not os.path.exists(perturbed_sub):
        raise ValueError("perturbed_submission.csv missing")
    
    if not os.path.exists(truth):
        raise ValueError(f"Truth file not found: {truth}")

    f1_ideal = evaluate(ideal_sub, truth)
    f1_perturbed = evaluate(perturbed_sub, truth)

    robustness_gap = abs(f1_ideal - f1_perturbed)

    print("Evaluation Results")
    print("------------------")
    print(f"F1 Ideal: {f1_ideal:.4f}")
    print(f"F1 Perturbed: {f1_perturbed:.4f}")
    print(f"Robustness Gap: {robustness_gap:.4f}")
    
    # Save results to a file
    results = {
        "f1_ideal": f1_ideal,
        "f1_perturbed": f1_perturbed,
        "robustness_gap": robustness_gap
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv("scoring_results.csv", index=False)
    
    return results


if __name__ == "__main__":
    main()
