# leaderboard/calculate_scores.py
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
import os

def calculate_scores(submission_path: Path):
    """
    Compute F1 score for a single CSV submission and return as dict
    """
    print(f"DEBUG: calculate_scores called with submission: {submission_path}")
    
    # Get test labels path from environment variable
    test_labels_path = os.environ.get('TEST_LABELS_CSV')
    print(f"DEBUG: TEST_LABELS_CSV = {test_labels_path}")
    
    if not test_labels_path:
        # Try default location
        default_path = Path(__file__).parent.parent / "data" / "test_labels_hidden.csv"
        if default_path.exists():
            test_labels_path = str(default_path)
            print(f"DEBUG: Using default test labels path: {test_labels_path}")
        else:
            raise FileNotFoundError(
                "TEST_LABELS_CSV environment variable not set and no default test labels found"
            )
    
    ground_truth_path = Path(test_labels_path)
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    print(f"DEBUG: Loading submission from {submission_path}")
    submission_df = pd.read_csv(submission_path)
    
    print(f"DEBUG: Loading ground truth from {ground_truth_path}")
    gt_df = pd.read_csv(ground_truth_path)

    print(f"DEBUG: Submission columns: {list(submission_df.columns)}")
    print(f"DEBUG: Ground truth columns: {list(gt_df.columns)}")

    # Merge on graph_index
    merged = submission_df.merge(gt_df, on="graph_index", how="inner")
    print(f"DEBUG: Merged shape: {merged.shape}")
    
    # Find prediction column
    pred_col = None
    for col in submission_df.columns:
        if col != "graph_index":
            pred_col = col
            break
    
    if pred_col is None:
        raise ValueError("No prediction column found in submission")
    
    # Find ground truth column
    truth_col = None
    for col in gt_df.columns:
        if col != "graph_index":
            truth_col = col
            break
    
    if truth_col is None:
        raise ValueError("No ground truth column found in test labels")
    
    print(f"DEBUG: Using prediction column: {pred_col}")
    print(f"DEBUG: Using ground truth column: {truth_col}")
    
    y_pred = merged[pred_col]
    y_true = merged[truth_col]
    
    print(f"DEBUG: y_pred sample: {y_pred.head()}")
    print(f"DEBUG: y_true sample: {y_true.head()}")
    
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"DEBUG: Calculated F1 score: {f1}")

    return {"validation_f1_score": f1}
