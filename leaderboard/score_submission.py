import os
import pandas as pd
from pathlib import Path
from leaderboard.calculate_scores import calculate_scores_pair


def score():

    submissions_dir = Path("submissions")
    leaderboard_file = Path("leaderboard/leaderboard.csv")

    ideal_path = submissions_dir / "ideal_submission.csv"
    perturbed_path = submissions_dir / "perturbed_submission.csv"

    scores = calculate_scores_pair(ideal_path, perturbed_path)

    print("Scores:", scores)

    # Create leaderboard file if it doesn't exist
    if not leaderboard_file.exists():
        df = pd.DataFrame(columns=[
            "validation_f1_ideal",
            "validation_f1_perturbed",
            "robustness_gap",
            "validation_accuracy_ideal",
            "validation_accuracy_perturbed"
        ])
    else:
        df = pd.read_csv(leaderboard_file)

    new_row = {
        "validation_f1_ideal": scores["validation_f1_ideal"],
        "validation_f1_perturbed": scores["validation_f1_perturbed"],
        "robustness_gap": scores["robustness_gap"],
        "validation_accuracy_ideal": scores["validation_accuracy_ideal"],
        "validation_accuracy_perturbed": scores["validation_accuracy_perturbed"]
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(leaderboard_file, index=False)

    return scores


if __name__ == "__main__":
    score()
