# scripts/leaderboard/update_leaderboard.py
from pathlib import Path
import pandas as pd
import subprocess
import json

from scripts.encryption.decrypt import decrypt_file

SUBMISSIONS_DIR = Path("submissions")
LEADERBOARD_CSV = Path(__file__).resolve().parent / "leaderboard.csv"


def get_leaderboard_data():
    leaderboard = []

    for team_dir in SUBMISSIONS_DIR.iterdir():
        if not team_dir.is_dir():
            continue

        ideal_enc = team_dir / "ideal.enc"
        pert_enc = team_dir / "perturbed.enc"

        if not ideal_enc.exists() or not pert_enc.exists():
            print(f"Skipping {team_dir.name}: missing files")
            continue

        # Decrypted paths
        ideal_csv = team_dir / "ideal_submissions.csv"
        pert_csv = team_dir / "perturbed_submission.csv"

        # Decrypt files
        decrypt_file(ideal_enc, ideal_csv)
        decrypt_file(pert_enc, pert_csv)

        # Score the ideal CSV
        try:
            ideal_scores_json = subprocess.check_output([
                "python",
                "scripts/score_submission.py",
                str(ideal_csv),
                "--require-metadata"
            ])
            ideal_scores = json.loads(ideal_scores_json)
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {ideal_csv}: {e}")
            continue

        # Score the perturbed CSV
        try:
            pert_scores_json = subprocess.check_output([
                "python",
                "scripts/score_submission.py",
                str(pert_csv),
                "--require-metadata"
            ])
            pert_scores = json.loads(pert_scores_json)
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {pert_csv}: {e}")
            continue

        # Compute leaderboard scores
        scores = {
            "validation_f1_ideal": ideal_scores["validation_f1_score"],
            "validation_f1_perturbed": pert_scores["validation_f1_score"],
            "robustness_gap": ideal_scores["validation_f1_score"] - pert_scores["validation_f1_score"]
        }

        leaderboard.append({
            "team_name": team_dir.name,
            **scores
        })

    return leaderboard


def update_leaderboard_csv():
    leaderboard_data = get_leaderboard_data()
    if not leaderboard_data:
        print("No submissions found")
        return

    df = pd.DataFrame(leaderboard_data)

    # Rank: sort by perturbed F1 descending, then smaller gap wins
    df = df.sort_values(
        ["validation_f1_perturbed", "robustness_gap"],
        ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"Updated leaderboard at {LEADERBOARD_CSV}")


if __name__ == "__main__":
    update_leaderboard_csv()
