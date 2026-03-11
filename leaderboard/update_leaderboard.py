# leaderboard/update_leaderboard.py
from pathlib import Path
import pandas as pd
import subprocess
import json
import sys

# Resolve repo root
repo_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(repo_root))

from encryption.decrypt import decrypt_file
from leaderboard.calculate_scores import calculate_scores

# Submissions folder and leaderboard CSV
SUBMISSIONS_DIR = repo_root / "submissions"
LEADERBOARD_CSV = repo_root / "leaderboard/leaderboard.csv"

def get_leaderboard_data():
    leaderboard = []

    # Debug: list submissions folder
    print(f"DEBUG: Repo root: {repo_root}")
    print(f"DEBUG: Looking for submissions in: {SUBMISSIONS_DIR}")
    if not SUBMISSIONS_DIR.exists():
        print("DEBUG: Submissions directory does not exist!")
        return leaderboard
    else:
        print("DEBUG: Found team folders:", [d.name for d in SUBMISSIONS_DIR.iterdir() if d.is_dir()])

    # Iterate over each team
    for team_dir in SUBMISSIONS_DIR.iterdir():
        if not team_dir.is_dir():
            continue

        print(f"\nDEBUG: Processing team folder: {team_dir.name}")
        print("DEBUG: Files in team folder:", [f.name for f in team_dir.iterdir()])

        ideal_enc = team_dir / "ideal.enc"
        pert_enc = team_dir / "perturbed.enc"

        if not ideal_enc.exists() or not pert_enc.exists():
            print(f"Skipping {team_dir.name}: missing files (expected ideal.enc and perturbed.enc)")
            continue

        # Decrypted CSV files
        ideal_csv = team_dir / "ideal_submissions.csv"
        pert_csv = team_dir / "perturbed_submission.csv"

        # Decrypt
        print(f"DEBUG: Decrypting {ideal_enc} -> {ideal_csv}")
        decrypt_file(ideal_enc, ideal_csv)
        print(f"DEBUG: Decrypting {pert_enc} -> {pert_csv}")
        decrypt_file(pert_enc, pert_csv)

        # Score ideal
        try:
            ideal_scores_json = subprocess.check_output([
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(ideal_csv),
                "--require-metadata"
            ])
            ideal_scores = json.loads(ideal_scores_json)
            print(f"DEBUG: Ideal scores: {ideal_scores}")
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {ideal_csv}: {e}")
            continue

        # Score perturbed
        try:
            pert_scores_json = subprocess.check_output([
                sys.executable,
                str(repo_root / "leaderboard/score_submission.py"),
                str(pert_csv),
                "--require-metadata"
            ])
            pert_scores = json.loads(pert_scores_json)
            print(f"DEBUG: Perturbed scores: {pert_scores}")
        except subprocess.CalledProcessError as e:
            print(f"Error scoring {pert_csv}: {e}")
            continue

        leaderboard.append({
            "team_name": team_dir.name,
            "validation_f1_ideal": ideal_scores["validation_f1_score"],
            "validation_f1_perturbed": pert_scores["validation_f1_score"],
            "robustness_gap": ideal_scores["validation_f1_score"] - pert_scores["validation_f1_score"]
        })

    return leaderboard

def update_leaderboard_csv():
    leaderboard_data = get_leaderboard_data()
    if not leaderboard_data:
        print("No submissions found")
        return

    df = pd.DataFrame(leaderboard_data)
    df = df.sort_values(
        ["validation_f1_perturbed", "robustness_gap"], ascending=[False, True]
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"Updated leaderboard at {LEADERBOARD_CSV}")

if __name__ == "__main__":
    update_leaderboard_csv()
