from pathlib import Path
import pandas as pd
from datetime import datetime
from calculate_scores import calculate_scores

SUBMISSIONS_DIR = Path(__file__).resolve().parent.parent / "submissions"

def get_participant_submissions():
    participants = {}
    for f in SUBMISSIONS_DIR.iterdir():
        if f.is_file() and f.suffix == ".csv":
            name = f.stem.lower()
            team = "participant"  # replace with logic to extract participant if needed
            if "ideal" in name:
                participants.setdefault(team, {})["ideal"] = f
            elif "perturbed" in name:
                participants.setdefault(team, {})["perturbed"] = f
    return participants

def update_leaderboard_csv():
    participants = get_participant_submissions()
    rows = []

    for team, files in participants.items():
        if "ideal" in files and "perturbed" in files:
            ideal_scores = calculate_scores(files["ideal"])
            perturbed_scores = calculate_scores(files["perturbed"])
            row = {
                "team_name": team,
                "validation_f1_ideal": ideal_scores["validation_f1_score"],
                "validation_f1_perturbed": perturbed_scores["validation_f1_score"],
                "robustness_gap": ideal_scores["validation_f1_score"] - perturbed_scores["validation_f1_score"],
                "timestamp": datetime.fromtimestamp(files["ideal"].stat().st_mtime).isoformat(),
            }
            rows.append(row)
        else:
            print(f"Skipping {team}, missing ideal or perturbed submission.")

    df = pd.DataFrame(rows)
    df = df.sort_values("validation_f1_ideal", ascending=False)
    output_path = Path(__file__).resolve().parent / "leaderboard.csv"
    df.to_csv(output_path, index=False)
    print(f"Leaderboard updated: {output_path}")

if __name__ == "__main__":
    update_leaderboard_csv()
