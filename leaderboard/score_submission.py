import pandas as pd
import os
from leaderboard.calculate_scores import compute_scores


def score():

    submissions_dir = "submissions"
    data_dir = "data"

    ideal_path = os.path.join(submissions_dir, "ideal_submission.csv")
    perturbed_path = os.path.join(submissions_dir, "perturbed_submission.csv")

    truth_path = os.path.join(data_dir, "test_labels_hidden.csv")

    ideal = pd.read_csv(ideal_path)
    perturbed = pd.read_csv(perturbed_path)
    truth = pd.read_csv(truth_path)

    scores = compute_scores(ideal, perturbed, truth)

    print("Scores:", scores)

    return scores
