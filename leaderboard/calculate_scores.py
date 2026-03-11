def calculate_scores_pair(ideal_path: Path, perturbed_path: Path):
    scores_ideal = calculate_scores(ideal_path)
    scores_perturbed = calculate_scores(perturbed_path)
    
    robustness_gap = scores_ideal["validation_f1_score"] - scores_perturbed["validation_f1_score"]
    
    return {
        "validation_f1_ideal": scores_ideal["validation_f1_score"],
        "validation_f1_perturbed": scores_perturbed["validation_f1_score"],
        "robustness_gap": robustness_gap,
        "validation_accuracy_ideal": scores_ideal["validation_accuracy"],
        "validation_accuracy_perturbed": scores_perturbed["validation_accuracy"],
    }
