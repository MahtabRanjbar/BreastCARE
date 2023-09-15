import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def save_report(y_pred: list, y_prob: list, y_test: list) -> None:

    """
        Saves a report including a confusion matrix, classification report, and ROC curve plot.

    Args:
        y_pred (array-like): Predicted labels.
        y_prob (array-like): Predicted probabilities.
        y_test (array-like): True labels.

    Returns:
        None
    """

    report_folder = "report"
    os.makedirs(report_folder, exist_ok=True)

    # Save confusion matrix
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["B", "M"])
    plt.yticks([0, 1], ["B", "M"])
    plt.savefig(os.path.join(report_folder, "confusion_matrix.png"))

    # Save classification report
    classification_rep = classification_report(y_test, y_pred)
    with open(os.path.join(report_folder, "classification_report.txt"), "w") as f:
        f.write(classification_rep)

    # Calculate the ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(report_folder, "roc_curve.png"))
