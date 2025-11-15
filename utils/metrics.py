import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)


def evaluate_binary(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_binary_from_logits(y_true, logits, threshold=0.5):
    probs = 1 / (1 + np.exp(-logits))
    return evaluate_binary(y_true, probs, threshold)


def compute_curves(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_true, (y_probs >= 0.5).astype(int))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "cm": cm
    }
