import os
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history, out_dir, title_prefix=""):
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss and not val_loss:
        return

    plt.figure()
    if train_loss:
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train loss")
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curves")
    plt.legend()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved loss curves to {path}")


def plot_roc_pr_cm(curves, out_dir, title_prefix=""):
    fpr = curves["fpr"]
    tpr = curves["tpr"]
    roc_auc = curves["roc_auc"]
    precision = curves["precision"]
    recall = curves["recall"]
    pr_auc = curves["pr_auc"]
    cm = curves["cm"]

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved ROC curve to {roc_path}")

    # PR
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.legend()
    pr_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved PR curve to {pr_path}")

    # Confusion matrix
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = [0, 1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm.max() / 2.0 if cm.max() != 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix to {cm_path}")
