import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_predictions(csv_path):
    return pd.read_csv(csv_path)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Match", "Match"],
                yticklabels=["No Match", "Match"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    pred_path = "checkpoints/predictions3.csv"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    df = load_predictions(pred_path)
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    y_prob = df["pred_prob"]

    metrics = compute_metrics(y_true, y_pred, y_prob)

    #print("\n📊 Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")

    plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, "match_confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, os.path.join(output_dir, "match_roc_curve.png"))
    plot_precision_recall(y_true, y_prob, os.path.join(output_dir, "match_precision_recall_curve.png"))
    #print("\n✅ Saved evaluation plots to 'results/'")