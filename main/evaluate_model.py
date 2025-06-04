import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
import torch

def load_predictions(csv_path):
    return pd.read_csv(csv_path)

def load_loss(loss_path):
    return torch.load(loss_path)

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

def prep_loss_data(loss_dict):
    train_epoch_loss = []
    for epoch in range(len(loss_dict['val_step'])): # for epoch in range(epochs):
        train_loss = 0
        for step in range(len(loss_dict['train_step']) // len(loss_dict['val_step'])): # for step in range(number of steps per epoch):
            train_loss += loss_dict['train_loss'][step + epoch * len(loss_dict['train_step']) // len(loss_dict['val_step'])] # add loss per step to train_loss
        train_epoch_loss.append(train_loss / (len(loss_dict['train_step']) // len(loss_dict['val_step']))) # append average loss per epoch
    return train_epoch_loss 
            

def plot_loss(loss_dict, train_epoch_loss, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(train_epoch_loss)), train_epoch_loss, label='Training Loss')
    plt.plot(range(len(train_epoch_loss)), loss_dict['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    loss_dir = "2025-06-03 16.02.45_loss.pth"
    model_id = "_06-03_16.02_e49"
    prediction_dir = config.prediction_dir
    predictions = os.path.join(prediction_dir, f'predictions{model_id}.csv')
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    df = load_predictions(predictions)
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    y_prob = df["pred_prob"]

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print("Evaluation Metrics:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")

    plot_confusion_matrix(y_true, y_pred, os.path.join(output_dir, f'confusion_matrix{model_id}.png'))
    plot_roc_curve(y_true, y_prob, os.path.join(output_dir, f'roc_curve{model_id}.png'))
    plot_precision_recall(y_true, y_prob, os.path.join(output_dir, f'precision_recall_curve{model_id}.png'))

    # loss plot
    checkpoint_dir = config.checkpoints_dir
    loss_path = os.path.join(checkpoint_dir, loss_dir)
    loss_output_path = os.path.join(output_dir, f'loss_plot{model_id}.png')
    loss_dict = load_loss(loss_path)
    train_epoch_loss = prep_loss_data(loss_dict)
    plot_loss(loss_dict, train_epoch_loss, loss_output_path)
    