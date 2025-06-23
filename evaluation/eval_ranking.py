import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from external.dataset_utils import get_centroid_of_mask
import torchvision.transforms.functional as TF
import config
from external.DualInputViT import DualInputViT
from external.DualBranchViT import DualBranchViT
from external.ViT_utils import convert_state_dict
from torch import nn
import torch.nn.functional as F
import json

# === Image and position loading ===

def load_image(img_dir, img_mask_dir, patch_size=16):
    img = Image.open(img_dir).convert('RGB')
    mask = Image.open(img_mask_dir).convert('L')

    img = TF.to_tensor(img)
    pos = get_positions(img, mask, patch_size)

    return img, pos

def get_positions(img, mask, patch_size):
    centroid = get_centroid_of_mask(mask)
    cx = math.floor(centroid[0])
    cy = math.floor(centroid[1])

    C, H, W = img.shape 
    h_patches = H // patch_size
    w_patches = W // patch_size

    centroid_patch_x = int(cx // patch_size)
    centroid_patch_y = int(cy // patch_size)

    x_range = torch.arange(w_patches) - centroid_patch_x
    y_range = -(torch.arange(h_patches) - centroid_patch_y)

    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    pos = torch.stack([y_grid, x_grid], dim=2).reshape(-1,2)

    return pos

def compute_recall_and_rank_metrics(ranks, ks=[1, 5, 10]):
    """
    Computes Recall@k, mean rank, and median rank.
    Assumes ranks is a list of 1-based ranks of correct matches.
    """
    ranks = np.array(ranks)
    n = len(ranks)

    # Recall@k
    recall_at_k = {}
    for k in ks:
        recall_at_k[f"Recall@{k}"] = np.sum(ranks <= k) / n

    # Rank statistics
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    std_rank = np.std(ranks)
    # calculate inter-quartile range
    q1 = np.percentile(ranks, 25)
    q3 = np.percentile(ranks, 75)
    IQR = (q1, q3)


    return recall_at_k, mean_rank, median_rank, std_rank, IQR

def get_code(img_path):
    """
    Extracts the code from the image filename.
    Assumes the filename format is 'code_xxx_yyy.png'.
    """
    return img_path.split('_')[0] + '_' + img_path.split('_')[1]

# === Evaluation and plot saving ===

def evaluate_ranking_and_save(
    model,
    match_csv,
    he_folder,
    ihc_folder,
    he_mask_folder,
    ihc_mask_folder,
    model_id,
    device='cuda',
    top_k=10
):
    model.eval().to(device)

    output_dir = os.path.join("results_test", model_id)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(match_csv)
    match_pairs = set((row['HE'], row['IHC']) for _, row in df.iterrows())

    he_to_ihc = dict(zip(df['HE'], df['IHC']))
    ihc_to_he = dict(zip(df['IHC'], df['HE']))

    he_images = sorted(df['HE'].unique())
    ihc_images = sorted(df['IHC'].unique())

    all_ranks = []
    worst_error = 0
    worst_pair = None
    all_scores = {}

    # HE -> IHC
    for he_name in tqdm(he_images, desc="Ranking H&E → IHC"):
        he_path = os.path.join(he_folder, he_name)
        he_mask_path = os.path.join(he_mask_folder, he_name)
        he_img, he_pos = load_image(he_path, he_mask_path)
        he_img, he_pos = he_img.unsqueeze(0).to(device), he_pos.unsqueeze(0).to(device)

        scores = []
        for ihc_name in ihc_images:
            ihc_path = os.path.join(ihc_folder, ihc_name)
            ihc_mask_path = os.path.join(ihc_mask_folder, ihc_name)
            ihc_img, ihc_pos = load_image(ihc_path, ihc_mask_path)
            ihc_img, ihc_pos = ihc_img.unsqueeze(0).to(device), ihc_pos.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(he_img, he_pos, ihc_img, ihc_pos)
                prob = F.sigmoid(logit).item()

            scores.append((ihc_name, prob))

            if (he_name, ihc_name) in match_pairs:
                error = abs(prob - 1.0)
                if error > worst_error:
                    worst_error = error
                    worst_pair = (he_name, ihc_name, prob)

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        ranked_names = [ihc_name for ihc_name, _ in sorted_scores]
        true_match = df[df['HE'] == he_name]['IHC'].values[0]
        rank = ranked_names.index(true_match) + 1

        all_ranks.append(rank)
        all_scores[he_name] = sorted_scores

    # Save H&E to IHC scores
    with open(os.path.join(output_dir, "scores_he_to_ihc.json"), 'w') as f:
        json.dump(all_scores, f)


    # Select percentile ranks
    sorted_he = [he for _, he in sorted(zip(all_ranks, he_images))]
    n = len(sorted_he)
    selected_indices = [0, int(0.01 * n), int(0.5 * n), int(0.99 * n)]
    selected_names = [sorted_he[i] for i in selected_indices]

    counter = 0
    for he_plot in selected_names:
        counter += 1
        he_code = get_code(he_plot)
        fig, axs = plt.subplots(1, top_k + 1, figsize=(2.5 * (top_k + 1), 4))
        axs[0].imshow(Image.open(os.path.join(he_folder, he_plot)))
        axs[0].set_title("H&E\n" + he_code)
        axs[0].axis('off')
        for i in range(top_k):
            ihc_name, score = all_scores[he_plot][i]
            axs[i + 1].imshow(Image.open(os.path.join(ihc_folder, ihc_name)))
            axs[i + 1].set_title(f"Rank {i+1}\n{score:.2f}")
            axs[i + 1].axis('off')
        plt.tight_layout()
        true_match = he_to_ihc[he_plot]
        match_rank = [ihc for ihc, _ in all_scores[he_plot]].index(true_match) + 1
        print(f"{counter} H&E: {he_code} — True match: {true_match} — Rank: {match_rank}")
        plt.savefig(os.path.join(output_dir, f"analysis_HE_{he_code}_rank{match_rank}.png"))
        plt.close()

    # Save most incorrect prediction plot
    if worst_pair:
        he_name, ihc_name, prob = worst_pair
        he_code = get_code(he_name)
        ihc_code = get_code(ihc_name)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(Image.open(os.path.join(he_folder, he_name)))
        axs[0].set_title(f"H&E\n{he_code}")
        axs[0].axis('off')
        axs[1].imshow(Image.open(os.path.join(ihc_folder, ihc_name)))
        axs[1].set_title(f"IHC\n{ihc_code}\nPred: {prob:.2f}, True: 1.0")
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"worst_prediction_{he_code}_{ihc_code}.png"))
        plt.close()




    # IHC -> HE
    ihc_ranks = []
    all_scores_ihc = {}
    worst_error = 0
    worst_pair = None
    for ihc_name in tqdm(ihc_images, desc="Ranking IHC → H&E"):
        ihc_path = os.path.join(ihc_folder, ihc_name)
        ihc_mask_path = os.path.join(ihc_mask_folder, ihc_name)
        ihc_img, ihc_pos = load_image(ihc_path, ihc_mask_path)
        ihc_img, ihc_pos = ihc_img.unsqueeze(0).to(device), ihc_pos.unsqueeze(0).to(device)

        scores = []
        for he_name in he_images:
            he_path = os.path.join(he_folder, he_name)
            he_mask_path = os.path.join(he_mask_folder, he_name)
            he_img, he_pos = load_image(he_path, he_mask_path)
            he_img, he_pos = he_img.unsqueeze(0).to(device), he_pos.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(he_img, he_pos, ihc_img, ihc_pos)
                prob = F.sigmoid(logit).item()

            scores.append((he_name, prob))

            if (he_name, ihc_name) in match_pairs:
                error = abs(prob - 1.0)
                if error > worst_error:
                    worst_error = error
                    worst_pair = (he_name, ihc_name, prob)

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        all_scores_ihc[ihc_name] = sorted_scores
        ranked_names = [he_name for he_name, _ in sorted_scores]
        true_match = df[df['IHC'] == ihc_name]['HE'].values[0]
        rank = ranked_names.index(true_match) + 1

        ihc_ranks.append(rank)

    # Save IHC to H&E scores
    with open(os.path.join(output_dir, "scores_ihc_to_he.json"), 'w') as f:
        json.dump(all_scores_ihc, f)

    # Select percentile ranks
    sorted_ihc = [ihc for _, ihc in sorted(zip(ihc_ranks, ihc_images))]
    n = len(sorted_ihc)
    selected_indices = [0, int(0.01 * n), int(0.5 * n), int(0.99 * n)]
    selected_names = [sorted_ihc[i] for i in selected_indices]

    counter = 0
    for ihc_plot in selected_names:
        counter += 1
        ihc_code = get_code(ihc_plot)
        fig, axs = plt.subplots(1, top_k + 1, figsize=(2.5 * (top_k + 1), 4))
        axs[0].imshow(Image.open(os.path.join(ihc_folder, ihc_plot)))
        axs[0].set_title("IHC\n" + ihc_code)
        axs[0].axis('off')
        for i in range(top_k):
            he_name, score = all_scores_ihc[ihc_plot][i]
            axs[i + 1].imshow(Image.open(os.path.join(he_folder, he_name)))
            axs[i + 1].set_title(f"Rank {i+1}\n{score:.2f}")
            axs[i + 1].axis('off')
        plt.tight_layout()
        true_match = ihc_to_he[ihc_plot]
        match_rank = [he for he, _ in all_scores_ihc[ihc_plot]].index(true_match) + 1
        print(f"{counter} IHC: {ihc_code} — True match: {true_match} — Rank: {match_rank}")
        plt.savefig(os.path.join(output_dir, f"analysis_IHC_{ihc_code}_rank{match_rank}.png"))
        plt.close()

    if worst_pair:
        he_name, ihc_name, prob = worst_pair
        ihc_code = get_code(ihc_name)
        he_code = get_code(he_name)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(Image.open(os.path.join(ihc_folder, ihc_name)))
        axs[0].set_title(f"IHC\n{ihc_code}")
        axs[0].axis('off')
        axs[1].imshow(Image.open(os.path.join(he_folder, he_name)))
        axs[1].set_title(f"H&E\n{he_code}\nPred: {prob:.2f}, True: 1.0")
        axs[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"worst_prediction_{ihc_code}_{he_code}.png"))
        plt.close()


    # === Compute and save stats ===
    recall_he, mean_he, median_he, std_he, iqr_he = compute_recall_and_rank_metrics(all_ranks)
    recall_ihc, mean_ihc, median_ihc, std_ihc, iqr_ihc = compute_recall_and_rank_metrics(ihc_ranks)
    
    stats_file = os.path.join(output_dir, "ranking_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("H&E → IHC direction:\n")
        for k, v in recall_he.items():
            f.write(f"{k}: {v:.3f}\n")
        f.write(f"Mean Rank: {mean_he:.2f}\n")
        f.write(f"Median Rank: {median_he:.1f}\n")
        f.write(f"Standard Deviation: {std_he:.2f}\n")
        f.write(f"IQR: {iqr_he[0]:.2f} - {iqr_he[1]:.2f}\n\n")

        f.write("IHC → H&E direction:\n")
        for k, v in recall_ihc.items():
            f.write(f"{k}: {v:.3f}\n")
        f.write(f"Mean Rank: {mean_ihc:.2f}\n")
        f.write(f"Median Rank: {median_ihc:.1f}\n")
        f.write(f"Standard Deviation: {std_ihc:.2f}\n")
        f.write(f"IQR: {iqr_ihc[0]:.2f} - {iqr_ihc[1]:.2f}\n")

    print(f"Saved ranking statistics to {stats_file}")

    return {
        "he_to_ihc_recall": recall_he,
        "ihc_to_he_recall": recall_ihc,
        "mean_rank_he": mean_he,
        "mean_rank_ihc": mean_ihc
    }

# === Main ===

if __name__ == "__main__":
    model_dict_path = "checkpoints/2025-06-20 10.49.02_DualInputViT_ep99.pth"
    model_id = '_06-20_10.49_e99'

    eval_csv = config.test_csv
    he_dir = config.he_dir
    ihc_dir = config.ihc_dir
    he_mask_dir = config.he_mask_dir
    ihc_mask_dir = config.ihc_mask_dir
    device = config.device

    model_architecture = config.model_architecture
    patch_shape = config.patch_shape
    input_dim = config.input_dim
    embed_dim = config.embed_dim
    n_classes = config.n_classes
    depth = config.depth
    n_heads = config.n_heads
    mlp_ratio = config.mlp_ratio
    load_pretrained_param = config.load_pretrained_param

    if model_architecture == "DualInputViT":
        model = DualInputViT(
            patch_shape=patch_shape,
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_classes=n_classes,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            pytorch_attn_imp=False,
            init_values=1e-5,
            act_layer=nn.GELU
        )
    elif model_architecture == "DualBranchViT":
        model = DualBranchViT(
            patch_shape=patch_shape,
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_classes=n_classes,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            pytorch_attn_imp=False,
            init_values=1e-5,
            act_layer=nn.GELU
        )

    model.load_state_dict(torch.load(model_dict_path, map_location=device), strict=False)
    model = model.to(device)

    stats = evaluate_ranking_and_save(
        model=model,
        match_csv=eval_csv,
        he_folder=he_dir,
        ihc_folder=ihc_dir,
        he_mask_folder=he_mask_dir,
        ihc_mask_folder=ihc_mask_dir,
        model_id=model_id,
        device=device,
        top_k=10
    )
