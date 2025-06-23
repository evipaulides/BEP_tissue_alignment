import json
import numpy as np
import config
import os
from collections import defaultdict, Counter

# === HELPER ===
def get_case_id(filename):
    return filename.split('_')[0]

# Load the saved predictions
model_id = "_06-19_03.36_e99"
match_csv = config.test_csv
output_dir = os.path.join("results_test", model_id)
os.makedirs(output_dir, exist_ok=True)


with open(f"results_test/{model_id}/scores_he_to_ihc.json", "r") as f:
    scores_he_to_ihc = json.load(f)

with open(f"results_test/{model_id}/scores_ihc_to_he.json", "r") as f:
    scores_ihc_to_he = json.load(f)

# Load original CSV with true matches
import pandas as pd
df = pd.read_csv(match_csv)
he_to_ihc = dict(zip(df['HE'], df['IHC']))
ihc_to_he = dict(zip(df['IHC'], df['HE']))

# Compute ranks from predictions
he_ranks = []
for he_img, sorted_scores in scores_he_to_ihc.items():
    predicted_names = [ihc_name for ihc_name, _ in sorted_scores]
    rank = predicted_names.index(he_to_ihc[he_img]) + 1
    he_ranks.append(rank)

ihc_ranks = []
for ihc_img, sorted_scores in scores_ihc_to_he.items():
    predicted_names = [he_name for he_name, _ in sorted_scores]
    rank = predicted_names.index(ihc_to_he[ihc_img]) + 1
    ihc_ranks.append(rank)

# Compute metrics
def compute_metrics(ranks , ks=[1, 5, 10]):
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
    q1 = np.percentile(ranks, 25)
    q3 = np.percentile(ranks, 75)
    IQR = (q1, q3)


    return recall_at_k, mean_rank, median_rank, std_rank, IQR
    # return {
    #     'Recall@1': np.mean(ranks <= 1),
    #     'Recall@5': np.mean(ranks <= 5),
    #     'Recall@10': np.mean(ranks <= 10),
    #     'Mean Rank': np.mean(ranks),
    #     'Median Rank': np.median(ranks),
    #     'Standard Deviation': np.std(ranks),
    #     '95 confidence interval': (lower_bound, upper_bound)
    # }

print("H&E → IHC:", compute_metrics(he_ranks))
print("IHC → H&E:", compute_metrics(ihc_ranks))

recall_he, mean_he, median_he, std_he, iqr_he = compute_metrics(he_ranks)
recall_ihc, mean_ihc, median_ihc, std_ihc, iqr_ihc = compute_metrics(ihc_ranks)

# === GROUP BY CASE ===
case_to_he = defaultdict(list)
case_to_ihc = defaultdict(list)

for he in df['HE']:
    case_to_he[get_case_id(he)].append(he)

for ihc in df['IHC']:
    case_to_ihc[get_case_id(ihc)].append(ihc)

# === CASE-LEVEL RANKING H&E → IHC ===
he_case_ranks = []
he_case_correct = 0

for case_id, he_imgs in case_to_he.items():
    all_correct = True
    for he_img in case_to_he[case_id]:
        if he_img not in scores_he_to_ihc:
            continue
        true_ihc = he_to_ihc[he_img]
        candidate_ihcs = case_to_ihc[case_id]

        case_scores = [(ihc, score) for ihc, score in scores_he_to_ihc[he_img] if ihc in candidate_ihcs]
        if not case_scores:
            continue
        ranked = sorted(case_scores, key=lambda x: x[1], reverse=True)
        names_ranked = [ihc for ihc, _ in ranked]
        rank = names_ranked.index(true_ihc) + 1
        he_case_ranks.append(rank)

        if rank != 1:
            all_correct = False
    if all_correct and he_imgs:
        he_case_correct += 1

# === CASE-LEVEL RANKING IHC → H&E ===
ihc_case_ranks = []
ihc_case_correct = 0

for case_id, ihc_imgs in case_to_ihc.items():
    all_correct = True
    for ihc_img in case_to_ihc[case_id]:
        if ihc_img not in scores_ihc_to_he:
            continue
        true_he = ihc_to_he[ihc_img]
        candidate_hes = case_to_he[case_id]

        case_scores = [(he, score) for he, score in scores_ihc_to_he[ihc_img] if he in candidate_hes]
        if not case_scores:
            continue
        ranked = sorted(case_scores, key=lambda x: x[1], reverse=True)
        names_ranked = [he for he, _ in ranked]
        rank = names_ranked.index(true_he) + 1
        ihc_case_ranks.append(rank)
        if rank != 1:
            all_correct = False
    if all_correct and ihc_imgs:
        ihc_case_correct += 1

# === CASE-LEVEL METRICS ===
case_recall_he, case_mean_he, case_median_he, case_std_he, case_iqr_he = compute_metrics(he_case_ranks)
case_recall_ihc, case_mean_ihc, case_median_ihc, case_std_ihc, case_iqr_ihc = compute_metrics(ihc_case_ranks)

# === Compute distribution of matches per case
def get_case_size_distribution(case_to_images, label):
    match_counts = [len(images) for images in case_to_images.values()]
    dist = Counter(match_counts)
    print(f"\n{label} Case Size Distribution (number of images per case):")
    for size, count in sorted(dist.items()):
        print(f"  {size} image(s): {count} case(s)")
    print(f"  Max size: {max(match_counts)}, Min size: {min(match_counts)}, Mean: {np.mean(match_counts):.2f}, Median: {np.median(match_counts):.1f}, IQR: {np.percentile(match_counts, 25):.2f} - {np.percentile(match_counts, 75):.2f}")

    max_size = max(match_counts)
    min_size = min(match_counts)
    mean_size = np.mean(match_counts)
    median_size = np.median(match_counts)
    iqr_size = (np.percentile(match_counts, 25), np.percentile(match_counts, 75))

    return {max_size, min_size, mean_size, median_size, iqr_size}
# Print distributions
max_size_he, min_size_he, mean_size_he, median_size_he, iqr_size_he = get_case_size_distribution(case_to_he, "H&E")
max_size_ihc, min_size_ihc, mean_size_ihc, median_size_ihc, iqr_size_ihc = get_case_size_distribution(case_to_ihc, "IHC")


# === Save results to file ===

stats_file = os.path.join(output_dir, "analyse_ranking_stats.txt")
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

    f.write("\nCase-level H&E → IHC direction:\n")
    for k, v in case_recall_he.items():
        f.write(f"{k}: {v:.3f}\n")
    f.write(f"Mean Rank: {case_mean_he:.2f}\n")
    f.write(f"Median Rank: {case_median_he:.1f}\n")
    f.write(f"Standard Deviation: {case_std_he:.2f}\n")
    f.write(f"IQR: {case_iqr_he[0]:.2f} - {case_iqr_he[1]:.2f}\n")
    f.write(f"Correct cases: {he_case_correct}/{len(case_to_he)} ({he_case_correct/len(case_to_he)*100:.1f}%)\n\n")
    f.write("Case-level IHC → H&E direction:\n")
    for k, v in case_recall_ihc.items():
        f.write(f"{k}: {v:.3f}\n")
    f.write(f"Mean Rank: {case_mean_ihc:.2f}\n")
    f.write(f"Median Rank: {case_median_ihc:.1f}\n")
    f.write(f"Standard Deviation: {case_std_ihc:.2f}\n")
    f.write(f"IQR: {case_iqr_ihc[0]:.2f} - {case_iqr_ihc[1]:.2f}\n")
    f.write(f"Correct cases: {ihc_case_correct}/{len(case_to_ihc)} ({ihc_case_correct/len(case_to_ihc)*100:.1f}%)\n\n")

    f.write("\nCase Size Distribution H&E:\n")
    f.write(f"Max size: {max_size_he}, Min size: {min_size_he}, Mean: {mean_size_he:.2f}, Median: {median_size_he:.1f}, IQR: {iqr_size_he[0]:.2f} - {iqr_size_he[1]:.2f}\n")
    f.write("\nCase Size Distribution IHC:\n")
    f.write(f"Max size: {max_size_ihc}, Min size: {min_size_ihc}, Mean: {mean_size_ihc:.2f}, Median: {median_size_ihc:.1f}, IQR: {iqr_size_ihc[0]:.2f} - {iqr_size_ihc[1]:.2f}\n")

print(f"Saved ranking statistics to {stats_file}")