import json
from collections import defaultdict

# ==== CONFIG ====
json_path = "data/image_matches/image_matches.json"  # Path to your JSON file
# ================

# Load the JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Dictionaries to count occurrences
he_counts = defaultdict(list)
ihc_counts = defaultdict(list)

# Iterate through JSON and collect filenames
for case_id, matches in data.items():
    if isinstance(matches, str) and matches == "NO_MATCHES":
        continue

    for match_id, stains in matches.items():
        he_filename = stains.get("he")
        ihc_filename = stains.get("ihc")

        if he_filename:
            he_counts[he_filename].append((case_id, match_id))
        if ihc_filename:
            ihc_counts[ihc_filename].append((case_id, match_id))

# Print duplicates
print("Duplicate HE images:")
for filename, occurrences in he_counts.items():
    if len(occurrences) > 1:
        print(f"{filename} appears {len(occurrences)} times at: {occurrences}")

print("\nDuplicate IHC images:")
for filename, occurrences in ihc_counts.items():
    if len(occurrences) > 1:
        print(f"{filename} appears {len(occurrences)} times at: {occurrences}")
