import json
import os
import re

files = ['image_rotations_e.json', 'image_rotations_r.json', 'image_rotations.json']
merged = {}

# Merge and normalize
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        for path, value in data.items():
            filename = os.path.basename(path)
            merged[filename] = value

# Sort by setNumber and subsetNumber
def extract_sort_key(filename):
    # Match something like 42_2_something.png
    match = re.match(r"(\d+)(?:_(\d+))?_.*", filename)
    if match:
        set_num = int(match.group(1))
        subset_num = int(match.group(2)) if match.group(2) else 0
        return (set_num, subset_num, filename)
    else:
        return (float('inf'), float('inf'), filename)  # fallback if pattern doesn't match

# Sort the items
sorted_items = sorted(merged.items(), key=lambda item: extract_sort_key(item[0]))

# Write final JSON
final_output = {k: v for k, v in sorted_items}

with open('image_rotations_HE.json', 'w') as out:
    json.dump(final_output, out, indent=4)
