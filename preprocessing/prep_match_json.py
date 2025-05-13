import json
import os
from collections import defaultdict

def is_empty(value):
    return value in (None, "", {}, [])

def merge_json_files(json_files):
    merged = {}
    seen_conflicts = set()

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for key, value in data.items():
            if key not in merged:
                merged[key] = value
            else:
                # Conflict: key exists already
                seen_conflicts.add(key)
                existing_value = merged[key]

                if is_empty(existing_value) and not is_empty(value):
                    merged[key] = value
                elif not is_empty(existing_value) and is_empty(value):
                    continue  # Keep existing
                elif not is_empty(existing_value) and not is_empty(value):
                    # Both non-empty: choose the one with more keys if dict
                    if isinstance(existing_value, dict) and isinstance(value, dict):
                        if len(value) > len(existing_value):
                            merged[key] = value

    # Print conflicts
    if seen_conflicts:
        print("Conflicting keys found and resolved:")
        for key in sorted(seen_conflicts):
            print(f" - {key}")
    
    return merged

def strip_path(merged):
    for case, matches in merged.items():
        if isinstance(matches, dict):
            for match_nr, stain in matches.items():
                for stain_name, filename in stain.items():
                    # Strip the path to get just the filename
                    
                    filename = filename.split("/")[-1]
                    merged[case][match_nr][stain_name] = filename

    merged = dict(sorted(merged.items()))  # Sort the dictionary by keys

    return merged

if __name__ == "__main__":
    json_files = [
        'image_matches_new.json',
        'new_image_matches.json'
    ]
    output_path='image_matches.json'

    merged = merge_json_files(json_files)
    merged_stripped = strip_path(merged)

    # Save merged result
    with open(output_path, 'w') as out_file:
        json.dump(merged_stripped, out_file, indent=4)
