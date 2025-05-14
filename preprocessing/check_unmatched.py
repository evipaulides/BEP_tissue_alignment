import json

# ==== CONFIG ====
json_path = "data/image_matches/image_matches.json"  # Path to your JSON file
# ================

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Track incomplete matches
incomplete_matches = []

# Iterate through cases and matches
for case_id, matches in data.items():
    if isinstance(matches, str) and matches == "NO_MATCHES":
        continue

    for match_id, stains in matches.items():
        has_he = "he" in stains
        has_ihc = "ihc" in stains

        if not (has_he and has_ihc):
            incomplete_matches.append({
                "case_id": case_id,
                "match_id": match_id,
                "has_he": has_he,
                "has_ihc": has_ihc,
                "data": stains
            })

# Print the results
if incomplete_matches:
    print("Found matches with only one stain (either HE or IHC):\n")
    for match in incomplete_matches:
        status = "only HE" if match["has_he"] else "only IHC"
        print(f"Case {match['case_id']} - Match {match['match_id']}: {status}")
        print(f"  Data: {match['data']}")
else:
    print("All matches contain both HE and IHC images.")
