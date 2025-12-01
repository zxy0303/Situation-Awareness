import pandas as pd
import json

print("--- Starting combined 2-step process ---")

# --- Step 1: Load JSON features and original CSV ---
json_file_path = 'selected_features_structured.json'
csv_file_path = 'cluster_profiles_for_llm_FULL.csv'

# Load JSON to get base features
with open(json_file_path, 'r') as f:
    feature_config = json.load(f)

base_features = set()
for category in feature_config:
    for sub_category in category.get('sub_categories', []):
        for feature in sub_category.get('features', []):
            base_features.add(feature)
print(f"Loaded {len(base_features)} unique base features from JSON.")

# Load the original CSV
df = pd.read_csv(csv_file_path)
all_csv_columns = df.columns.tolist()
print(f"Loaded original CSV with {len(all_csv_columns)} total columns.")

# --- Step 2: Build column list and filter DataFrame in memory ---
columns_to_select = ['category_id']
suffixes = ['_mean', '_std', '_max', '_min']

for base in base_features:
    if base == 'SOC_drop_rate':
        if 'SOC_drop_rate' in all_csv_columns:
            columns_to_select.append('SOC_drop_rate')
    else:
        for suffix in suffixes:
            col_name = f"{base}{suffix}"
            if col_name in all_csv_columns:
                columns_to_select.append(col_name)

print(f"Found {len(columns_to_select)} matching columns to extract.")

# Filter the DataFrame in memory
df_filtered = df[columns_to_select]
print("DataFrame filtered in memory.")

# --- Step 3: Convert the filtered DataFrame to the desired JSON format ---
all_clusters_data = []

# Iterate over each row (each cluster) in the filtered DataFrame
for index, row in df_filtered.iterrows():
    # Create the base structure for this cluster
    cluster_object = {
        "Cluster_ID": int(row['category_id']),
        "Feature_Summary": {}
    }

    # Get all feature values for this row (excluding 'category_id')
    feature_summary = row.drop('category_id').to_dict()

    # Assign the feature summary to the cluster object
    cluster_object["Feature_Summary"] = feature_summary

    # Add this cluster's data to the main list
    all_clusters_data.append(cluster_object)

print("JSON structure created in memory.")

# --- Step 4: Save and print the final JSON output ---
# Convert the list of dictionaries to a JSON formatted string
json_output = json.dumps(all_clusters_data, indent=2)

# Define a new file to save the JSON output
output_json_path = 'cluster_summaries_combined.json'

# Save the JSON string to the file
with open(output_json_path, 'w') as f:
    f.write(json_output)

# Print the final JSON output to be displayed
print("\n--- Generated JSON Output (Combined Script) ---")
print(json_output)
print(f"\nJSON data successfully saved to '{output_json_path}'.")