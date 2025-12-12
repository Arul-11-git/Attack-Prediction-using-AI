import pandas as pd
import pandas as pd
import numpy as np

# -----------------------------
# 1. Load both datasets
# -----------------------------
old_df = pd.read_csv("final_cleaned_cve_dataset.csv")
new_df = pd.read_csv("parsed_nvd_2024.csv")

# -----------------------------
# 2. Standardize column names for the 2024 dataset
# -----------------------------
new_df = new_df.rename(columns={
    "cve_id": "cve_id",
    "description": "description",
    "severity": "severity",
    "impact_score": "impact_score",
    "attack_vector": "attack_vector",
    "attack_complexity": "attack_complexity",
    "privileges_required": "privileges_required"
})

# Ensure all required columns exist
required_cols = [
    "cve_id",
    "description",
    "severity",
    "impact_score",
    "attack_vector",
    "attack_complexity",
    "privileges_required"
]

for col in required_cols:
    if col not in new_df.columns:
        new_df[col] = np.nan

# -----------------------------
# 3. Clean missing CVSS fields in 2024 dataset
# -----------------------------
new_df["attack_vector"] = new_df["attack_vector"].fillna("unknown")
new_df["attack_complexity"] = new_df["attack_complexity"].fillna("unknown")
new_df["privileges_required"] = new_df["privileges_required"].fillna("unknown")
new_df["impact_score"] = new_df["impact_score"].fillna(0)

# -----------------------------
# 4. Concatenate datasets
# -----------------------------
combined_df = pd.concat([old_df, new_df], ignore_index=True)

# -----------------------------
# 5. Clean final dataset
# -----------------------------
# Remove rows missing important fields
combined_df = combined_df[
    combined_df["description"].notna() &
    combined_df["severity"].notna()
]

combined_df = combined_df[combined_df["description"].str.len() > 10]

# Fill any remaining missing CVSS fields
combined_df["attack_vector"] = combined_df["attack_vector"].fillna("unknown")
combined_df["attack_complexity"] = combined_df["attack_complexity"].fillna("unknown")
combined_df["privileges_required"] = combined_df["privileges_required"].fillna("unknown")
combined_df["impact_score"] = combined_df["impact_score"].fillna(0)

# -----------------------------
# 6. Save final combined dataset
# -----------------------------
combined_df.to_csv("combined_cve_dataset.csv", index=False)

print("✅ Combined dataset created successfully!")
print("Total rows:", len(combined_df))
print("Saved as combined_cve_dataset.csv")


"""

import pandas as pd

df = pd.read_csv("cleaned_cve_dataset.csv")

# Fill missing categorical fields
df["attack_vector"] = df["attack_vector"].fillna("unknown")
df["attack_complexity"] = df["attack_complexity"].fillna("unknown")
df["privileges_required"] = df["privileges_required"].fillna("unknown")

# Fill missing impact score with 0
df["impact_score"] = df["impact_score"].fillna(0)

# Save final fully usable dataset
df.to_csv("final_cleaned_cve_dataset.csv", index=False)

print("✅ Final cleaned dataset saved as final_cleaned_cve_dataset.csv")
print("Total rows:", len(df))
"""

