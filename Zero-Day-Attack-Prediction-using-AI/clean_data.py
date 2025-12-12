import pandas as pd
import numpy as np

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("cve_data.csv", engine='python', on_bad_lines='skip')

# ------------------------------
# 2. Rename Columns to Match 2024 Template
# ------------------------------
df = df.rename(columns={
    "details": "description",
    "Impact": "impact_score",
    "AttackVector": "attack_vector",
    "complexity": "attack_complexity",
    "requiredPrivilege": "privileges_required",
    "CVE": "cve_id"
})

# Ensure all required columns exist
required_cols = ["cve_id", "description", "Severity", "impact_score",
                 "attack_vector", "attack_complexity", "privileges_required"]

for col in required_cols:
    if col not in df.columns:
        df[col] = np.nan  # add missing column

# ------------------------------
# 3. Option A: Keep only rows with Severity
# ------------------------------
df = df[df["Severity"].notna()]

# ------------------------------
# 4. Normalize Severity Values
# ------------------------------
df["Severity"] = df["Severity"].astype(str).str.lower().str.strip()

severity_map = {
    "critical": "critical",
    "crit": "critical",
    "c": "critical",

    "high": "high",
    "h": "high",

    "medium": "medium",
    "med": "medium",
    "m": "medium",

    "low": "low",
    "l": "low"
}

df["severity"] = df["Severity"].map(severity_map)

# Remove rows where severity is unknown after mapping
df = df[df["severity"].notna()]

# ------------------------------
# 5. Clean & Normalize attack_vector
# ------------------------------
df["attack_vector"] = df["attack_vector"].astype(str).str.lower()

map_av = {
    "network": "network", "n": "network",
    "adjacent": "adjacent", "a": "adjacent",
    "local": "local", "l": "local",
    "physical": "physical", "p": "physical"
}

df["attack_vector"] = df["attack_vector"].map(map_av)

# ------------------------------
# 6. Normalize attack_complexity
# ------------------------------
df["attack_complexity"] = df["attack_complexity"].astype(str).str.lower()

map_ac = {
    "low": "low", "l": "low",
    "high": "high", "h": "high"
}

df["attack_complexity"] = df["attack_complexity"].map(map_ac)

# ------------------------------
# 7. Normalize privileges_required
# ------------------------------
df["privileges_required"] = df["privileges_required"].astype(str).str.lower()

map_pr = {
    "none": "none", "n": "none",
    "low": "low", "l": "low",
    "high": "high", "h": "high"
}

df["privileges_required"] = df["privileges_required"].map(map_pr)

# ------------------------------
# 8. Clean Description Text
# ------------------------------
df["description"] = df["description"].astype(str).str.strip()

df = df[df["description"].str.len() > 10]   # remove junk descriptions

# ------------------------------
# 9. Final Dataset Columns
# ------------------------------
final_df = df[[
    "cve_id",
    "description",
    "severity",
    "impact_score",
    "attack_vector",
    "attack_complexity",
    "privileges_required"
]]

# ------------------------------
# 10. Save Cleaned Dataset
# ------------------------------
final_df.to_csv("cleaned_cve_dataset.csv", index=False)
print("✅ Dataset cleaned and saved as cleaned_cve_dataset.csv")
print("Total rows after cleaning:", len(final_df))


