import pandas as pd
import os
import requests
from io import StringIO

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download QM9 dataset directly
print("Downloading QM9 dataset...")
qm9_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
response = requests.get(qm9_url)
df_qm9 = pd.read_csv(StringIO(response.text))

# Save QM9 dataset
print(f"Processing QM9 dataset...")
# Ensure it has the expected columns
if 'smiles' not in df_qm9.columns and 'mol_id' in df_qm9.columns:
    # If 'smiles' column is missing but there's a different column with SMILES
    possible_smiles_columns = ['mol_id', 'smiles', 'SMILES', 'canonical_smiles']
    for col in possible_smiles_columns:
        if col in df_qm9.columns:
            df_qm9['smiles'] = df_qm9[col]
            break

# Save to CSV
df_qm9.to_csv("data/qm9.csv", index=False)
print(f"QM9 dataset saved to data/qm9.csv with {len(df_qm9)} molecules")

# Download ESOL dataset directly
print("Downloading ESOL dataset...")
esol_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
response = requests.get(esol_url)
df_esol = pd.read_csv(StringIO(response.text))

# Save ESOL dataset
print(f"Processing ESOL dataset...")
# Ensure it has the expected columns
if 'smiles' not in df_esol.columns and 'SMILES' in df_esol.columns:
    df_esol['smiles'] = df_esol['SMILES']
if 'logSolubility' not in df_esol.columns and 'measured log solubility in mols per litre' in df_esol.columns:
    df_esol['logSolubility'] = df_esol['measured log solubility in mols per litre']

# Save to CSV
df_esol.to_csv("data/esol.csv", index=False)
print(f"ESOL dataset saved to data/esol.csv with {len(df_esol)} molecules")