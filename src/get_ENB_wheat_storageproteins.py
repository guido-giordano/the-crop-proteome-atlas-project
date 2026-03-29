# %% [markdown]
# # Extract common FASTA IDs
# This notebook extracts sequence IDs from the storage list (science 2018) and look for them in the ENB wheat reference proteome

# %%
from pathlib import Path
import re
import pandas as pd

# --- Paths ---
ref_dir = Path('/home/ggiordano/snap/storage_proteins_project_ENB/import/reference_proteomes')
inventory_fa = Path('/home/ggiordano/snap/storage_proteins_project_ENB/import/inventory/storage_inventory_science2018.fa')

out_dir = Path('/home/ggiordano/snap/storage_proteins_project_ENB/data/storage_proteins_ID_extraction')
out_dir.mkdir(parents=True, exist_ok=True)

wheat_ref_fa = ref_dir / 'Triticum_aestivum_julius.PGSBv2.1.pep.all.fa'

# --- Regex ---
# canonical Julius gene ID
canonical_jul_re = re.compile(r'(TraesJUL[0-9][A-Z]?[0-9]{2}G[0-9]+)')
# any JUL identifier (used to capture non-canonical ones)
any_jul_re = re.compile(r'(TraesJUL[^ \t>]*)')

def extract_inventory_jul_ids(fasta_path):
    """
    Extract a single vector of JUL IDs from the inventory:
    - canonical IDs normalized (TraesJULxxxxGxxxxxx)
    - non-canonical JUL entries kept as-is
    - duplicates removed
    """
    ids = set()

    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # canonical first
                m = canonical_jul_re.search(line)
                if m:
                    ids.add(m.group(1))
                else:
                    # fallback: non-canonical JUL entry
                    m2 = any_jul_re.search(line)
                    if m2:
                        ids.add(m2.group(1))

    return sorted(ids)

def extract_wheat_canonical_jul_ids(fasta_path):
    """
    Extract canonical Julius gene IDs from the ENB wheat proteome.
    """
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                m = canonical_jul_re.search(line)
                if m:
                    ids.add(m.group(1))
    return sorted(ids)

# --- Extract IDs ---
inventory_ids = extract_inventory_jul_ids(inventory_fa)
wheat_ids = extract_wheat_canonical_jul_ids(wheat_ref_fa)

# --- Intersection (only meaningful for canonical IDs) ---
matching_ids = sorted(set(inventory_ids) & set(wheat_ids))

# --- Write CSVs ---
inventory_csv = out_dir / 'inventory_JUL_IDs_including_noncanonical.csv'
wheat_csv = out_dir / 'wheat_ENB_canonical_TraesJUL_IDs.csv'
matching_csv = out_dir / 'inventory_vs_wheat_matching_JUL_IDs.csv'

pd.DataFrame({"JUL_ID": inventory_ids}).to_csv(inventory_csv, index=False)
pd.DataFrame({"JUL_ID": wheat_ids}).to_csv(wheat_csv, index=False)
pd.DataFrame({"JUL_ID": matching_ids}).to_csv(matching_csv, index=False)

# --- Summary ---
print("=== JUL ID EXTRACTION SUMMARY ===")
print(f"Inventory JUL IDs (incl. non-canonical): {len(inventory_ids)}")
print(f"Wheat ENB canonical JUL IDs           : {len(wheat_ids)}")
print(f"Intersection                          : {len(matching_ids)}")

print("\nFiles written to:")
print(inventory_csv)
print(wheat_csv)
print(matching_csv)


# %%



