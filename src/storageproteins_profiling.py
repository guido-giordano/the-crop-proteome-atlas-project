# %% [markdown]
# # Storage Proteins – Crop Proteome Atlas (ENB)
# 
# Analysis of proteomics search results from FragPipe for multiple crop species.
# 
# **Author:** Guido Giordano
# 

# %%
# =============================================================================
# Storage Proteins – Crop Proteome Atlas (ENB)
# Project setup & environment
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Project root (notebook lives in /script)
BASE_DIR = Path.cwd().parent

IMPORT_DIR = BASE_DIR / "import"
EXPORT_DIR = BASE_DIR / "export"
EXPORT_MOD_DIR = EXPORT_DIR / "search_results_mod"

EXPORT_MOD_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("IMPORT_DIR:", IMPORT_DIR)
print("EXPORT_MOD_DIR:", EXPORT_MOD_DIR)


# %%
# =============================================================================
# Load proteomics search results
# =============================================================================

SEARCH_RESULTS_DIR = IMPORT_DIR / "search_results"

search_results = {}

for file_path in SEARCH_RESULTS_DIR.glob("*.tsv"):
    sp = file_path.stem.replace("combined_protein.", "")
    search_results[sp] = pd.read_csv(file_path, sep="\t", low_memory=False)
    print(f"[OK] Loaded {sp}: {search_results[sp].shape}")

print(f"\nTotal species loaded: {len(search_results)}")


# %%
# =============================================================================
# Remove decoys and contaminants
# =============================================================================

def filter_decoys(df):
    df = df[~df["Protein"].str.contains("rev_", case=False, na=False)]
    df = df[~df["Protein"].str.contains("sp\\|", case=False, na=False)]
    return df

for sp in search_results:
    before = len(search_results[sp])
    search_results[sp] = filter_decoys(search_results[sp])
    after = len(search_results[sp])
    print(f"{sp}: {before} → {after} (removed {before - after})")


# %%
# ============================================================================
# 4. TISSUE MAPPING: Map plate numbers to tissue names and update columns
# ============================================================================

# --- Tissue mapping dictionary (from user-provided info) ---
species_plate_tissue = {
    'Zm': [
        ('P090741', 'primary root'), ('P090742', 'seminal root'), ('P090743', 'crown root'), ('P090744', 'coleoptile'),
        ('P090745', 'mesocotyl'), ('P090746', 'leaf 3 blade'), ('P090747', 'leaf 3 sheath'), ('P090748', 'first elongated internode'),
        ('P090749', 'leaf 8 blade'), ('P090750', 'cob leaf'), ('P090751', 'immature tassel'), ('P090752', 'meiotic tassel'),
        ('P090753', 'immature cob'), ('P090754', 'pre-pollination cob'), ('P090755', 'silk'), ('P090756', 'pollen'), ('P104005', 'mature seed')
    ],
    'De': [
        ('P097949', 'Seed, dry'), ('P097950', 'Seed imbibed'), ('P097951', 'Seedling, cotyledon'), ('P097952', 'Seedling, hypocotyl'),
        ('P097953', 'Seedling, root'), ('P097954', 'Seedling, 1st leaf'), ('P097955', 'Seedling, 2nd leaf'), ('P097956', 'Adult plant, primary leaf'),
        ('P097957', 'Adult plant, secondary leaf'), ('P100314', 'Adult plant, tertiary leaf'), ('P100315', 'Adult plant, roots'),
        ('P097958', 'Adult plant, stem, node'), ('P097959', 'Adult plant, stem, internode'), ('P097960', 'Seedling shoot'),
        ('P097961', 'Whole seedling'), ('P097962', 'Green leaf'), ('P097963', 'Undeveloped flowers/before emergence'), ('P097964', 'Flag leaf')
    ],
    'Ca': [
        ('P092517', 'Embryogenic Calli'), ('P092518', 'Non-Embryogenic Calli'), ('P092519', 'Seedling Roots'), ('P092520', 'Seedling aerial parts'),
        ('P092521', 'Primary Root'), ('P092522', 'Crown Root'), ('P092523', 'Young leaf'), ('P092524', 'Stem node'),
        ('P092525', 'Stem internode'), ('P092526', 'Immature flower'), ('P092527', 'Mature inflorescence (Pistil)'), ('P092528', 'Stamen'),
        ('P092529', 'Pollen'), ('P092530', 'Mature seed'), ('P092531', 'Germinating seed'), ('P092532', 'Stem')
    ],
    'Hv': [
        ('P095141', 'Primary leaf'), ('P095142', 'Secondary leaf'), ('P095143', 'Young root'), ('P095144', 'Node'),
        ('P095145', 'Internode'), ('P095146', 'Adult root (before ears)'), ('P095147', 'Adult root (with tiny ears)'),
        ('P095148', 'Adult root (3 weeks before seed harvest)'), ('P095149', 'Anther (before pollen shedding)'),
        ('P095150', 'Anther (after pollen shedding)'), ('P095151', 'Stigma, style and ovary (before pollination)'),
        ('P095152', 'Stigma, style and ovary (after pollination)'), ('P095153', 'immature seed (short time after flowering)'),
        ('P095154', 'immature seed (3 weeks before seed harvest)'), ('P095155', 'mature seed')
    ],
    'Os': [
        ('P091307', 'seed dry'), ('P091308', 'seed 24 h embibed'), ('P091309', 'seed 48h embryo'), ('P091310', 'seed 48h endosperm'),
        ('P091311', 'seeding 72h coleoptile dark'), ('P091312', 'seedling 96h coleoptile light'), ('P091313', 'seedling 72h under water'),
        ('P091314', 'seedling 72h root dark'), ('P091315', 'Adult plant young root whole'), ('P091316', 'Adult plant old root lower part'),
        ('P091317', 'Adult plant old root upper part'), ('P091318', 'Adult plant old root whole'), ('P091319', 'Adult plant leaf blade'),
        ('P091320', 'Adult plant leaf blade (hydroponics)'), ('P091321', 'Adult plant leaf sheath'),
        ('P091322', 'Adult plant leaf sheath (hydroponics)'), ('P091323', 'Adult plant node'), ('P091324', 'Adult plant internode'),
        ('P091325', 'Adult plant flower young'), ('P091326', 'Adult plant flower stalk young'), ('P091327', 'Adut plant seed young'),
        ('P091328', 'Adult plant flower male part (incl. pollen)'), ('P091329', 'Adult plant flower male part IR64'),
        ('P091330', 'Adult plant flower male part Nipponbare')
    ],
    'Sb': [
        ('P094332', 'Primary leaf'), ('P094333', 'Secondary leaf'), ('P094334', 'Young root'), ('P094335', 'Node'),
        ('P094336', 'Internode'), ('P094337', 'Adult root (before ears)'), ('P094338', 'Adult root (with tiny ears)'),
        ('P094339', 'Adult root (3 weeks before seed harvest)'), ('P094340', 'Anther (before pollen shedding)'),
        ('P094341', 'Anther (after pollen shedding)'), ('P094342', 'Stigma, style and ovary (before pollination)'),
        ('P094343', 'Stigma, style and ovary (after pollination)'), ('P094344', 'immature seed (short time after flowering)'),
        ('P094345', 'mature seed')
    ],
    'Sc': [
        ('P100625', 'Primary leaf'), ('P100626', 'Secondary leaf'), ('P100627', 'Young root'), ('P100628', 'Node'),
        ('P100629', 'Internode'), ('P100630', 'Adult root (before ears)'), ('P100631', 'Adult root (with tiny ears) (ad_ro-flow)'),
        ('P100632', 'Adult root (3 weeks before seed harvest)'), ('P100633', 'Anther (before pollen shedding)'),
        ('P100634', 'Anther (after pollen shedding)'), ('P100635', 'Stigma, style and ovary (before pollination)'),
        ('P100636', 'Stigma, style and ovary (after pollination)'), ('P100637', 'immature seed (short time after flowering)'),
        ('P100638', 'immature seed (3 weeks before seed harvest)'), ('P100639', 'mature seed')
    ],
    'Ta': [
        ('P093501', 'Primary leaf'), ('P093502', 'Secondary leaf'), ('P093503', 'Young root'), ('P093504', 'Node'),
        ('P093505', 'Internode'), ('P093506', 'Adult root (before ears)'), ('P093507', 'Adult root (with tiny ears)'),
        ('P093508', 'Adult root (3 weeks before seed harvest)'), ('P093509', 'Anther (before pollen shedding)'),
        ('P093510', 'Anther (after pollen shedding)'), ('P093511', 'Pollen'), ('P093512', 'Stigma, style and ovary (before pollination)'),
        ('P093513', 'Stigma, style and ovary (after pollination)'), ('P093514', 'immature seed (short time after flowering)'),
        ('P093515', 'immature seed (3 weeks before seed harvest)'), ('P093516', 'mature seed')
    ]
}

plate_to_tissue = {
    sp: dict(pairs) for sp, pairs in species_plate_tissue.items()
}

updated_search_results = {}

for sp, df in search_results.items():
    mapping = plate_to_tissue.get(sp, {})
    new_columns = {}

    for col in df.columns:
        match = re.match(r"(P\d+)_\d+ MaxLFQ Intensity", col)
        if match:
            plate = match.group(1)
            tissue = mapping.get(plate)
            if tissue:
                new_columns[col] = f"{tissue} ({plate}) MaxLFQ Intensity"

    if new_columns:
        df = df.rename(columns=new_columns)

    updated_search_results[sp] = df
    print(f"{sp}: renamed {len(new_columns)} columns")

search_results = updated_search_results

# %%
# =============================================================================
# Export modified search results
# =============================================================================

for sp, df in search_results.items():
    out_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Exported {out_path.name}")


# %%
# =============================================================================
# Step 6: Keep Protein / Protein ID / MaxLFQ Intensity columns only
# =============================================================================

for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"

    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    keep_cols = [
        c for c in df.columns
        if c in ["Protein", "Protein ID"] or "MaxLFQ Intensity" in c
    ]

    df = df[keep_cols]
    df.to_csv(mod_path, sep="\t", index=False)

    print(f"{sp}: kept {len(keep_cols)} columns")


# %%
# =============================================================================
# Step 7: Remove rows where all MaxLFQ values are zero
# =============================================================================

for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]

    df[maxlfq_cols] = df[maxlfq_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    all_zero = (df[maxlfq_cols] == 0).all(axis=1)

    print(f"{sp}: removing {all_zero.sum()} / {len(df)} proteins")

    df = df.loc[~all_zero]
    df.to_csv(mod_path, sep="\t", index=False)


# %%
# =============================================================================
# Step 8: Log2(x + 1) transformation
# =============================================================================

for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]

    df[maxlfq_cols] = np.log2(
        df[maxlfq_cols].apply(pd.to_numeric, errors="coerce").fillna(0) + 1
    )

    df.to_csv(mod_path, sep="\t", index=False)
    print(f"{sp}: log2 transform applied")


# %%
# =============================================================================
# Step 8.1: Heatmaps of storage-associated proteins (orthologues, per species)
# Raw log2(MaxLFQ + 1) values
# Fixed square cells, vertical column labels, adaptive figure size
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# Mapping: proteomics species code -> storage BLAST filename tag
# -----------------------------------------------------------------------------
SPECIES_TO_STORAGE_TAG = {
    "Ta": "Triticum_aestivum",
    "Hv": "Hordeum_vulgare",
    "Os": "Oryza_sativa",
    "Zm": "Zea_mays",
    "Sb": "Sorghum_bicolor",
    "Sc": "Secale.cereale",
    "De": "Digitaria_exilis",
    "Ca": "Cenchrus_americanus",
}

# -----------------------------------------------------------------------------
# Mapping: proteomics species code -> common species name (for titles)
# -----------------------------------------------------------------------------
SPECIES_TO_COMMON_NAME = {
    "Ta": "Wheat",
    "Hv": "Barley",
    "Os": "Rice",
    "Zm": "Maize",
    "Sb": "Sorghum",
    "Sc": "Rye",
    "De": "Fonio",
    "Ca": "Pearl millet",
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HEATMAP_OUT_DIR = Path(
    "/home/ggiordano/snap/storage_proteins_project_ENB/export/"
    "heatmaps_storage_proteins_orthologues"
)
HEATMAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_LIST_DIR = Path(
    "/home/ggiordano/snap/storage_proteins_project_ENB/data/"
    "lists_orthologues_8proteomes"
)

# -----------------------------------------------------------------------------
# Fixed square cell size (inches)
# -----------------------------------------------------------------------------
CELL_SIZE = 0.35

# -----------------------------------------------------------------------------
# White → yellow → red palette
# -----------------------------------------------------------------------------
w_y_r = LinearSegmentedColormap.from_list(
    "white_yellow_red",
    ["#FFFFFF", "#FEE8C8", "#D73000"]
)

print("Heatmaps will be saved to:")
print(HEATMAP_OUT_DIR)

# -----------------------------------------------------------------------------
# Loop over species
# -----------------------------------------------------------------------------
for sp in search_results:
    print("\n============================================")
    print(f"Processing species: {sp}")

    if sp not in SPECIES_TO_STORAGE_TAG:
        print("❌ Species not in mapping, skipping.")
        continue

    storage_tag = SPECIES_TO_STORAGE_TAG[sp]
    common_name = SPECIES_TO_COMMON_NAME.get(sp, sp)

    # -------------------------------------------------------------------------
    # Load proteomics table
    # -------------------------------------------------------------------------
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    if not mod_path.exists():
        print("❌ Proteomics file not found, skipping.")
        continue

    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    # -------------------------------------------------------------------------
    # Load storage-associated protein list
    # -------------------------------------------------------------------------
    storage_lists = list(
        STORAGE_LIST_DIR.glob(f"*{storage_tag}*_associated_proteins_pident20.csv")
    )
    if not storage_lists:
        print(f"❌ No storage list found for {storage_tag}")
        continue

    storage_ids = set(
        pd.read_csv(storage_lists[0])["protein_id"].astype(str)
    )

    # -------------------------------------------------------------------------
    # Protein ID column
    # -------------------------------------------------------------------------
    if "Protein ID" in df.columns:
        pid_col = "Protein ID"
    elif "Protein" in df.columns:
        pid_col = "Protein"
    else:
        print("❌ No protein ID column found.")
        continue

    df_sub = df[df[pid_col].astype(str).isin(storage_ids)]
    if df_sub.empty:
        print("⚠️ No overlapping proteins → skipping.")
        continue

    # -------------------------------------------------------------------------
    # Extract & clean MaxLFQ columns
    # -------------------------------------------------------------------------
    maxlfq_cols = [c for c in df_sub.columns if "MaxLFQ Intensity" in c]
    if not maxlfq_cols:
        print("❌ No MaxLFQ columns found.")
        continue

    df_sub = df_sub.rename(
        columns={c: c.replace(" MaxLFQ Intensity", "") for c in maxlfq_cols}
    )
    maxlfq_cols = [c.replace(" MaxLFQ Intensity", "") for c in maxlfq_cols]

    mat = df_sub.set_index(pid_col)[maxlfq_cols]

    # -------------------------------------------------------------------------
    # Figure size derived from number of cells
    # -------------------------------------------------------------------------
    n_rows, n_cols = mat.shape
    fig_width  = CELL_SIZE * n_cols + 2.5
    fig_height = CELL_SIZE * n_rows + 2.2  # slightly taller for small n_rows

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        constrained_layout=False
    )

    # -------------------------------------------------------------------------
    # Heatmap
    # -------------------------------------------------------------------------
    sns.heatmap(
        mat,
        ax=ax,
        cmap=w_y_r,
        square=True,
        linewidths=0.2,
        linecolor="lightgrey",
        cbar_kws={
            "label": "log2MaxLFQ",
            "shrink": 0.6,
            "aspect": 25
        },
        xticklabels=True,
        yticklabels=True
    )

    # -------------------------------------------------------------------------
    # Axis formatting
    # -------------------------------------------------------------------------
    ax.set_title(
        f"{common_name}",
        pad=20
    )

    ax.set_xlabel("Tissue", labelpad=25)
    ax.set_ylabel("Protein")

    ax.tick_params(
        axis="x",
        rotation=90,
        labelsize=8,
        pad=8
    )
    ax.tick_params(axis="y", labelsize=8)

    plt.subplots_adjust(
        left=0.25,
        bottom=min(0.55, 0.18 + 0.025 * n_cols),
        right=0.95,
        top=0.93
    )

    # -------------------------------------------------------------------------
    # Save (content-aware bounding box → NOTHING gets cut)
    # -------------------------------------------------------------------------
    out_file = HEATMAP_OUT_DIR / f"{sp}_storage_orthologue_heatmap.png"
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3
    )
    plt.close()

    print(f"[SAVED] {out_file}")

print("\n✅ All heatmaps generated with identical parameters and no clipping.")


# %%
# =============================================================================
# Step 9: Normalisation
# Mean scaling around the grand mean (excluding zeros)
# =============================================================================

print("\n--- Normalisation: mean scaling around grand mean (excluding zeros) ---")

# Store column means before and after normalisation
col_means_before = {}
col_means_after = {}
maxlfq_cols_by_species = {}

# -------------------------------------------------------------------------
# 1. Compute column means (excluding zeros)
# -------------------------------------------------------------------------
for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]
    maxlfq_cols_by_species[sp] = maxlfq_cols

    for col in maxlfq_cols:
        mean_val = df[col].replace(0, np.nan).mean()
        col_means_before[(sp, col)] = mean_val

# -------------------------------------------------------------------------
# 2. Compute grand mean across all columns
# -------------------------------------------------------------------------
grand_mean = np.nanmean(list(col_means_before.values()))
print(f"Grand mean (excluding zeros): {grand_mean:.4f}")

# -------------------------------------------------------------------------
# 3. Compute correction factors
# -------------------------------------------------------------------------
correction_factors = {
    (sp, col): (grand_mean / mean if mean and not np.isnan(mean) else 1.0)
    for (sp, col), mean in col_means_before.items()
}

# -------------------------------------------------------------------------
# 4. Apply correction factors (only to non-zero values)
# -------------------------------------------------------------------------
for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    for col in maxlfq_cols_by_species[sp]:
        factor = correction_factors[(sp, col)]
        df[col] = df[col].where(df[col] == 0, df[col] * factor)

    df.to_csv(mod_path, sep="\t", index=False)

    # Store post-normalisation means
    for col in maxlfq_cols_by_species[sp]:
        col_means_after[(sp, col)] = df[col].replace(0, np.nan).mean()

# -------------------------------------------------------------------------
# 5. Summary table
# -------------------------------------------------------------------------
print("\nSpecies\tColumn\tMean_before\tMean_after\tFactor")
for (sp, col), mean_before in col_means_before.items():
    mean_after = col_means_after[(sp, col)]
    factor = correction_factors[(sp, col)]
    print(
        f"{sp}\t{col.replace(' MaxLFQ Intensity','')}\t"
        f"{mean_before:.4f}\t{mean_after:.4f}\t{factor:.4f}"
    )


# %%
# =============================================================================
# Step 10: QC plots
# Mean log2(MaxLFQ+1) intensities per tissue (excluding zeros)
# =============================================================================

for sp in search_results:
    mod_path = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    df = pd.read_csv(mod_path, sep="\t", low_memory=False)

    maxlfq_cols = [c for c in df.columns if "MaxLFQ Intensity" in c]

    # Compute means excluding zeros
    means = df[maxlfq_cols].replace(0, np.nan).mean(axis=0)

    # Clean labels for plotting
    labels = [c.replace(" MaxLFQ Intensity", "") for c in maxlfq_cols]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels, means)
    plt.title(f"Mean log2(MaxLFQ+1) intensities (excluding zeros) – {sp}")
    plt.ylabel("Mean log2(MaxLFQ+1)")
    plt.xlabel("Tissue")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()

    # Save and show
    plot_path = EXPORT_MOD_DIR / f"mean_log2_maxlfq_{sp}.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"[OK] Plot saved: {plot_path}")


# %%
# =============================================================================
# Inspect N0 structure (no sensitive data exposed)
# =============================================================================

import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd().parent
N0_PATH = BASE_DIR / "import" / "N0.tsv"

N0 = pd.read_csv(N0_PATH, sep="\t", low_memory=False)

print("=== N0 OVERVIEW ===")
print("Shape:", N0.shape)
print()

print("=== COLUMN NAMES ===")
for i, c in enumerate(N0.columns):
    print(f"{i:02d}  {c}")
print()


print("=== FIRST 3 ROWS (raw strings) ===")
display(N0.head(3))

print("=== NON-NA COUNTS PER COLUMN ===")
display(N0.notna().sum())

print("=== EXAMPLE CELLS WITH MULTIPLE IDS ===")
for col in N0.columns:
    example = N0[col].dropna().astype(str)
    example = example[example.str.contains(r"[;,]|\s")]
    if not example.empty:
        print(f"\nColumn: {col}")
        print("Example value:", example.iloc[0])


# %%
# =============================================================================
# Step 11: Attach MaxLFQ intensities to N0 orthology table
# Many-to-many aware, lossless
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path.cwd().parent
EXPORT_MOD_DIR = BASE_DIR / "export" / "search_results_mod"
N0_PATH = BASE_DIR / "import" / "N0.tsv"

# Load N0
N0 = pd.read_csv(N0_PATH, sep="\t", low_memory=False)

# Species ↔ N0 column mapping
SPECIES_MAP = {
    "Hv": "Hordeum_vulgare_rgtplanet.Hvulgare_cv_RGT_Planet_BPGv2.pep.all",
    "Sc": "Secale.cereale.Lo7V3.pgsb.r2.Feb2025_all",
    "Ta": "Triticum_aestivum_julius.PGSBv2.1.pep.all",
    "Zm": "GCF_902167145.1_Zea_mays",
    "Os": "GCF_034140825.1_Oryza_sativa",
    "Sb": "GCF_000003195.3_Sorghum_bicolor",
    "Ca": "GCA_963924085.1_Cenchrus_americanus.helixer",
    "De": "GCA_015342445.1_Digitaria_exilis"
}

# Helper: split protein IDs safely
def split_ids(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]

# Process species one by one
for sp, n0_col in SPECIES_MAP.items():

    print(f"\n[INFO] Processing {sp}")

    prot_file = EXPORT_MOD_DIR / f"combined_protein.{sp}.tsv"
    if not prot_file.exists():
        print(f"[WARN] Missing proteomics file for {sp}, skipping")
        continue

    prot_df = pd.read_csv(prot_file, sep="\t", low_memory=False)

    maxlfq_cols = [c for c in prot_df.columns if "MaxLFQ Intensity" in c]

    # Build lookup: Protein ID → list of intensities per column
    prot_lookup = (
        prot_df
        .set_index("Protein ID")[maxlfq_cols]
        .apply(lambda r: r.tolist(), axis=1)
        .to_dict()
    )

    # --- 1. Create all new columns at once (object dtype) ---
    new_cols = {
        f"{sp}|{col}": pd.Series([np.nan] * len(N0), dtype="object")
        for col in maxlfq_cols
    }
    N0 = pd.concat([N0, pd.DataFrame(new_cols)], axis=1)

    # --- 2. Fill values ---
    for idx, cell in N0[n0_col].items():
        prot_ids = split_ids(cell)
        if not prot_ids:
            continue

        for i, col in enumerate(maxlfq_cols):
            values = []
            for pid in prot_ids:
                if pid in prot_lookup:
                    val = prot_lookup[pid][i]
                    if pd.notna(val) and val != 0:
                        values.append(f"{val:.6g}")

            if values:
                N0.at[idx, f"{sp}|{col}"] = ";".join(values)

    print(f"[OK] Intensities attached for {sp}")


# Save expanded N0
OUT_PATH = BASE_DIR / "export" / "N0_with_MaxLFQ.tsv"
N0.to_csv(OUT_PATH, sep="\t", index=False)

print("\n[FINAL] Expanded N0 saved to:", OUT_PATH)


# %%
# =============================================================================
# Step 12: Collapse multi-valued cells to mean intensity
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

def mean_from_semicolon(cell):
    """
    Convert a semicolon-separated string of numbers into their mean.
    Returns NaN if input is empty or invalid.
    """
    if pd.isna(cell):
        return np.nan

    if isinstance(cell, str):
        try:
            values = [float(x) for x in cell.split(";") if x.strip() != ""]
            return np.mean(values) if values else np.nan
        except ValueError:
            return np.nan

    if isinstance(cell, (int, float)):
        return float(cell)

    return np.nan


# Identify all intensity columns (species|tissue)
intensity_cols = [c for c in N0.columns if "|" in c]

print(f"[INFO] Collapsing {len(intensity_cols)} intensity columns to mean values")

# Apply transformation column-wise
N0_mean = N0.copy()

for col in intensity_cols:
    N0_mean[col] = N0_mean[col].apply(mean_from_semicolon)

print("[OK] Mean-collapsing completed")


# =============================================================================
# EXPORT mean-collapsed matrix
# =============================================================================

EXPORT_DIR = Path.cwd().parent / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

out_path = EXPORT_DIR / "N0_quant_mean.tsv"
N0_mean.to_csv(out_path, sep="\t", index=False)

print(f"[SAVED] Mean-collapsed orthology matrix written to:\n{out_path}")


# =============================================================================
# Optional sanity check
# =============================================================================
print("\nExample before / after:")
example_col = intensity_cols[0]

display(
    pd.DataFrame({
        "original": N0[example_col].head(5),
        "mean": N0_mean[example_col].head(5)
    })
)



# %%
#Load the vector with enb wheat proteins to show them in an heatmap 
from pathlib import Path
import pandas as pd

id_path = Path.cwd().parent / "data/orthogroups_stats/ENB_storage_hits_pident10.txt"
enb_ids = pd.read_csv(id_path, header=None)[0].astype(str).tolist()
enb_set = set(enb_ids)

print(f"[INFO] Loaded {len(enb_set)} ENB wheat protein IDs")


# %%
#Normalise entries so that we find all of them 
import re

def canonical_wheat_id(x: str) -> str:
    """
    Normalize wheat IDs across pipelines.
    Keeps: TraesJULxYzzG########.n
    Removes: .CDS1, descriptions, whitespace
    """
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = x.split()[0]          # remove descriptions
    x = x.replace(".CDS1", "")  # drop CDS suffix
    return x
enb_set = set(canonical_wheat_id(x) for x in enb_ids_raw if canonical_wheat_id(x))

print("[INFO] Canonical ENB IDs:", len(enb_set))


# %%
#use the vector of enb wheat proteins and make an heatmap 
WHEAT_COL = "Triticum_aestivum_julius.PGSBv2.1.pep.all"

def wheat_cell_contains_enb(cell, enb_set):
    if pd.isna(cell):
        return False
    ids = [canonical_wheat_id(x) for x in str(cell).split(",")]
    return any(i in enb_set for i in ids if i)

mask = N0_mean[WHEAT_COL].apply(
    lambda x: wheat_cell_contains_enb(x, enb_set)
)

N0_sel = N0_mean[mask]

print(f"[INFO] Selected {N0_sel.shape[0]} orthogroups containing ENB storage proteins")


# %%
#generate subset df
import numpy as np

heatmap_mat = N0_sel[intensity_cols].copy()
heatmap_mat = np.log2(heatmap_mat + 1)

print("[INFO] Heatmap matrix shape:", heatmap_mat.shape)


# %%
#make heatmap 
heatmap_mat_z = heatmap_mat.sub(
    heatmap_mat.mean(axis=1), axis=0
).div(
    heatmap_mat.std(axis=1).replace(0, np.nan), axis=0
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, max(6, 0.25 * heatmap_mat_z.shape[0])))

sns.heatmap(
    heatmap_mat_z,
    cmap="vlag",
    center=0,
    cbar_kws={"label": "Row-wise z-score (log2 mean intensity)"},
    xticklabels=True,
    yticklabels=False   # turn on if you really want OG labels
)

plt.title("Orthogroups containing ENB wheat storage proteins")
plt.xlabel("Species | Tissue")
plt.ylabel("Orthogroup")

plt.tight_layout()
plt.show()


# %%



