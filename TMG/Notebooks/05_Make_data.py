"""
05_Make_data.py
===============
Produces the six cache files consumed by 05_Figure5.ipynb:

    Data/mi_c.npy
    Data/mi_r.npy
    Data/alphabet_results.pkl
    Data/region_alphabet_results.pkl
    Data/matrix_gene_cell_mi_based_mapping.csv
    Data/matrix_cell_region_mi_based_mapping.csv

Each file is skipped if it already exists on disk.
Source: extracted from 08b_MI_subsampling.ipynb.
"""

# Must run before any numba/pynndescent import.
# Switches numba from TBB (fork-unsafe) to workqueue (fork-safe),
# allowing multiprocessing fork to work correctly.
import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"

for repo_path in [ROOT / "Repos" / "TMG", ROOT / "Repos" / "max_info_atlas" / "src"]:
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

from TMG.Analysis.TissueGraph import TissueMultiGraph
from TMG.Utils.tmgu import (
    average_type_curves,
    correct_mi_bias,
    get_eligible_types,
    get_gene_names,
    list_entropy,
    run_cell_type_alphabet_discovery,
    run_mi_subsampling,
)

# ---------------------------------------------------------------------------
# Load TMG
# ---------------------------------------------------------------------------
tmg_path = DATA_DIR / "TMG2"
TMG = TissueMultiGraph(basepath=str(tmg_path))
TMG.update_current_type(0, "opt_cell")

# Gene-expression matrix (first 500 genes)
X = TMG.Layers[0].adata.X.toarray()
X = X[:, :500]

# ---------------------------------------------------------------------------
# mi_c.npy — MI between genes and cell types (global tissue curve)
# ---------------------------------------------------------------------------
mi_c_path = DATA_DIR / "mi_c.npy"
Nvec_cell = [5000, 10000, 15000, 20000, 25000]
Cvec_cell = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
             100, 125, 150, 200, 250, 300, 350, 400, 450, 500]
Niter_cell = 100

TMG.update_current_type(0, "opt_cell")
if not mi_c_path.exists():
    print("Computing mi_c ...")
    mi_c = run_mi_subsampling(X, TMG.Layers[0].Type, Nvec_cell, Cvec_cell, Niter_cell)
    np.save(mi_c_path, mi_c)
    print(f"Saved: {mi_c_path}")
else:
    print(f"Loading: {mi_c_path}")
    mi_c = np.load(mi_c_path)

# ---------------------------------------------------------------------------
# mi_r.npy — MI between cell-type neighbourhood features and region types
# ---------------------------------------------------------------------------
mi_r_path = DATA_DIR / "mi_r.npy"
Nvec_region = [5000, 10000, 15000, 20000, 25000]
Cvec_region = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45,
               50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 165]
Niter_region = 100

TMG.update_current_type(0, "opt_cell")
if not mi_r_path.exists():
    print("Computing neighbourhood features R (k=100) ...")
    R = TMG.Layers[0].extract_environments(k=100, cache_results=True)
    TMG.update_current_type(0, "opt_region")
    print("Computing mi_r ...")
    mi_r = run_mi_subsampling(R, TMG.Layers[0].Type, Nvec_region, Cvec_region, Niter_region)
    np.save(mi_r_path, mi_r)
    print(f"Saved: {mi_r_path}")
else:
    print(f"Loading: {mi_r_path}")
    mi_r = np.load(mi_r_path)
    # R is still needed downstream for region_alphabet_results
    TMG.update_current_type(0, "opt_cell")
    print("Computing neighbourhood features R (k=100) for alphabet step ...")
    R = TMG.Layers[0].extract_environments(k=100, cache_results=True)

# ---------------------------------------------------------------------------
# alphabet_results.pkl — greedy gene-alphabet per cell type
# ---------------------------------------------------------------------------
alphabet_results_path = DATA_DIR / "alphabet_results.pkl"

TMG.update_current_type(0, "opt_cell")
Type_cell = np.asarray(TMG.Layers[0].Type)
gene_names = get_gene_names(TMG.Layers[0].adata, n_features=X.shape[1])
eligible_types = get_eligible_types(Type_cell, min_cells=1000)

alphabet_cfg = dict(
    min_cells=1000,
    n_per_class=1000,
    max_genes=30,
    k=25,
    n_repeats=10,
    jitter_scale=1e-6,
    random_state=42,
    n_jobs=64,
    knn_workers=1,
    mp_start_method="fork",
)

if not alphabet_results_path.exists():
    print("Computing alphabet_results ...")
    target_types = eligible_types["target_type"].tolist()
    alphabet_results = run_cell_type_alphabet_discovery(
        X,
        Type_cell,
        gene_names=gene_names,
        target_types=target_types,
        **alphabet_cfg,
    )
    with open(alphabet_results_path, "wb") as f:
        pickle.dump(alphabet_results, f)
    print(f"Saved: {alphabet_results_path}")
else:
    print(f"Loading: {alphabet_results_path}")
    with open(alphabet_results_path, "rb") as f:
        alphabet_results = pickle.load(f)

# ---------------------------------------------------------------------------
# region_alphabet_results.pkl — greedy cell-type-alphabet per region type
# ---------------------------------------------------------------------------
region_alphabet_results_path = DATA_DIR / "region_alphabet_results.pkl"

TMG.update_current_type(0, "opt_region")
Type_region = np.asarray(TMG.Layers[0].Type)
region_feature_names = np.asarray(TMG.get_tax("opt_cell").Type).astype(str)[:165]
eligible_regions = get_eligible_types(Type_region, min_cells=1000)

region_alphabet_cfg = alphabet_cfg.copy()

if not region_alphabet_results_path.exists():
    print("Computing region_alphabet_results ...")
    target_regions = eligible_regions["target_type"].tolist()
    region_alphabet_results = run_cell_type_alphabet_discovery(
        R,
        Type_region,
        gene_names=region_feature_names,
        target_types=target_regions,
        **region_alphabet_cfg,
    )
    with open(region_alphabet_results_path, "wb") as f:
        pickle.dump(region_alphabet_results, f)
    print(f"Saved: {region_alphabet_results_path}")
else:
    print(f"Loading: {region_alphabet_results_path}")
    with open(region_alphabet_results_path, "rb") as f:
        region_alphabet_results = pickle.load(f)

# ---------------------------------------------------------------------------
# Helper: build feature × type matrix from alphabet results
# ---------------------------------------------------------------------------
def build_feature_type_matrix(alphabet_results, threshold_bits=0.8, all_types=None):
    curves = alphabet_results["curves"]
    selected_genes = alphabet_results["selected_genes"]
    screens = alphabet_results["screens"]

    all_gene_names = (
        screens.groupby("gene_name")["gene_index"]
        .min()
        .reset_index()
        .sort_values("gene_index")["gene_name"]
        .tolist()
    )

    if all_types is not None:
        all_type_labels = list(all_types)
    else:
        all_type_labels = alphabet_results["eligible_types"]["target_type"].tolist()

    reaching = (
        curves.loc[curves["mi_bits"] >= threshold_bits]
        .groupby(["target_type", "repeat"])["step"]
        .min()
        .rename("min_step")
        .reset_index()
    )

    if reaching.empty:
        raise ValueError(f"No repeats reach the MI threshold of {threshold_bits} bits.")

    global_min = (
        reaching.groupby("target_type")["min_step"]
        .min()
        .rename("global_min_step")
        .reset_index()
    )

    reaching = reaching.merge(global_min, on="target_type")
    best_repeats = reaching.loc[
        reaching["min_step"] == reaching["global_min_step"],
        ["target_type", "repeat", "global_min_step"],
    ]

    n_best = (
        best_repeats.groupby("target_type")["repeat"]
        .nunique()
        .rename("n_best")
        .reset_index()
    )

    sg = selected_genes.merge(best_repeats, on=["target_type", "repeat"])
    sg = sg.loc[sg["step"] <= sg["global_min_step"]]

    gene_counts = (
        sg.groupby(["target_type", "gene_name"])
        .size()
        .rename("count")
        .reset_index()
    )
    gene_counts = gene_counts.merge(n_best, on="target_type")
    gene_counts["fraction"] = gene_counts["count"] / gene_counts["n_best"]

    matrix = gene_counts.pivot_table(
        index="gene_name",
        columns="target_type",
        values="fraction",
        fill_value=0,
    )
    matrix.columns.name = None
    matrix.index.name = None
    matrix = matrix.reindex(index=all_gene_names, columns=all_type_labels, fill_value=0)
    return matrix

# ---------------------------------------------------------------------------
# matrix_gene_cell_mi_based_mapping.csv
# ---------------------------------------------------------------------------
matrix_gene_cell_path = DATA_DIR / "matrix_gene_cell_mi_based_mapping.csv"
if not matrix_gene_cell_path.exists():
    print("Building matrix_gene_cell ...")
    matrix_gene_cell = build_feature_type_matrix(
        alphabet_results,
        threshold_bits=0.9,
        all_types=np.unique(Type_cell),
    )
    matrix_gene_cell.to_csv(matrix_gene_cell_path)
    print(f"Saved: {matrix_gene_cell_path}")
else:
    print(f"Already exists: {matrix_gene_cell_path}")

# ---------------------------------------------------------------------------
# matrix_cell_region_mi_based_mapping.csv
# ---------------------------------------------------------------------------
matrix_cell_region_path = DATA_DIR / "matrix_cell_region_mi_based_mapping.csv"
if not matrix_cell_region_path.exists():
    print("Building matrix_cell_region ...")
    matrix_cell_region = build_feature_type_matrix(
        region_alphabet_results,
        threshold_bits=0.9,
        all_types=np.unique(Type_region),
    )
    matrix_cell_region.to_csv(matrix_cell_region_path)
    print(f"Saved: {matrix_cell_region_path}")
else:
    print(f"Already exists: {matrix_cell_region_path}")

print("\nDone. All cache files are up to date.")
