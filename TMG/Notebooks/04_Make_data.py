"""
04_Make_data.py
===============
Produces the GraphPercolation cache files consumed by 04_Figure4.ipynb:

    Data/AllenRefSpace/GraphPercolation/{ccf_level}/{section}.npz   (5 CCF levels)
    Data/AllenRefSpace/GraphPercolation/top_down/{section}.npz
    Data/ccf_topdown.npz    (merged top-down type vector cache)
    Data/ccf_buttomup.npz   (merged bottom-up type vector cache)

NOTE: Data/OptResults/Opt_GPs/Regions/{section}.npy.npz is also consumed by
04_Figure4.ipynb but is produced by the region optimisation pipeline (not TMG
directly) and is therefore NOT generated here.

Each output is skipped when it already exists on disk (redo=False propagates
into run_percolation; the caches are checked explicitly below).
Source: extracted from 04_Figure4.ipynb.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"

for repo_path in [ROOT / "Repos" / "TMG", ROOT / "Repos" / "max_info_atlas" / "src"]:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

warnings.filterwarnings("ignore", category=UserWarning)

from max_info_atlases.percolation import GraphPercolation
from TMG.Analysis.TissueGraph import Taxonomy, TissueMultiGraph
from TMG.Utils import tmgu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
tmg_path          = DATA_DIR / "TMG2"
canonical_gp_dir  = DATA_DIR / "AllenRefSpace" / "GraphPercolation"
top_down_gp_dir   = canonical_gp_dir / "top_down"
topdown_cache     = DATA_DIR / "ccf_topdown.npz"
bottomup_cache    = DATA_DIR / "ccf_buttomup.npz"

CCF_LEVELS = [
    "parcellation_organ",
    "parcellation_category",
    "parcellation_division",
    "parcellation_structure",
    "parcellation_substructure",
]

# ---------------------------------------------------------------------------
# Helpers (copied verbatim from 04_Figure4.ipynb)
# ---------------------------------------------------------------------------
def build_ccf_type_matrix(tmg):
    ccf_taxs = {}
    ccf_type_mat = np.zeros((tmg.N[0], len(CCF_LEVELS)), dtype=int)
    tax_basepath = str(DATA_DIR / "TMG2" / "Taxonomies")
    for idx, level in enumerate(CCF_LEVELS):
        values = np.asarray(tmg.Layers[0].adata.obs[level])
        ccf_taxs[level] = Taxonomy(name=level, basepath=tax_basepath, Types=np.unique(values))
        ccf_type_mat[:, idx] = np.asarray(ccf_taxs[level].get_type_ix(values), dtype=int)
    return ccf_taxs, ccf_type_mat


def load_gp_matrix(sections, levels, base_dir):
    gp_matrix = np.empty((len(sections), len(levels)), dtype=object)
    for col, level in enumerate(levels):
        for row, section in enumerate(sections):
            gp = GraphPercolation(np.zeros(1), np.zeros(1))
            gp.load(str(base_dir / level / f"{section}.npz"))
            gp_matrix[row, col] = gp
    return gp_matrix


def load_merge_cache_or_compute(cache_path, gp_matrix, top_down):
    if top_down:
        keys = ("type_ix_td", "best_score_td", "type_vec_td")
    else:
        keys = ("type_ix_bu", "best_score_bu", "type_vec_bu")

    if cache_path.exists():
        print(f"Loading cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return cache[keys[0]], float(cache[keys[1]]), cache[keys[2]]

    print(f"Computing merge ({'top-down' if top_down else 'bottom-up'}) ...")
    type_ix, best_score, type_vec = tmgu.merge_nested_clusters(gp_matrix, top_down=top_down)
    np.savez(cache_path, **{keys[0]: type_ix, keys[1]: best_score, keys[2]: type_vec})
    print(f"Saved: {cache_path}")
    return type_ix, float(best_score), type_vec


# ---------------------------------------------------------------------------
# Load TMG
# ---------------------------------------------------------------------------
print(f"Loading TMG from {tmg_path} ...")
TMG = TissueMultiGraph(basepath=str(tmg_path))
TMG.load_geoms()
sections = list(TMG.unqS)
print(f"  {len(sections)} sections loaded.")

# ---------------------------------------------------------------------------
# Step 1: CCF-level percolations
#   Data/AllenRefSpace/GraphPercolation/{level}/{section}.npz
# ---------------------------------------------------------------------------
print("\n--- Step 1: CCF-level graph percolations ---")
_, ccf_type_mat = build_ccf_type_matrix(TMG)

for idx, level in enumerate(CCF_LEVELS):
    output_dir = canonical_gp_dir / level
    output_dir.mkdir(parents=True, exist_ok=True)
    n_existing = sum(1 for s in sections if (output_dir / f"{s}.npz").exists())
    if n_existing == len(sections):
        print(f"  [{level}] all {len(sections)} files exist, skipping.")
        continue
    print(f"  [{level}] running percolation ({n_existing}/{len(sections)} already done) ...")
    TMG.Layers[0].run_percolation(
        label_vec=ccf_type_mat[:, idx],
        output_pth=str(output_dir),
        redo=False,
        compute_type_entropy=True,
    )
    print(f"  [{level}] done → {output_dir}")

# ---------------------------------------------------------------------------
# Step 2: Merge CCF levels → top-down / bottom-up type vectors
#   Data/ccf_topdown.npz
#   Data/ccf_buttomup.npz
# ---------------------------------------------------------------------------
print("\n--- Step 2: Merge CCF levels (top-down & bottom-up) ---")
GPmat = load_gp_matrix(sections, CCF_LEVELS, canonical_gp_dir)
_, _, type_vec_td = load_merge_cache_or_compute(topdown_cache,  GPmat, top_down=True)
_, _, _           = load_merge_cache_or_compute(bottomup_cache, GPmat, top_down=False)

# ---------------------------------------------------------------------------
# Step 3: Top-down percolation
#   Data/AllenRefSpace/GraphPercolation/top_down/{section}.npz
# ---------------------------------------------------------------------------
print("\n--- Step 3: Top-down graph percolation ---")
top_down_gp_dir.mkdir(parents=True, exist_ok=True)
n_existing = sum(1 for s in sections if (top_down_gp_dir / f"{s}.npz").exists())
if n_existing == len(sections):
    print(f"  All {len(sections)} top_down files exist, skipping.")
else:
    print(f"  Running top-down percolation ({n_existing}/{len(sections)} already done) ...")
    TMG.Layers[0].run_percolation(
        label_vec=type_vec_td,
        output_pth=str(top_down_gp_dir),
        redo=False,
        compute_type_entropy=False,
    )
    print(f"  Done → {top_down_gp_dir}")

print("\nDone. Summary of outputs:")
print(f"  CCF levels : {canonical_gp_dir}/{{level}}/{{section}}.npz")
print(f"  Top-down   : {top_down_gp_dir}/{{section}}.npz")
print(f"  TD cache   : {topdown_cache}")
print(f"  BU cache   : {bottomup_cache}")
print()
print("NOT generated here (requires region optimisation pipeline):")
print(f"  {DATA_DIR}/OptResults/Opt_GPs/Regions/{{section}}.npy.npz")
