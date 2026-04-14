from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import pandas as pd

from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

import seaborn as sns
import os

import igraph
import pynndescent


from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from multiprocessing import Pool, cpu_count


import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 512

from IPython.display import HTML

import sys
import time

from scipy.special import digamma
from scipy.stats import entropy

import multiprocessing as mp
import contextlib, io, os

_shared_X    = None
_shared_Type = None

def _mi_worker(args):
    nrows, ncols, K, metric, seed = args
    rng = np.random.default_rng(seed)
    row_idx = rng.choice(_shared_X.shape[0], size=nrows, replace=False)
    col_idx = rng.choice(_shared_X.shape[1], size=ncols, replace=False)
    X_sub    = _shared_X[np.ix_(row_idx, col_idx)]
    Type_sub = _shared_Type[row_idx]
    with contextlib.redirect_stdout(io.StringIO()):
        return calc_knn_and_mi(X_sub, Type_sub, K=K, metric=metric)


def run_mi_subsampling(X, Type, Nrows, Ncols, n_iter, K=25, metric='correlation', n_jobs=None):
    """
    Run MI estimation over a grid of subsampling conditions, in parallel.

    Runs every combination of Nrows × Ncols (Cartesian product), each repeated
    n_iter times. X is shared with workers via fork copy-on-write — no copying.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    Type : array-like, length N
    Nrows : int or array-like of int
        Row subsample size(s).
    Ncols : int or array-like of int
        Column subsample size(s).
    n_iter : int
        Number of MI estimates per (Nrows, Ncols) condition.
    K : int
        kNN neighbours (default 25).
    metric : str
        pynndescent distance metric (default 'correlation').
    n_jobs : int or None
        Worker processes. None → all available CPUs.

    Returns
    -------
    np.ndarray, shape (n_nrows, n_ncols, n_iter)
    """
    global _shared_X, _shared_Type

    if n_jobs is None:
        n_jobs = os.cpu_count()

    Nrows_arr = np.atleast_1d(np.asarray(Nrows))
    Ncols_arr = np.atleast_1d(np.asarray(Ncols))
    n_r, n_c  = len(Nrows_arr), len(Ncols_arr)

    _shared_X    = X
    _shared_Type = np.asarray(Type)

    rng   = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=n_r * n_c * n_iter)

    tasks = [
        (int(nr), int(nc), K, metric, int(seeds[(i * n_c + j) * n_iter + t]))
        for i, nr in enumerate(Nrows_arr)
        for j, nc in enumerate(Ncols_arr)
        for t in range(n_iter)
    ]

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_jobs) as pool:
        flat = pool.map(_mi_worker, tasks)

    return np.array(flat).reshape(n_r, n_c, n_iter)

def calculate_mi_from_indices(indices: np.ndarray, labels, k: int) -> float:
    """Ross/KSG-style discrete-continuous MI estimate from a kNN index matrix."""
    labels = np.asarray(labels)
    n = len(labels)

    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    n_s = np.array([label_counts[label] for label in labels])

    query_labels = labels[:, None]
    neighbor_labels = labels[indices]
    m = (query_labels == neighbor_labels).sum(axis=1)
    m = np.maximum(m, 1)

    mi_nats = digamma(n) - np.mean(digamma(n_s)) + np.mean(digamma(m)) - digamma(k)
    return float(mi_nats / np.log(2.0))

def calc_knn_and_mi(X, Type, K=25, metric='correlation', n_jobs=1):
    """
    Calculates the k-nearest neighbors (KNN) using pynndescent and computes the MI.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, F). Data matrix where each row is a cell.
    Type : np.ndarray or pd.Series
        The cell type labels (length N).
    K : int or array-like of int
        Number of neighbors. If scalar, returns a single MI value.
        If array, builds the KNN graph once with max(K) neighbors and returns
        an MI value for each k — useful for estimating K-dependent bias.
    metric : str, optional
        Distance metric for pynndescent (default 'correlation').
    n_jobs : int, optional
        Cores for pynndescent (default 1; use -1 for all when calling outside a parallel context).

    Returns
    -------
    mi_bits : float or np.ndarray
        Single MI value if K is scalar; array of shape (len(K),) if K is array-like.
    """
    import pynndescent

    scalar_K = np.isscalar(K)
    K_arr    = np.atleast_1d(np.asarray(K))
    K_max    = int(K_arr.max())

    index = pynndescent.NNDescent(X, n_neighbors=K_max + 1, metric=metric,
                                  random_state=42, n_jobs=n_jobs)
    all_indices, _ = index.neighbor_graph
    neighbor_indices = all_indices[:, 1:]   # strip self; shape (N, K_max)

    mi_values = np.array([
        calculate_mi_from_indices(neighbor_indices[:, :k], Type, k)
        for k in K_arr
    ])

    return float(mi_values[0]) if scalar_K else mi_values

from scipy.optimize import curve_fit

def _bias_model(N, A, B, C):
    return A - B * (N ** -C)

def correct_mi_bias(mi, Nvec, bounds=None):
    """
    Fit MI(N) = A - B*N^(-C) independently for each (Ncols, iter) pair and
    return A, the bias-corrected asymptotic MI estimate.

    Parameters
    ----------
    mi : np.ndarray, shape (n_nrows, n_ncols, n_iter)
        Output of run_mi_subsampling.
    Nvec : array-like, length n_nrows
        The Nrows values used (must match mi's first axis order).
    bounds : tuple or None
        (lower, upper) bounds for (A, B, C).
        Default: ([0, 0, 0], [inf, inf, 2]).

    Returns
    -------
    np.ndarray, shape (n_ncols, n_iter)
        Bias-corrected MI. NaN where the fit did not converge.
    """
    if bounds is None:
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, 2.0])

    N = np.asarray(Nvec, dtype=float)
    n_nrows, n_ncols, n_iter = mi.shape
    result = np.full((n_ncols, n_iter), np.nan)

    for j in range(n_ncols):
        for t in range(n_iter):
            try:
                popt, _ = curve_fit(_bias_model, N, mi[:, j, t],
                                    bounds=bounds, maxfev=10000)
                result[j, t] = popt[0]   # A = asymptotic MI
            except RuntimeError:
                pass   # leave as NaN if fit fails

    return result


def entropy_and_mi(type_mat, label1='0', label2='1'):
    """
    Calculate entropy, conditional entropy, and mutual information between columns 0 and 1 of type_mat.
    Returns a dict:
    {
        f'H_{label1}': ...,
        f'H_{label2}': ...,
        f'H_{label2}_given_{label1}': ...,
        f'H_{label1}_given_{label2}': ...,
        'MI': ...,
    }
    Numbers are rounded to two significant digits.
    """

    # Compute joint table
    counts = pd.crosstab(type_mat[:,0], type_mat[:,1])

    # Normalize to get joint and marginal probabilities
    p_xy = counts / counts.values.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Compute mutual information
    mi = 0.0
    for i in p_x.index:
        for j in p_y.index:
            p_ij = p_xy.loc[i, j]
            if p_ij > 0:
                mi += p_ij * np.log2(p_ij / (p_x[i] * p_y[j]))

    # Entropy of type_mat[:,0]
    values_0, counts_0 = np.unique(type_mat[:,0], return_counts=True)
    p0 = counts_0 / counts_0.sum()
    H_0 = entropy(p0, base=2)

    # Entropy of type_mat[:,1]
    values_1, counts_1 = np.unique(type_mat[:,1], return_counts=True)
    p1 = counts_1 / counts_1.sum()
    H_1 = entropy(p1, base=2)

    # Conditional entropy H(type_mat[:,1] | type_mat[:,0])
    H_1_given_0 = 0.0
    for val0, count0 in zip(values_0, counts_0):
        mask = type_mat[:,0] == val0
        sub_1 = type_mat[mask,1]
        values_1_sub, counts_1_sub = np.unique(sub_1, return_counts=True)
        p1_sub = counts_1_sub / counts_1_sub.sum()
        H_1_given_val0 = entropy(p1_sub, base=2)
        H_1_given_0 += (count0 / len(type_mat)) * H_1_given_val0

    # Conditional entropy H(type_mat[:,0] | type_mat[:,1])
    H_0_given_1 = 0.0
    for val1, count1 in zip(values_1, counts_1):
        mask = type_mat[:,1] == val1
        sub_0 = type_mat[mask,0]
        values_0_sub, counts_0_sub = np.unique(sub_0, return_counts=True)
        p0_sub = counts_0_sub / counts_0_sub.sum()
        H_0_given_val1 = entropy(p0_sub, base=2)
        H_0_given_1 += (count1 / len(type_mat)) * H_0_given_val1

    # Round to two significant digits
    def sig2(x):
        if x == 0:
            return 0.0
        else:
            return float(f"{x:.2f}")

    return {
        f'H_{label1}': sig2(H_0),
        f'H_{label2}': sig2(H_1),
        'MI': sig2(mi),
        f'H_{label2}_given_{label1}': sig2(H_1_given_0),
        f'H_{label1}_given_{label2}': sig2(H_0_given_1)
    }


def list_entropy(X):
        _, cnt = np.unique(X, return_counts=True)
        freq = cnt / len(X)
        return -(np.sum(freq * np.log2(freq)))

def joint_entropy(x, y):
    """
    Calculate the joint entropy of two vectors.
    
    Args:
        x (array-like): First vector
        y (array-like): Second vector
    
    Returns:
        float: Joint entropy H(X,Y)
    """
    # Convert to numpy arrays if needed
    x = np.array(x)
    y = np.array(y)
    
    # Create joint distribution by pairing elements
    joint_values = list(zip(x, y))
    
    # Get unique joint values and their counts
    _, cnt = np.unique(joint_values, axis=0, return_counts=True)
    
    # Calculate frequencies
    freq = cnt / len(joint_values)
    
    # Calculate joint entropy
    return -(np.sum(freq * np.log2(freq)))


def edge_list_from_XY_with_max_dist(XY,max_dist):
    nbrs = NearestNeighbors(radius = max_dist, algorithm = 'ball_tree').fit(XY)
    distances, indices = nbrs.radius_neighbors(XY)
    nn =[len(d) for d in distances]
    ix_rows = np.repeat(np.arange(len(nn)),nn)
    ix_cols = np.hstack(indices)
    ix_dist = np.hstack(distances)
    ELD = np.hstack((ix_rows[:,np.newaxis],ix_cols[:,np.newaxis],ix_dist[:,np.newaxis]))
    ELD = ELD[ELD[:,2]>0,:]
    ELD = ELD[ELD[:,2].argsort(),:]

    return ELD

def edge_list_from_XY_with_k(XY,k,include_dist = False):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(XY)
    distances, indices = nbrs.kneighbors(XY)
    distances = distances[:,1:]
    indices = indices[:,1:]

    ix_ks,ix_rows = np.meshgrid(np.arange(1,k+1),np.arange(XY.shape[0]))
    ix_rows = ix_rows.T.flatten()
    ix_ks = ix_ks.T.flatten()
    ix_cols = indices.T.flatten()
    dists = distances.T.flatten()
    ELK = np.hstack((ix_rows[:,np.newaxis],ix_cols[:,np.newaxis],ix_ks[:,np.newaxis]))
    if include_dist: 
        ELK = np.hstack((ELK,dists[:,np.newaxis]))
    return ELK


def adjacency_to_igraph(adj_mtx, weighted=False, directed=True, simplify=True):
    """
    Converts an adjacency matrix to an igraph object
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        directed (bool): If graph should be directed
    
    Returns:
        G (igraph object): igraph object of adjacency matrix
    
    Uses code from:
        https://github.com/igraph/python-igraph/issues/168
        https://stackoverflow.com/questions/29655111

    Author:
        Wayne Doyle 
        (Fangming Xie modified) 
    """
    nrow, ncol = adj_mtx.shape
    if nrow != ncol:
        raise ValueError('Adjacency matrix should be a square matrix')
    vcount = nrow
    sources, targets = adj_mtx.nonzero()
    edgelist = list(zip(sources, targets))
    G = igraph.Graph(n=vcount, edges=edgelist, directed=directed)
    if weighted:
        G.es['weight'] = adj_mtx.data
    if simplify:
        G.simplify() # simplify inplace; remove duplicated and self connection (Important to avoid double counting from adj_mtx)
    return G

def split_spatial_graph_to_sections(graph_csr,section_ids):
    """
    splits a spatial graph into components based on section information
    """
    
    # if input is iGraph, convert to sparse csr: 
    if isinstance(graph_csr,igraph.Graph):
        graph_csr = graph_csr.get_adjacency_sparse()
    
    unqS,countS = np.unique(section_ids,return_counts = True)
    SG = [None] * len(unqS)
                
    # break this sparse matrix into components, one per section:
    strt=0 
    for i in range(len(unqS)):
        sg_section = graph_csr[strt:strt+countS[i],strt:strt+countS[i]]
        strt=strt+countS[i]
        SG = adjacency_to_igraph(sg_section, directed=False)

    return SG

def get_local_type_abundance(
    types, 
    edgelist=None, 
    SG=None, 
    XY=None, 
    k_spatial=10,
    ):
    """
    types - type labels on the nodes
    
    edgelist - a list of edges (assume duplicated if undirected) 
    SG - spatial neighborhood graph (undirected); Use this to generate edgelist
    XY - spatial coordinates; Use this to first generate kNN graph; then to generate edgelist
    
    return - relative abundace of types for each node
    """
    N = len(types)
    ctg, ctg_idx = np.unique(types, return_inverse=True) 
    if edgelist is not None:
        i, j = edgelist

    elif SG is not None and isinstance(SG, igraph.Graph):
        # assume undirected; edges need to be counted twice
        edges = np.asarray(SG.get_edgelist()) 
        # once
        i = edges[:,0] # cells
        j = ctg_idx[edges[:,1]] # types it connects
        # twice
        i2 = edges[:,1] # cells
        j2 = ctg_idx[edges[:,0]] # types it connects
        # merge
        i = np.hstack([i,i2])
        j = np.hstack([j,j2])

    elif XY is not None and isinstance(XY, np.ndarray):
        NN = NearestNeighbors(n_neighbors=k_spatial)
        NN.fit(XY)
        knn = NN.kneighbors(XY, return_distance=False)

        i = np.repeat(knn[:,0], k_spatial-1) # cells
        j = ctg_idx[knn[:,1:]].reshape(-1,) # types it connects

    dat = np.repeat(1, len(i))

    # count
    env_mat = sparse.coo_matrix((dat, (i,j)), shape=(N, len(ctg))).toarray() # dense
    env_mat = env_mat/env_mat.sum(axis=1).reshape(-1,1)
    env_mat = np.nan_to_num(env_mat, 0)
    
    return env_mat

def build_knn_graph(X, metric, n_neighbors=15, accuracy={'prob':1, 'extras':1.5}, metric_kwds={}, allow_KDtree = True): 
    """
    Buils the knn graph. If X is small uses KDTree, if large pynndescent
    """

    # checks if we have enough rows 
    n_neighbors = min(X.shape[0]-1,n_neighbors)

    if X.shape[0] < 200000 and allow_KDtree:
        knn = cKDTree(X)
        distances, indices = knn.query(X, k=n_neighbors+1)
    else:
        knn = pynndescent.NNDescent(X, n_neighbors=n_neighbors+1,
                                    metric=metric,
                                    diversify_prob=accuracy['prob'],
                                    pruning_degree_multiplier=accuracy['extras'],
                                    metric_kwds=metric_kwds)
        indices, distances = knn.neighbor_graph
    
    indices = indices[:, 1:]  # remove self indices
    distances = distances[:, 1:]  # remove self distances

    id_from = np.tile(np.arange(indices.shape[0]),indices.shape[1])
    id_to = indices.flatten(order='F')

    # build graph
    edgeList = np.vstack((id_from,id_to)).T
    G = igraph.Graph(n=X.shape[0], edges=edgeList, edge_attrs={'weight': distances.flatten(order='F')})
    G.simplify()

    return (G,knn)

def compute_neighborhood(args):
    subgraph, offset, order = args
    
    # Compute neighborhood for each node in the subgraph
    neighborhoods = subgraph.neighborhood(order = order)
    adjusted_neighborhoods = np.full((neighborhoods.shape[0], order), -1, dtype=int)
    for i,nbr in enumerate(neighborhoods): 
        adjusted_neighborhoods[i, :len(nbr)] = nbr
    adjusted_neighborhoods += offset

    return adjusted_neighborhoods

def find_graph_neighborhoold_parallel_by_components(G,order):   
    components = G.clusters()
    subgraphs = [G.subgraph(component) for component in components]
    subgraph_sizes = np.array([subgraph.vcount() for subgraph in subgraphs])

    offsets = np.zeros(len(subgraphs))
    offsets[1:] = np.cumsum(subgraph_sizes[:-1])

    args = [(subgraphs[i], offsets[i], order) for i in range(len(subgraphs))]

    num_cores = cpu_count() //2

    with Pool(processes=num_cores) as pool:
        results = pool.map(compute_neighborhood, args)

    neighborhood_matrix = np.vstack(results)

    return neighborhood_matrix


def merge_nested_clusters(GPmat, top_down = True): 

    Ncells_vec = np.array([GPmat[i,0].N for i in range(GPmat.shape[0])])
    Ncells = Ncells_vec.sum()
    sample_gp = GPmat[0,0]
    score_x = np.asarray(sample_gp.pbond_vec, dtype=float)
    n_score_points = sample_gp.ent_type_real.shape[0]
    score_x = score_x[:n_score_points]

    # create the type matrix from GP objects: 
    type_vecs=list()
    for row in GPmat: 
        type_vecs.append(np.hstack([gp.type_vec[:,np.newaxis] for gp in row]).astype(int))
    type_int_mat = np.vstack(type_vecs)
    Ntypes_per_lvl = (type_int_mat.max(axis=0)+1).astype(int)

    # now extract the type entropy from each GP object to create the overall type x entropy across all possible types (mutliple levels)
    
    # Init all empty matrices inside the list
    delta_ent_per_pbond_type_lvl = [None] * GPmat.shape[1]
    type_freq_lvl = [None] * GPmat.shape[1]

    for lvl in range(len(delta_ent_per_pbond_type_lvl)):
        delta_ent_per_pbond_type_lvl[lvl] = np.zeros((Ntypes_per_lvl[lvl],GPmat.shape[0],n_score_points))
        type_freq_lvl[lvl] = np.zeros((Ntypes_per_lvl[lvl],GPmat.shape[0]))

    # extract the actual values
    for lvl in range(GPmat.shape[1]):
        for sec in range(GPmat.shape[0]): 
            types_in_section,type_freq = np.unique(GPmat[sec,lvl].type_vec,return_counts=True)
            type_freq_lvl[lvl][types_in_section.astype(int),sec] = type_freq/Ncells
            dent = np.abs(GPmat[sec,lvl].ent_type_perm.T-GPmat[sec,lvl].ent_type_real.T)
            # dent /= np.log2(GPmat[sec,lvl].N) 
            delta_ent_per_pbond_type_lvl[lvl][types_in_section.astype(int),sec,:] = dent[:, :n_score_points]


    # update the type numbers (so they are continous and not restaring each lebvel)
    # then concatenate the entropys x pbond and frequencies
    # now each row of delta_ent_per_pbond correspond to a type from the continous numbering

    cmsm_type_mat = type_int_mat.copy()
    cmsm_type_mat[:,1:] = cmsm_type_mat[:,1:]+np.cumsum(Ntypes_per_lvl[np.newaxis,:-1])
    cmsm_type_mat = cmsm_type_mat.astype(int)
    delta_ent_per_pbond = np.concatenate(delta_ent_per_pbond_type_lvl,axis=0)
    cmsm_type_freq = np.concatenate(type_freq_lvl,axis=0)

    # Pre-compute type contributions for optimization
    def precompute_type_contributions(delta_ent_per_pbond, cmsm_type_freq, score_x):
        """Pre-compute each type contribution curve and its integrated score."""
        print("Computing type contributions...")
        n_types = delta_ent_per_pbond.shape[0]
        n_bonds = delta_ent_per_pbond.shape[2]
        contributions = np.zeros((n_types, n_bonds))
        integrated_contributions = np.zeros(n_types)
        
        for i in range(n_types):
            if i % 1000 == 0:
                print(f"  Computing contribution {i}/{n_types}")
            freq = cmsm_type_freq[i,:,np.newaxis]  # shape: (sections, 1)
            dent = delta_ent_per_pbond[i,:,:]      # shape: (sections, bonds)
            # Sum over sections to get contribution per bond for this type
            contributions[i, :] = (dent * freq).sum(axis=0)  # shape: (bonds,)
            integrated_contributions[i] = np.trapz(contributions[i, :], x=score_x)
        
        return contributions, integrated_contributions

    def score(curr_ix):
        """Score with the same integration rule used by GraphPercolation."""
        dent_by_type_by_section_for_curr_types = delta_ent_per_pbond[curr_ix,:,:]
        freq_by_type_by_section_for_curr_types = cmsm_type_freq[curr_ix,:,np.newaxis]
        dent_by_section_summed_over_types = (dent_by_type_by_section_for_curr_types * freq_by_type_by_section_for_curr_types).sum(axis=0)
        dent_by_pbond = dent_by_section_summed_over_types.sum(axis=0)
        scr = np.trapz(dent_by_pbond, x=score_x)
        return scr
    
    def fast_score(curr_ix_list, type_contribution_scores):
        """Fast score calculation using pre-computed integrated contributions."""
        return type_contribution_scores[curr_ix_list].sum()

    # now create the mapping dict so that each type (key) has the list of subtype under it (values)
    # Initialize an empty dictionary
    adj_dict = {}

    # Iterate over the rows of the matrix
    for row in cmsm_type_mat:
        # Iterate over each element in the row
        for i in range(len(row) - 1):
            # If the element is not already a key in the dictionary, add it with an empty list as its value
            if row[i] not in adj_dict:
                adj_dict[row[i]] = []
            # If the next element is not already in the list of adjacent values for the current element, add it
            if row[i+1] not in adj_dict[row[i]]:
                adj_dict[row[i]].append(row[i+1])

    # ok, now we have everything we need, we can loop over all types to see if it's worth expanding any of them: 
    if top_down: 
        curr_ix = list(np.unique(cmsm_type_mat[:,0]))
    else: #i.e. bottom up
        curr_ix = list(np.unique(cmsm_type_mat[:,-1]))
    
    best_score = score(curr_ix)

    improved = True
    if top_down: 
        print("*** OPTIMIZED TOP-DOWN ALGORITHM ***")
        
        # Pre-compute all type contributions once (major optimization!)
        type_contributions_per_bond, type_contribution_scores = precompute_type_contributions(
            delta_ent_per_pbond,
            cmsm_type_freq,
            score_x,
        )
        
        curr_ix_set = set(curr_ix)
        current_score = fast_score(curr_ix, type_contribution_scores)
        iteration_count = 0
        
        print(f"Initial curr_ix size: {len(curr_ix)}, initial score: {current_score:.6f}")
        
        while improved:
            iteration_count += 1
            improved = False
            improvements_this_round = 0
            
            print(f"\n--- Iteration {iteration_count} ---")
            t0 = time.time()
            
            for i in range(len(curr_ix)):
                if curr_ix[i] in adj_dict:
                    parent_type = curr_ix[i]
                    subtypes = adj_dict[parent_type]
                    
                    # Calculate score change incrementally using integrated contributions
                    old_contribution_score = type_contribution_scores[parent_type]
                    new_contribution_score = type_contribution_scores[subtypes].sum()
                    score_change = new_contribution_score - old_contribution_score
                    new_score = current_score + score_change
                    
                    if new_score > current_score:
                        # Update curr_ix and curr_ix_set efficiently
                        curr_ix_set.remove(parent_type)
                        curr_ix = curr_ix[:i] + curr_ix[i+1:] + subtypes
                        curr_ix_set.update(subtypes)
                        current_score = new_score
                        improved = True
                        improvements_this_round += 1
                        
                        # Report improvement
                        print(f"    Improvement #{improvements_this_round}: replaced 1 with {len(subtypes)} types, "
                              f"score: {current_score:.6f}, curr_ix size: {len(curr_ix)}")
                        break  # Start over since curr_ix changed
            
            elapsed = time.time() - t0
            print(f"Iteration {iteration_count} complete:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Improvements: {improvements_this_round}")
            print(f"  Current score: {current_score:.6f}")
            print(f"  Current curr_ix size: {len(curr_ix)}")
            
        best_score = current_score
        print(f"\n*** TOP-DOWN OPTIMIZATION COMPLETE ***")
        print(f"Final score: {best_score:.6f}")
        print(f"Final size: {len(curr_ix)}")
        print(f"Total iterations: {iteration_count}")
        
        type_vec = cmsm_type_mat[np.isin(cmsm_type_mat,curr_ix)]
    else: #i.e. bottom up - OPTIMIZED VERSION
        print("*** OPTIMIZED BOTTOM-UP ALGORITHM ***")
        
        # Pre-compute all type contributions once (major optimization!)
        type_contributions_per_bond, type_contribution_scores = precompute_type_contributions(
            delta_ent_per_pbond,
            cmsm_type_freq,
            score_x,
        )
        
        keys_per_level = [np.unique(cmsm_type_mat[:, lvl]) for lvl in range(4)]
        curr_ix_set = set(curr_ix)
        current_score = fast_score(curr_ix, type_contribution_scores)
        
        print(f"Initial curr_ix size: {len(curr_ix)}, initial score: {current_score:.6f}")
        print(f"Keys per level: {[len(keys) for keys in keys_per_level]}")
        
        for lvl in range(2, -1, -1):
            t0 = time.time()
            print(f"\n--- Starting level: {lvl} ---")
            print(f"Number of keys at level {lvl}: {len(keys_per_level[lvl])}")
            
            improved_count = 0
            keys_with_valid_subtypes = 0
            
            for ii, key in enumerate(keys_per_level[lvl]):
                # Progress update less frequently to reduce overhead
                if ii % 200 == 0:
                    elapsed = time.time() - t0
                    rate = ii / elapsed if elapsed > 0 else 0
                    remaining = (len(keys_per_level[lvl]) - ii) / rate if rate > 0 else float('inf')
                    print(f"  Progress: {ii}/{len(keys_per_level[lvl])} ({100*ii/len(keys_per_level[lvl]):.1f}%) - "
                          f"improvements: {improved_count} - rate: {rate:.1f} keys/s - ETA: {remaining:.1f}s")
                
                if key not in adj_dict:
                    continue
                    
                subtypes = adj_dict[key]
                if all(subtype in curr_ix_set for subtype in subtypes):
                    keys_with_valid_subtypes += 1
                    
                    # Calculate score change incrementally using integrated contributions
                    old_contribution_score = type_contribution_scores[subtypes].sum()
                    new_contribution_score = type_contribution_scores[key]
                    score_change = new_contribution_score - old_contribution_score
                    new_score = current_score + score_change
                    
                    if new_score > current_score:
                        # Update curr_ix and curr_ix_set efficiently
                        for subtype in subtypes:
                            curr_ix.remove(subtype)
                            curr_ix_set.remove(subtype)
                        curr_ix.append(key)
                        curr_ix_set.add(key)
                        current_score = new_score
                        improved_count += 1
                        
                        # Report significant improvements
                        if improved_count <= 50 or improved_count % 50 == 0:
                            print(f"    Improvement #{improved_count}: key={key}, score: {current_score:.6f}, "
                                  f"curr_ix size: {len(curr_ix)} (merged {len(subtypes)} -> 1)")
            
            elapsed = time.time() - t0
            print(f"Level {lvl} complete:")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Improvements: {improved_count}")
            print(f"  Keys with valid subtypes: {keys_with_valid_subtypes}/{len(keys_per_level[lvl])}")
            print(f"  Final score: {current_score:.6f}")
            print(f"  Final curr_ix size: {len(curr_ix)}")
        
        best_score = current_score
        print(f"\n*** OPTIMIZATION COMPLETE ***")
        print(f"Final score: {best_score:.6f}")
        print(f"Final size: {len(curr_ix)}")
        
        mask = np.isin(cmsm_type_mat,curr_ix)
        new_mask = np.full(mask.shape, False)
        last_true_indices = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1) - 1
        new_mask[np.arange(mask.shape[0]), last_true_indices] = True                        
        type_vec = cmsm_type_mat[new_mask]

    return curr_ix,best_score,type_vec

    


class ToyGraph:
    def __init__(self, Nside, pattern = 'random', n_types = None, perm_xy = True, **kwargs,):
        self.N = Nside**2
        self.X, self.Y = np.meshgrid(np.arange(Nside), np.arange(Nside))
        self.XY = np.hstack((self.X.flatten()[:,np.newaxis], self.Y.flatten()[:,np.newaxis])).astype(float)


        if pattern == 'random': 
            self.frac = kwargs.get('frac', 0.5)
            self.type_vec = np.zeros(self.N)
            indices = np.random.choice(np.arange(self.N), int(self.frac * self.N), replace=False)
            self.type_vec[indices] = 1
            self.G = np.zeros((Nside,Nside))
            rows, cols = np.unravel_index(indices, (Nside, Nside))
            self.G[rows,cols] = 1
        elif pattern == 'squares': 
            self.G = np.zeros((Nside,Nside))
            self.border = kwargs.get('border', 0)
            self.square_side = kwargs.get('size', int(Nside/4))
            ix1=np.arange(self.border,self.border + self.square_side)
            ix2=np.arange(Nside - self.border - self.square_side,Nside - self.border)
            self.G[ix1[:, None],ix1] = 1
            self.G[ix1[:, None],ix2] = 1
            self.G[ix2[:, None],ix1] = 1
            self.G[ix2[:, None],ix2] = 1
            self.type_vec = self.G.ravel()
            self.frac = self.G.mean()
        elif pattern == 'grid': 
            self.G = np.zeros((Nside,Nside))
            self.num_squares = kwargs.get('num_squares', 4)
            square_size = Nside // self.num_squares
            cnt=-1
            for i in range(self.num_squares):
                for j in range(self.num_squares):
                    cnt += 1
                    self.G[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = cnt
            self.type_vec = self.G.ravel()
            self.frac = self.G.mean()

        if perm_xy:
            prm = np.random.permutation(np.arange(self.N))
        else: 
            prm = np.arange(self.N)

        self.ordr = np.argsort(prm)
        self.XY=self.XY[prm,:]
        self.type_vec=self.type_vec[prm]

        self.n_types = len(np.unique(self.type_vec))

    def update_type(self,type_vec): 
        self.type_vec = type_vec
        self.G = type_vec.reshape(self.G.shape)
        self.n_types = len(np.unique(self.type_vec))

    def show_graph(self,ax = None):
        ax = sns.heatmap(self.G,cbar=False,ax=ax)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def show_zones(self, ZoneVec, ax = None, clrs = None):
        ZoneVec = ZoneVec[self.ordr]
        zones = np.reshape(ZoneVec,(np.sqrt(self.N).astype(int),np.sqrt(self.N).astype(int)))
        cmap = plt.cm.get_cmap('nipy_spectral', len(np.unique(ZoneVec)))
        if clrs is None: 
            clrs = cmap(np.linspace(0, 1, len(np.unique(ZoneVec))))
            np.random.shuffle(clrs)
        cmap = mcolors.ListedColormap(clrs)
        ax = sns.heatmap(zones, cmap=cmap,cbar=False,ax = ax,vmin=0,vmax=len(clrs))
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def show_all(self, ZoneVec):
        fig,axs = plt.subplots(1,2,figsize=(10,5))
        axs = axs.T
        self.show_graph(ax=axs.flatten()[0])
        self.show_zones(ZoneVec,ax=axs.flatten()[1])
        plt.show()

    def create_animation(self, Zones, filename = None):
        fig, ax = plt.subplots()

        all_rgbs = list()
        cmap = plt.cm.get_cmap('nipy_spectral', self.N)

        rgb_real = cmap(np.linspace(0, 1, self.N))
        np.random.shuffle(rgb_real)

        def animate(i):
            ax.clear()    
            self.show_zones(Zones[:,i], ax=ax, clrs = rgb_real)

        ani = animation.FuncAnimation(fig, animate, frames=Zones.shape[1], interval=200)
        if filename is not None:
            ani.save(filename + '.gif', writer='imagemagick')

        html = HTML(ani.to_jshtml())
        if filename is not None: 
            with open(filename + '.html', 'w') as f:
                f.write("<html>\n")
                f.write("<head>\n")
                f.write("<title>Animation</title>\n")
                f.write("</head>\n")
                f.write("<body>\n")
                f.write(html.data)
                f.write("</body>\n")
                f.write("</html>")
                
        return html


# ---------------------------------------------------------------------------
# Cell-type alphabet discovery (MI-based gene selection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BalancedBinarySample:
    target_type: object
    row_indices: np.ndarray
    labels: np.ndarray
    X: np.ndarray


_shared_discovery_X = None
_shared_discovery_types = None
_shared_discovery_gene_names = None


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def get_gene_names(adata, n_features: int | None = None) -> np.ndarray:
    """Best-effort gene-name extraction for the first n_features columns."""
    candidates = ("gene_symbol", "gene", "symbol", "feature_name")
    names = None

    if hasattr(adata, "var"):
        for column in candidates:
            if column in adata.var.columns:
                names = adata.var[column].astype(str).to_numpy()
                break
        if names is None:
            names = adata.var.index.astype(str).to_numpy()
    else:
        raise ValueError("adata must expose a .var table or index")

    if n_features is None:
        return names
    return names[:n_features]


def get_eligible_types(types, min_cells: int = 1000) -> pd.DataFrame:
    values = _as_numpy(types)
    unique_types, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    table = pd.DataFrame(
        {
            "target_type": unique_types[order],
            "n_cells": counts[order],
        }
    )
    return table.loc[table["n_cells"] >= min_cells].reset_index(drop=True)


def draw_balanced_binary_sample(
    X: np.ndarray,
    types,
    target_type,
    n_per_class: int = 1000,
    rng=None,
) -> BalancedBinarySample:
    """Create a balanced target-vs-background problem for one cell type."""
    rng = np.random.default_rng(rng)
    types = _as_numpy(types)
    target_mask = types == target_type
    target_idx = np.flatnonzero(target_mask)
    background_idx = np.flatnonzero(~target_mask)

    if len(target_idx) < n_per_class:
        raise ValueError(
            f"Target '{target_type}' has only {len(target_idx)} cells, "
            f"but n_per_class={n_per_class}."
        )
    if len(background_idx) < n_per_class:
        raise ValueError(
            f"Background for '{target_type}' has only {len(background_idx)} cells, "
            f"but n_per_class={n_per_class}."
        )

    pos_idx = rng.choice(target_idx, size=n_per_class, replace=False)
    neg_idx = rng.choice(background_idx, size=n_per_class, replace=False)
    row_indices = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate(
        [
            np.ones(n_per_class, dtype=np.int8),
            np.zeros(n_per_class, dtype=np.int8),
        ]
    )

    shuffle = rng.permutation(len(row_indices))
    row_indices = row_indices[shuffle]
    labels = labels[shuffle]

    return BalancedBinarySample(
        target_type=target_type,
        row_indices=row_indices,
        labels=labels,
        X=np.asarray(X)[row_indices],
    )


def add_micro_jitter(X: np.ndarray, jitter_scale: float = 1e-6, rng=None) -> np.ndarray:
    """Break ties without changing the large-scale geometry of expression space."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=np.float64)
    scale = np.nanstd(X, axis=0, ddof=0)
    if X.ndim == 1:
        scale = np.asarray([float(scale)])
    scale = np.where(scale > 0, scale, 1.0)
    noise = rng.normal(scale=jitter_scale * scale, size=X.shape)
    return X + noise


def zscore_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std


def sorted_1d_knn_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Exact 1D kNN via sorting rather than repeated tree construction."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = len(x)
    if not 0 < k < n:
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")

    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    result = np.empty((n, k), dtype=np.int32)

    for rank, row_idx in enumerate(order):
        left = rank - 1
        right = rank + 1
        chosen = []
        while len(chosen) < k:
            if left < 0:
                chosen.append(order[right])
                right += 1
            elif right >= n:
                chosen.append(order[left])
                left -= 1
            else:
                left_dist = abs(sorted_x[rank] - sorted_x[left])
                right_dist = abs(sorted_x[right] - sorted_x[rank])
                if left_dist <= right_dist:
                    chosen.append(order[left])
                    left -= 1
                else:
                    chosen.append(order[right])
                    right += 1
        result[row_idx] = chosen

    return result


def exact_knn_indices(X: np.ndarray, k: int, workers: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return sorted_1d_knn_indices(X, k=k)

    n = X.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")

    tree = cKDTree(X)
    _, indices = tree.query(X, k=k + 1, workers=workers)
    return np.asarray(indices[:, 1:], dtype=np.int32)


def mi_from_expression(
    X: np.ndarray,
    labels,
    k: int = 25,
    standardize: bool = True,
    knn_workers: int = 1,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        indices = sorted_1d_knn_indices(X, k=k)
    else:
        X_eval = zscore_columns(X) if standardize else X
        indices = exact_knn_indices(X_eval, k=k, workers=knn_workers)
    return calculate_mi_from_indices(indices, labels, k=k)


def screen_primary_markers(
    X: np.ndarray,
    labels,
    gene_names: Sequence[str],
    k: int = 25,
    jitter_scale: float = 1e-6,
    rng=None,
    knn_workers: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng)
    X_jittered = add_micro_jitter(X, jitter_scale=jitter_scale, rng=rng)
    rows = []
    for gene_idx in range(X_jittered.shape[1]):
        mi_bits = mi_from_expression(
            X_jittered[:, gene_idx],
            labels,
            k=k,
            standardize=False,
            knn_workers=knn_workers,
        )
        rows.append(
            {
                "gene_index": gene_idx,
                "gene_name": str(gene_names[gene_idx]),
                "mi_bits": mi_bits,
            }
        )

    screen = pd.DataFrame(rows).sort_values("mi_bits", ascending=False).reset_index(drop=True)
    screen["rank"] = np.arange(1, len(screen) + 1)
    return screen


def greedy_forward_selection(
    X: np.ndarray,
    labels,
    gene_names: Sequence[str],
    max_genes: int = 10,
    k: int = 25,
    jitter_scale: float = 1e-6,
    rng=None,
    standardize: bool = True,
    knn_workers: int = 1,
) -> dict[str, object]:
    """Greedily build an information-maximizing gene alphabet for one binary task."""
    if max_genes < 1:
        raise ValueError("max_genes must be at least 1")

    rng = np.random.default_rng(rng)
    X_jittered = add_micro_jitter(X, jitter_scale=jitter_scale, rng=rng)
    screen = screen_primary_markers(
        X_jittered,
        labels,
        gene_names=gene_names,
        k=k,
        jitter_scale=0.0,
        knn_workers=knn_workers,
    )

    selected = [int(screen.iloc[0]["gene_index"])]
    curve_rows = [
        {
            "step": 1,
            "selected_gene_index": selected[0],
            "selected_gene": str(gene_names[selected[0]]),
            "mi_bits": float(screen.iloc[0]["mi_bits"]),
            "delta_bits": float(screen.iloc[0]["mi_bits"]),
        }
    ]

    remaining = set(range(X_jittered.shape[1])) - set(selected)

    while len(selected) < min(max_genes, X_jittered.shape[1]):
        best_gene = None
        best_mi = -np.inf

        for gene_idx in remaining:
            cols = selected + [gene_idx]
            candidate_mi = mi_from_expression(
                X_jittered[:, cols],
                labels,
                k=k,
                standardize=standardize,
                knn_workers=knn_workers,
            )
            if candidate_mi > best_mi:
                best_mi = candidate_mi
                best_gene = gene_idx

        previous_mi = float(curve_rows[-1]["mi_bits"])
        recorded_mi = max(float(best_mi), previous_mi)
        selected.append(int(best_gene))
        remaining.remove(best_gene)
        curve_rows.append(
            {
                "step": len(selected),
                "selected_gene_index": int(best_gene),
                "selected_gene": str(gene_names[best_gene]),
                "mi_bits": recorded_mi,
                "delta_bits": float(recorded_mi - previous_mi),
            }
        )

    return {
        "curve": pd.DataFrame(curve_rows),
        "screen": screen,
        "selected_gene_indices": selected,
        "selected_gene_names": [str(gene_names[idx]) for idx in selected],
    }


def _set_discovery_globals(X, types, gene_names) -> None:
    global _shared_discovery_X, _shared_discovery_types, _shared_discovery_gene_names
    _shared_discovery_X = np.asarray(X)
    _shared_discovery_types = _as_numpy(types)
    _shared_discovery_gene_names = np.asarray(gene_names, dtype=object)


def _run_single_discovery_task(
    X: np.ndarray,
    types: np.ndarray,
    gene_names: Sequence[str],
    task,
) -> dict[str, object]:
    (
        target_type,
        repeat,
        n_per_class,
        max_genes,
        k,
        jitter_scale,
        seed,
        standardize,
        knn_workers,
    ) = task

    sample = draw_balanced_binary_sample(
        X,
        types,
        target_type=target_type,
        n_per_class=n_per_class,
        rng=seed,
    )
    run = greedy_forward_selection(
        sample.X,
        sample.labels,
        gene_names=gene_names,
        max_genes=max_genes,
        k=k,
        jitter_scale=jitter_scale,
        rng=seed,
        standardize=standardize,
        knn_workers=knn_workers,
    )

    curve = run["curve"].copy()
    curve["target_type"] = target_type
    curve["repeat"] = repeat

    screen = run["screen"].copy()
    screen["target_type"] = target_type
    screen["repeat"] = repeat

    selected_rows = [
        {
            "target_type": target_type,
            "repeat": repeat,
            "step": step,
            "gene_name": gene_name,
            "gene_index": int(run["selected_gene_indices"][step - 1]),
        }
        for step, gene_name in enumerate(run["selected_gene_names"], start=1)
    ]

    return {
        "curve": curve,
        "screen": screen,
        "selected_genes": pd.DataFrame(selected_rows),
    }


def _discovery_worker(task) -> dict[str, object]:
    if _shared_discovery_X is None or _shared_discovery_types is None or _shared_discovery_gene_names is None:
        raise RuntimeError("Shared discovery state is not initialized")
    return _run_single_discovery_task(
        _shared_discovery_X,
        _shared_discovery_types,
        _shared_discovery_gene_names,
        task,
    )


def _resolve_mp_context(start_method: str | None):
    if start_method is None:
        return mp.get_context()
    try:
        return mp.get_context(start_method)
    except ValueError:
        return mp.get_context()


def run_cell_type_alphabet_discovery(
    X: np.ndarray,
    types,
    gene_names: Sequence[str],
    target_types: Iterable | None = None,
    min_cells: int = 1000,
    n_per_class: int = 1000,
    max_genes: int = 10,
    k: int = 25,
    n_repeats: int = 1,
    jitter_scale: float = 1e-6,
    random_state: int = 0,
    standardize: bool = True,
    n_jobs: int | None = None,
    knn_workers: int = 1,
    mp_start_method: str | None = "fork",
    chunksize: int = 1,
) -> dict[str, object]:
    types = _as_numpy(types)
    gene_names = np.asarray(gene_names, dtype=object)
    eligible = get_eligible_types(types, min_cells=min_cells)

    if target_types is None:
        target_types = eligible["target_type"].tolist()
    else:
        target_types = list(target_types)

    rng = np.random.default_rng(random_state)
    tasks = [
        (
            target_type,
            repeat,
            n_per_class,
            max_genes,
            k,
            jitter_scale,
            int(rng.integers(0, 2**31 - 1)),
            standardize,
            knn_workers,
        )
        for target_type in target_types
        for repeat in range(n_repeats)
    ]

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(int(n_jobs), max(len(tasks), 1)))

    if not tasks:
        empty_curve = pd.DataFrame(
            columns=["step", "selected_gene_index", "selected_gene", "mi_bits", "delta_bits", "target_type", "repeat"]
        )
        empty_screen = pd.DataFrame(
            columns=["gene_index", "gene_name", "mi_bits", "rank", "target_type", "repeat"]
        )
        empty_selected = pd.DataFrame(columns=["target_type", "repeat", "step", "gene_name", "gene_index"])
        return {
            "curves": empty_curve,
            "screens": empty_screen,
            "selected_genes": empty_selected,
            "eligible_types": eligible,
            "config": {
                "min_cells": min_cells,
                "n_per_class": n_per_class,
                "max_genes": max_genes,
                "k": k,
                "n_repeats": n_repeats,
                "jitter_scale": jitter_scale,
                "random_state": random_state,
                "standardize": standardize,
                "n_jobs": n_jobs,
                "knn_workers": knn_workers,
                "mp_start_method": mp_start_method,
                "chunksize": chunksize,
            },
        }

    if n_jobs == 1:
        task_results = [
            _run_single_discovery_task(X, types, gene_names, task)
            for task in tasks
        ]
    else:
        ctx = _resolve_mp_context(mp_start_method)
        _set_discovery_globals(X, types, gene_names)
        pool_kwargs = {"processes": n_jobs}
        if ctx.get_start_method() != "fork":
            pool_kwargs["initializer"] = _set_discovery_globals
            pool_kwargs["initargs"] = (X, types, gene_names)
        with ctx.Pool(**pool_kwargs) as pool:
            task_results = pool.map(_discovery_worker, tasks, chunksize=chunksize)

    return {
        "curves": pd.concat([result["curve"] for result in task_results], ignore_index=True),
        "screens": pd.concat([result["screen"] for result in task_results], ignore_index=True),
        "selected_genes": pd.concat([result["selected_genes"] for result in task_results], ignore_index=True),
        "eligible_types": eligible,
        "config": {
            "min_cells": min_cells,
            "n_per_class": n_per_class,
            "max_genes": max_genes,
            "k": k,
            "n_repeats": n_repeats,
            "jitter_scale": jitter_scale,
            "random_state": random_state,
            "standardize": standardize,
            "n_jobs": n_jobs,
            "knn_workers": knn_workers,
            "mp_start_method": mp_start_method,
            "chunksize": chunksize,
        },
    }


def summarize_saturation_threshold(results, threshold_bits: float = 0.90) -> pd.DataFrame:
    curves = results["curves"].copy()
    final_step = curves.groupby(["target_type", "repeat"])["step"].max().rename("max_step")
    first_crossing = (
        curves.loc[curves["mi_bits"] >= threshold_bits]
        .groupby(["target_type", "repeat"])["step"]
        .min()
        .rename("genes_to_threshold")
    )

    summary = (
        curves.groupby(["target_type", "repeat"])["mi_bits"]
        .max()
        .rename("best_mi_bits")
        .to_frame()
        .join(final_step)
        .join(first_crossing)
        .reset_index()
    )
    summary["genes_to_threshold"] = summary["genes_to_threshold"].fillna(summary["max_step"] + 1)

    grouped = (
        summary.groupby("target_type")
        .agg(
            n_repeats=("repeat", "nunique"),
            mean_best_mi_bits=("best_mi_bits", "mean"),
            std_best_mi_bits=("best_mi_bits", "std"),
            mean_genes_to_threshold=("genes_to_threshold", "mean"),
            median_genes_to_threshold=("genes_to_threshold", "median"),
            min_genes_to_threshold=("genes_to_threshold", "min"),
            max_genes_to_threshold=("genes_to_threshold", "max"),
        )
        .reset_index()
        .sort_values(
            ["mean_genes_to_threshold", "mean_best_mi_bits", "target_type"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )
    return grouped


def average_type_curves(results) -> pd.DataFrame:
    curves = results["curves"].copy()
    return (
        curves.groupby(["target_type", "step"])["mi_bits"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mi_bits_mean", "std": "mi_bits_std"})
    )
