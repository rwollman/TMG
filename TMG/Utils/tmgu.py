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


from scipy.stats import entropy

def calculate_mi_from_indices(indices, Type, k=15):
    """
    Calculates the discrete-continuous Mutual Information using a kNN indices matrix.
    
    Parameters:
    -----------
    indices : np.ndarray
        Shape (N, k). The row indices of the k-nearest neighbors for each cell.
        Ensure the cell itself is NOT included in this matrix.
    Type : np.ndarray or pd.Series
        Shape (N,). The discrete cell type labels.
    k : int
        The number of neighbors (should match indices.shape[1]).
    """
    start_time = time.time()
    
    # Force Type to be a raw numpy array to avoid Pandas index alignment errors
    Type = np.asarray(Type)
    N = len(Type)
    
    # 1. Calculate N_{s_i}: The total population count for each cell's type
    unique_types, type_counts = np.unique(Type, return_counts=True)
    type_to_count = dict(zip(unique_types, type_counts))
    Ns_array = np.array([type_to_count[t] for t in Type])
    
    # 2. Calculate m_i: The number of neighbors of the SAME type
    # Reshape Type to (N, 1) to broadcast against the (N, k) neighbor types
    query_types = Type[:, None] 
    neighbor_types = Type[indices]
    
    # Creates a boolean matrix of shape (N, k)
    is_same_type = (query_types == neighbor_types)
    
    # Sum across the rows to get the m count for each cell
    m = is_same_type.sum(axis=1)
    
    # Prevent -inf from log(0) in case a cell is completely isolated from its type
    m = np.maximum(m, 1) 
    
    # 3. Compute the Digamma terms
    term_N = digamma(N)
    term_Ns = np.mean(digamma(Ns_array)) 
    term_m = np.mean(digamma(m))         
    term_k = digamma(k)
    
    # 4. Final Equation (Calculated in nats, converted to bits)
    MI_nats = term_N - term_Ns + term_m - term_k
    MI_bits = MI_nats / np.log(2)
    
    return MI_bits

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
