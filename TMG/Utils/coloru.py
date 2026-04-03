# Removed colormath dependency due to numpy.asscalar incompatibility
# from colormath.color_objects import sRGBColor, LabColor
# from colormath.color_conversions import convert_color
# from colormath.color_diff import delta_e_cie2000, delta_e_cie1976
import numpy as np
import random
import umap
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
import itertools
import time
from typing import Tuple, Dict, List, Optional, Union
import warnings


def _rgb_to_xyz(rgb):
    """Convert RGB to XYZ color space"""
    # Normalize RGB values to 0-1 if they're in 0-255 range
    if np.any(rgb > 1.0):
        rgb = rgb / 255.0
    
    # Apply gamma correction (sRGB to linear RGB)
    rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))
    
    # Convert to XYZ using sRGB matrix
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = rgb_linear @ matrix.T
    return xyz

def _xyz_to_lab(xyz):
    """Convert XYZ to LAB color space"""
    # D65 illuminant reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    # Normalize by reference white
    xyz_norm = xyz / np.array([xn, yn, zn])
    
    # Apply LAB transformation
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3, np.power(t, 1.0/3.0), t / (3 * delta**2) + 4.0/29.0)
    
    fx, fy, fz = f(xyz_norm[0]), f(xyz_norm[1]), f(xyz_norm[2])
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.array([L, a, b])

def _lab_to_xyz(lab):
    """Convert LAB to XYZ color space"""
    L, a, b = lab[0], lab[1], lab[2]
    
    # D65 illuminant reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    # Calculate intermediate values
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Apply inverse transformation
    def f_inv(t):
        delta = 6.0 / 29.0
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4.0/29.0))
    
    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)
    
    return np.array([x, y, z])

def _xyz_to_rgb(xyz):
    """Convert XYZ to RGB color space"""
    # Convert from XYZ to linear RGB using inverse sRGB matrix
    matrix_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb_linear = xyz @ matrix_inv.T
    
    # Apply inverse gamma correction (linear RGB to sRGB)
    rgb = np.where(rgb_linear <= 0.0031308, 
                   12.92 * rgb_linear, 
                   1.055 * np.power(rgb_linear, 1.0/2.4) - 0.055)
    
    return np.clip(rgb, 0, 1)

def rgb_array_to_hex(rgb_list):
    # Ensure input is a list of lists or numpy array
    hex_list = []
    for rgb in rgb_list:
        # Handle np arrays or tuples, scale if 0-1, and clamp
        vals = np.array(rgb)
        if vals.max() <= 1.01:  # Assume 0-1 range
            vals = np.clip(vals, 0, 1)
            vals = (vals * 255).round().astype(int)
        else:  # Already 0-255
            vals = np.clip(vals, 0, 255).astype(int)
        hex_str = '#{:02X}{:02X}{:02X}'.format(vals[0], vals[1], vals[2])
        hex_list.append(hex_str)
    return hex_list

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space"""
    xyz = _rgb_to_xyz(rgb)
    lab = _xyz_to_lab(xyz)
    return lab

def lab_to_rgb(lab):
    """Convert LAB to RGB color space"""
    xyz = _lab_to_xyz(lab)
    rgb = _xyz_to_rgb(xyz)
    return rgb

def delta_e_cie1976(lab1, lab2):
    """Calculate CIE 1976 Delta E color difference"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

def delta_e_cie2000(lab1, lab2):
    """Calculate CIE 2000 Delta E color difference (simplified version)"""
    # This is a simplified implementation of CIE 2000
    # For a full implementation, the formula is quite complex
    # This provides a reasonable approximation
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]
    
    # Calculate differences
    dL = L1 - L2
    da = a1 - a2
    db = b1 - b2
    
    # Calculate C values
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    
    # Calculate dH
    dH_squared = da**2 + db**2 - dC**2
    dH = np.sqrt(max(0, dH_squared))
    
    # Weighting functions (simplified) - use average chroma for symmetry
    C_avg = (C1 + C2) / 2.0
    SL = 1.0
    SC = 1.0 + 0.045 * C_avg
    SH = 1.0 + 0.015 * C_avg
    
    # Calculate Delta E 2000
    delta_e = np.sqrt((dL/SL)**2 + (dC/SC)**2 + (dH/SH)**2)
    return delta_e

def rgb_to_xyz(rgb):
    """
    Convert RGB to XYZ color space
    
    Parameters
    ----------
    rgb : numpy.ndarray
        RGB values in range [0, 1], shape (n, 3) or (3,)
    
    Returns
    -------
    numpy.ndarray
        XYZ values
    """
    # Ensure input is 2D
    rgb = np.atleast_2d(rgb)
    
    # Convert to linear RGB
    rgb_linear = np.where(rgb > 0.04045, 
                         np.power((rgb + 0.055) / 1.055, 2.4),
                         rgb / 12.92)
    
    # sRGB to XYZ transformation matrix (D65 illuminant)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    
    # Apply transformation
    xyz = np.dot(rgb_linear, M.T)
    
    return xyz

def xyz_to_lab(xyz):
    """
    Convert XYZ to CIELAB color space
    
    Parameters
    ----------
    xyz : numpy.ndarray
        XYZ values, shape (n, 3) or (3,)
    
    Returns
    -------
    numpy.ndarray
        LAB values
    """
    # Ensure input is 2D
    xyz = np.atleast_2d(xyz)
    
    # D65 illuminant reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    # Normalize by reference white
    x = xyz[:, 0] / Xn
    y = xyz[:, 1] / Yn
    z = xyz[:, 2] / Zn
    
    # Apply nonlinear transformation
    fx = np.where(x > 0.008856, np.power(x, 1/3), (7.787 * x + 16/116))
    fy = np.where(y > 0.008856, np.power(y, 1/3), (7.787 * y + 16/116))
    fz = np.where(z > 0.008856, np.power(z, 1/3), (7.787 * z + 16/116))
    
    # Calculate LAB values
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.column_stack([L, a, b])

def rgb_to_lab(rgb):
    """
    Convert RGB to CIELAB color space
    
    Parameters
    ----------
    rgb : numpy.ndarray
        RGB values in range [0, 1], shape (n, 3) or (3,)
    
    Returns
    -------
    numpy.ndarray
        LAB values
    """
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab

def delta_e_cie1976_vec(lab1, lab2):
    """
    Calculate CIE 1976 Delta E between LAB colors
    
    Parameters
    ----------
    lab1, lab2 : numpy.ndarray
        LAB color values, shape (n, 3) or (3,)
    
    Returns
    -------
    numpy.ndarray
        Delta E values
    """
    lab1 = np.atleast_2d(lab1)
    lab2 = np.atleast_2d(lab2)
    
    # Euclidean distance in LAB space
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=1))
    
    return delta_e

def scale_vec(x, low, high):
    """
    Scale vector between bounds

    Input
    -----
    x : 1D vector
    low : minimum value after rescaling
    high : maximum value after rescaling

    Output
    ------
    NumPy array : rescaled values 
    """
    return ((high - low) * (x - x.min()) / (x.max() - x.min())) + low

def color_diff(clr1, clr2, mode="RGB", de="1976"):
    """
    Calculate color difference between two colors
    
    Parameters
    ----------
    clr1, clr2 : array-like
        RGB color values in range [0, 1]
    mode : str
        Color space mode (currently only "RGB" supported)
    de : str
        Delta E method ("1976" or "2000", currently only "1976" supported)
    
    Returns
    -------
    float
        Delta E color difference
    """
    if mode == "RGB":
        # Convert RGB to LAB
        clr1_lab = rgb_to_lab(np.array(clr1).reshape(1, -1))
        clr2_lab = rgb_to_lab(np.array(clr2).reshape(1, -1))
    else:
        raise ValueError("Only RGB mode is currently supported")
    
    # Calculate delta E
    if de == "1976":
        delta_e = delta_e_cie1976_vec(clr1_lab, clr2_lab)[0]
    elif de == "2000":
        raise NotImplementedError("Delta E 2000 not yet implemented")
    else:
        raise ValueError("de must be '1976' or '2000' (strings)")
    
    return delta_e

def color_diff_vec(clr1, clr2, mode="RGB", de="1976"):
    """
    Calculate color differences between arrays of colors
    
    Parameters
    ----------
    clr1, clr2 : numpy.ndarray
        RGB color arrays, shape (n, 3), values in range [0, 1]
    mode : str
        Color space mode ("RGB" or "Lab")
    de : str
        Delta E method ("1976" supported)
    
    Returns
    -------
    numpy.ndarray
        Array of delta E values, shape (n,)
    """
    if mode == "Lab" and de == "1976":
        # Direct calculation in LAB space
        delta_e = np.sqrt(np.sum(np.power(clr1 - clr2, 2), axis=1))
    elif mode == "RGB":
        # Convert RGB to LAB first
        clr1_lab = rgb_to_lab(clr1)
        clr2_lab = rgb_to_lab(clr2)
        
        if de == "1976":
            delta_e = delta_e_cie1976_vec(clr1_lab, clr2_lab)
        elif de == "2000":
            raise NotImplementedError("Delta E 2000 not yet implemented")
        else:
            raise ValueError("de must be '1976' or '2000' (strings)")
    else:
        raise ValueError("mode must be 'RGB' or 'Lab'")
    
    return delta_e

def convert_lab01_2rgb(clr_best):
    if clr_best.shape[1] != 3:
        sz = int(len(clr_best)/3)
        clr_best = np.reshape(clr_best, (sz, 3))
    clr_rgb = np.zeros(clr_best.shape)
    for i in range(clr_rgb.shape[0]):
        # Convert normalized Lab (0-1) to actual Lab values
        L = clr_best[i, 0] * 100
        a = (clr_best[i, 1] - 0.5) * 255
        b = (clr_best[i, 2] - 0.5) * 255
        lab = np.array([L, a, b])
        clr_rgb[i, :] = lab_to_rgb(lab)

    clr_rgb = np.clip(clr_rgb, 0, 1)
    return clr_rgb

def rand_hex_codes(n):
    color_list = []

    for _ in range(n):
        # Generate random RGB values between 0 and 255
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        
        # Convert RGB values to hex format and add to the list
        hex_color = "#{:02X}{:02X}{:02X}".format(red, green, blue)
        color_list.append(hex_color)

    return color_list

def hex_to_rgb(hex_codes):
    def single_hex_to_rgb(hex_code):
        rgb_255 = tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5))
        rgb = (float(rgb_255[0])/255,float(rgb_255[1])/255,float(rgb_255[2])/255)
        return rgb
    
    return [single_hex_to_rgb(hex_code) for hex_code in hex_codes]


def type_color_using_supervized_umap(data,target):
    reducer = umap.UMAP(n_components = 3, metric = "cosine")
    embedding = reducer.fit_transform(data, y = target)
    L = embedding[:,0]
    L = (L-L.min())/(L.max()-L.min())
    a = embedding[:,1]
    a = ((a-a.min())/(a.max()-a.min()))
    b = embedding[:,2]
    b = ((b-b.min())/(b.max()-b.min()))
    Lab = np.hstack((L[:,np.newaxis],a[:,np.newaxis],b[:,np.newaxis]))
    rgb_by_type = convert_lab01_2rgb(Lab)
    return rgb_by_type

def type_color_using_linkage(data,cmap,metric = "cosine"):
    dvec = pdist(data,metric)
    z = hierarchy.linkage(dvec,method='average')
    ordr = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(z,dvec))
    rgb_by_type = cmap(np.linspace(0,1,data.shape[0]+1))
    rgb_by_type = rgb_by_type[1:,:]
    rgb_by_type = rgb_by_type[ordr,:]
    return rgb_by_type

def merge_colormaps(colormap_names,clr_range = (0,1),res = 128):
    """
    Merge multiple matplotlib colormaps into one. 
    clr_range: either a tuple (same for all colormaps, or an Nx2 array with ranges to use. 
           full range is 0-1, so to clip any side just use partial range (0.1,1)
    """

    # make colormap_names into a list (if it's just a string name)
    if not isinstance(colormap_names,list):
        colormap_names=[colormap_names]

    # process / verify the range input
    if not isinstance(clr_range, np.ndarray): 
        clr_range = np.tile(clr_range,(len(colormap_names),1))
    assert clr_range.shape[0]==len(colormap_names), "ranges dimension doesn't match colormap names"

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors = []
    for i,cmap_name in enumerate(colormap_names): 
        cmap = cm.get_cmap(cmap_name, res)
        colors.append(cmap(np.linspace(clr_range[i,0],clr_range[i,1],res)))
    colors = np.vstack(colors)
    mymap = LinearSegmentedColormap.from_list('my_colormap', colors)
    return mymap

def hex_color_distance_matrix(hex_codes, de="1976"):
    """
    Compute deltaE distance matrix between hex color codes in CIELAB space.
    
    Parameters
    ----------
    hex_codes : list
        List of hex color codes (e.g., ['#d60000', '#8c3bff', '#018700'])
    de : str, optional
        Delta E method to use: "1976" or "2000" (default: "1976")
    
    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix where element [i,j] is the deltaE 
        distance between hex_codes[i] and hex_codes[j]
    """
    # Convert hex codes to RGB tuples
    rgb_colors = hex_to_rgb(hex_codes)
    rgb_array = np.array(rgb_colors)
    
    n_colors = len(hex_codes)
    distance_matrix = np.zeros((n_colors, n_colors))
    
    # Get only unique pairs (excluding diagonal) for upper triangular matrix
    pairs = list(itertools.combinations(range(n_colors), 2))
    
    # Create arrays for only the unique pairwise comparisons
    indices_i = [pair[0] for pair in pairs]
    indices_j = [pair[1] for pair in pairs]
    
    clr1 = rgb_array[indices_i]
    clr2 = rgb_array[indices_j]
    
    # Compute distances for only the unique pairs (diagonal is already 0)
    distances = color_diff_vec(clr1, clr2, mode="RGB", de=de)
    
    # Fill the distance matrix using symmetry
    for k, (i, j) in enumerate(pairs):
        distance_matrix[i, j] = distances[k]
        distance_matrix[j, i] = distances[k]  # Fill symmetric position
    
    return distance_matrix

def greedy_color_assignment(
    d_features: np.ndarray,
    d_colors: np.ndarray,
    glasbey_colors_rgb: list,
    nan_color: str = "#808080"  # Default gray color
) -> list:
    """
    Assigns colors to cell types using a greedy anchor-and-accrete heuristic.
    
    This function implements Phase 1 of the ColorAssigner module, which creates
    a high-quality initial mapping of cell types to colors using a greedy
    approach that anchors the most dissimilar entities and iteratively adds
    the next most distinct items.

    Args:
        d_features: An (N, N) numpy array of pairwise feature distances.
                    Rows/columns corresponding to unseen types should be all NaN.
        d_colors: A pre-computed (256, 256) numpy array of pairwise Delta E
                  distances between the Glasbey colors.
        glasbey_colors_rgb: A list of 256 RGB hex strings from the Glasbey set.
        nan_color: The RGB hex string to assign to unseen cell types.

    Returns:
        A list of RGB hex strings in the same order as d_features, suitable
        for directly updating Taxonomy RGB values.
    """
    
    # Step 0: Pre-computation and Data Handling
    
    # Get number of types from the feature distance matrix
    n_types = d_features.shape[0]
    
    # Initialize result list with nan_color for all types
    final_assignments = [nan_color] * n_types
    
    # Identify unseen types (rows with all NaN values)
    nan_type_indices = np.where(np.all(np.isnan(d_features), axis=1))[0]
    
    # Identify valid types (rows with at least some non-NaN values)
    valid_type_indices = np.where(~np.all(np.isnan(d_features), axis=1))[0]
    
    # Create filtered d_features containing only valid types
    d_features_valid = d_features[np.ix_(valid_type_indices, valid_type_indices)]
    
    # If no valid types, return early (all will be nan_color)
    if len(valid_type_indices) == 0:
        return final_assignments
    
    # Step 1: Find Dissimilarity Anchors
    
    # Find feature anchors (most dissimilar pair of valid cell types)
    max_feat_pos = np.unravel_index(np.nanargmax(d_features_valid), d_features_valid.shape)
    valid_idx_A, valid_idx_B = max_feat_pos[0], max_feat_pos[1]
    
    # Find color anchors (most dissimilar pair of Glasbey colors)
    max_color_pos = np.unravel_index(np.argmax(d_colors), d_colors.shape)
    color_idx_1, color_idx_2 = max_color_pos[0], max_color_pos[1]
    
    # If we got the same color index (can happen with diagonal or duplicate maxima)
    # find a different second color by masking out the first one
    if color_idx_1 == color_idx_2:
        d_colors_temp = d_colors.copy()
        d_colors_temp[color_idx_1, :] = 0  # Mask out row
        d_colors_temp[:, color_idx_1] = 0  # Mask out column
        max_color_pos_2 = np.unravel_index(np.argmax(d_colors_temp), d_colors_temp.shape)
        color_idx_2 = max_color_pos_2[1]
    
    # Step 2: Initialize Core Data Structures
    
    assigned_valid_indices = set()
    assigned_color_indices = set()
    available_valid_indices = set(range(len(valid_type_indices)))
    available_color_indices = set(range(len(glasbey_colors_rgb)))
    current_mapping = {}  # Maps valid_idx -> color_idx
    
    # Step 3: Make the First Assignment
    
    # Assign the first anchor
    current_mapping[valid_idx_A] = color_idx_1
    assigned_valid_indices.add(valid_idx_A)
    available_valid_indices.remove(valid_idx_A)
    assigned_color_indices.add(color_idx_1)
    available_color_indices.remove(color_idx_1)
    
    # Assign the second anchor only if it's different and available
    if valid_idx_B in available_valid_indices:
        current_mapping[valid_idx_B] = color_idx_2
        assigned_valid_indices.add(valid_idx_B)
        available_valid_indices.remove(valid_idx_B)
        assigned_color_indices.add(color_idx_2)
        available_color_indices.remove(color_idx_2)
    
    # Step 4: The Main Loop (Accretion)
    
    num_valid_types = len(valid_type_indices)
    
    # Precompute normalization factors with epsilon to prevent division by zero
    epsilon = 1e-9
    max_feat_dist = np.nanmax(d_features_valid)
    max_color_dist = np.max(d_colors)
    
    while len(assigned_valid_indices) < num_valid_types:
        
        # Step 4a: Select the Next Type to Assign
        # Initialize with the first available type to ensure robustness
        next_type_to_assign = next(iter(available_valid_indices))
        distances = d_features_valid[next_type_to_assign, list(assigned_valid_indices)]
        max_avg_dist = np.nanmean(distances)

        # Iterate through the rest of the available types
        for type_idx in available_valid_indices:
            if type_idx == next_type_to_assign:  # Skip the one we used to initialize
                continue

            distances = d_features_valid[type_idx, list(assigned_valid_indices)]
            avg_dist = np.nanmean(distances)

            # Check for NaN and update if current is better
            if not np.isnan(avg_dist) and avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                next_type_to_assign = type_idx
        
        # Step 4b: Find the Best Available Color
        min_cost = float('inf')
        best_color_for_next_type = -1
        
        for candidate_color_idx in available_color_indices:
            total_cost = 0
            
            # Compare this candidate color to all colors already assigned
            for assigned_type_idx, assigned_color_idx in current_mapping.items():
                # Get the feature distance (between types)
                feature_dist = d_features_valid[next_type_to_assign, assigned_type_idx]
                # Get the color distance (between colors)
                color_dist = d_colors[candidate_color_idx, assigned_color_idx]
                
                # Normalize and find the squared error
                norm_feat_dist = feature_dist / (max_feat_dist + epsilon)
                norm_color_dist = color_dist / (max_color_dist + epsilon)
                total_cost += (norm_feat_dist - norm_color_dist)**2
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_color_for_next_type = candidate_color_idx
        
        # Step 4c: Assign and Update
        current_mapping[next_type_to_assign] = best_color_for_next_type
        
        # Update tracking sets
        assigned_valid_indices.add(next_type_to_assign)
        available_valid_indices.remove(next_type_to_assign)
        assigned_color_indices.add(best_color_for_next_type)
        available_color_indices.remove(best_color_for_next_type)
    
    # Step 5: Finalize the Output
    
    # Convert current_mapping from valid indices to actual type indices and assign colors
    for valid_idx, color_idx in current_mapping.items():
        actual_type_idx = valid_type_indices[valid_idx]
        color_hex = glasbey_colors_rgb[color_idx]
        final_assignments[actual_type_idx] = color_hex
    
    return final_assignments

def assign_colors_based_on_contact_matrix(contact_matrix: np.ndarray, abundances: np.ndarray, glasbey_colors: list) -> list:
    """
    Graph-walk color assignment exploiting the Glasbey palette structure.

    Starts at the most-connected type, then walks to adjacent neighbors
    (highest contact to the already-colored set) before jumping globally.
    This ensures spatially proximal types receive maximally distinct colors.

    Parameters:
    - contact_matrix: np.ndarray (N x N), pairwise contact counts.
    - abundances:     np.ndarray (N,), cell counts per type.
    - glasbey_colors: list of N colors ordered so successive entries are
                      maximally different from all previous entries.

    Returns:
    - list: Assigned colors, indexed by type.
    """
    N = len(abundances)
    assigned_colors = [None] * N
    colored_mask = np.zeros(N, dtype=bool)

    working_matrix = contact_matrix.copy().astype(float)
    np.fill_diagonal(working_matrix, 0)

    # Pre-compute total connectivity for each node (used for seed + fallback)
    total_contact = working_matrix.sum(axis=1)

    # --- Seed: most connected node ---
    best_node = int(np.argmax(total_contact))
    assigned_colors[best_node] = glasbey_colors[0]
    colored_mask[best_node] = True

    for color_idx in range(1, N):
        uncolored_idx = np.where(~colored_mask)[0]

        # Contact of each uncolored node to the entire colored set
        contact_to_colored = working_matrix[np.ix_(uncolored_idx, np.where(colored_mask)[0])].sum(axis=1)

        adjacent_mask = contact_to_colored > 0

        if adjacent_mask.any():
            # Walk: pick the adjacent node with the most contact to colored set
            adj_contacts = contact_to_colored[adjacent_mask]
            adj_indices  = uncolored_idx[adjacent_mask]
            max_c = adj_contacts.max()
            candidates = adj_indices[adj_contacts == max_c]
        else:
            # Jump: no neighbors — pick the globally most-connected uncolored node
            tc = total_contact[uncolored_idx]
            max_c = tc.max()
            candidates = uncolored_idx[tc == max_c]

        # Tie-break by abundance
        best_node = int(candidates[np.argmax(abundances[candidates])])
        assigned_colors[best_node] = glasbey_colors[color_idx]
        colored_mask[best_node] = True

    return assigned_colors

class DistanceMatrixOptimizer:
    """
    Class to find optimal permutation between two distance matrices.
    """
    
    def __init__(self, Dopt: np.ndarray, Dclr: np.ndarray, 
                 correlation_type: str = 'pearson'):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        Dopt : np.ndarray
            Reference distance matrix (n x n)
        Dclr : np.ndarray  
            Distance matrix to be permuted (n x n)
        correlation_type : str
            Type of correlation to maximize ('pearson' or 'spearman')
        """
        self.Dopt = np.array(Dopt)
        self.Dclr = np.array(Dclr)
        self.n = self.Dopt.shape[0]
        self.correlation_type = correlation_type
        
        # Validate inputs
        self._validate_inputs()
        
        # Store results
        self.results = {}
        
    def _validate_inputs(self):
        """Validate input matrices."""
        if self.Dopt.shape != self.Dclr.shape:
            raise ValueError("Distance matrices must have the same shape")
        
        if not (self.Dopt.shape[0] == self.Dopt.shape[1]):
            raise ValueError("Distance matrices must be square")
            
        if not np.allclose(self.Dopt, self.Dopt.T):
            warnings.warn("Dopt is not symmetric")
            
        if not np.allclose(self.Dclr, self.Dclr.T):
            warnings.warn("Dclr is not symmetric")
    
    def _compute_correlation(self, perm: np.ndarray) -> float:
        """
        Compute correlation between Dopt and permuted Dclr.
        
        Parameters:
        -----------
        perm : np.ndarray
            Permutation indices
            
        Returns:
        --------
        float : correlation coefficient
        """
        # Apply permutation to both rows and columns
        Dclr_perm = self.Dclr[np.ix_(perm, perm)]
        
        # Extract upper triangular parts (excluding diagonal)
        mask = np.triu(np.ones_like(self.Dopt, dtype=bool), k=1)
        dopt_vec = self.Dopt[mask]
        dclr_vec = Dclr_perm[mask]
        
        if self.correlation_type == 'pearson':
            corr, _ = pearsonr(dopt_vec, dclr_vec)
        elif self.correlation_type == 'spearman':
            corr, _ = spearmanr(dopt_vec, dclr_vec)
        else:
            raise ValueError("correlation_type must be 'pearson' or 'spearman'")
            
        return corr if not np.isnan(corr) else -1.0
    
    def hungarian_method(self) -> Dict:
        """
        Use Hungarian algorithm to minimize cost (maximize correlation).
        Note: This treats the problem as a bipartite matching problem.
        """
        print("Running Hungarian algorithm...")
        start_time = time.time()
        
        # Create cost matrix: cost = 1 - |correlation|
        # We'll use a heuristic: for each pair of points, compute how well
        # their distances to all other points correlate
        cost_matrix = np.zeros((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                # Compare distances from point i in Dopt to distances from point j in Dclr
                dopt_dists = self.Dopt[i, :]
                dclr_dists = self.Dclr[j, :]
                
                if self.correlation_type == 'pearson':
                    corr, _ = pearsonr(dopt_dists, dclr_dists)
                else:
                    corr, _ = spearmanr(dopt_dists, dclr_dists)
                
                cost_matrix[i, j] = 1 - abs(corr) if not np.isnan(corr) else 1.0
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create permutation
        perm = np.zeros(self.n, dtype=int)
        perm[row_ind] = col_ind
        
        correlation = self._compute_correlation(perm)
        runtime = time.time() - start_time
        
        result = {
            'method': 'hungarian',
            'permutation': perm,
            'correlation': correlation,
            'runtime': runtime,
            'cost_matrix': cost_matrix
        }
        
        self.results['hungarian'] = result
        return result
    
    def simulated_annealing(self, max_iter: int = 10000, 
                          initial_temp: float = 1.0,
                          cooling_rate: float = 0.995) -> Dict:
        """
        Use simulated annealing to find optimal permutation.
        """
        print("Running simulated annealing...")
        start_time = time.time()
        
        # Initialize with random permutation
        current_perm = np.random.permutation(self.n)
        current_corr = self._compute_correlation(current_perm)
        
        best_perm = current_perm.copy()
        best_corr = current_corr
        
        temp = initial_temp
        correlations = []
        
        for iteration in range(max_iter):
            # Generate neighbor by swapping two random elements
            new_perm = current_perm.copy()
            i, j = np.random.choice(self.n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            
            new_corr = self._compute_correlation(new_perm)
            
            # Accept or reject
            delta = new_corr - current_corr
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_perm = new_perm
                current_corr = new_corr
                
                if current_corr > best_corr:
                    best_perm = current_perm.copy()
                    best_corr = current_corr
            
            correlations.append(current_corr)
            temp *= cooling_rate
            
            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}: best_corr = {best_corr:.4f}, current_corr = {current_corr:.4f}")
        
        runtime = time.time() - start_time
        
        result = {
            'method': 'simulated_annealing',
            'permutation': best_perm,
            'correlation': best_corr,
            'runtime': runtime,
            'correlations_history': correlations,
            'parameters': {
                'max_iter': max_iter,
                'initial_temp': initial_temp,
                'cooling_rate': cooling_rate
            }
        }
        
        self.results['simulated_annealing'] = result
        return result
    
    def genetic_algorithm(self, population_size: int = 100, 
                         generations: int = 500,
                         mutation_rate: float = 0.1,
                         crossover_rate: float = 0.8) -> Dict:
        """
        Use genetic algorithm to find optimal permutation.
        """
        print("Running genetic algorithm...")
        start_time = time.time()
        
        # Initialize population
        population = [np.random.permutation(self.n) for _ in range(population_size)]
        
        best_perm = None
        best_corr = -np.inf
        correlations = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness = [self._compute_correlation(perm) for perm in population]
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_corr:
                best_corr = fitness[gen_best_idx]
                best_perm = population[gen_best_idx].copy()
            
            correlations.append(best_corr)
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size)
                winner_idx = tournament_indices[np.argmax([fitness[i] for i in tournament_indices])]
                new_population.append(population[winner_idx].copy())
            
            population = new_population
            
            # Crossover and mutation
            for i in range(0, population_size-1, 2):
                if np.random.random() < crossover_rate:
                    # Order crossover (OX)
                    parent1, parent2 = population[i], population[i+1]
                    child1, child2 = self._order_crossover(parent1, parent2)
                    population[i], population[i+1] = child1, child2
                
                # Mutation (swap mutation)
                if np.random.random() < mutation_rate:
                    perm = population[i]
                    idx1, idx2 = np.random.choice(self.n, 2, replace=False)
                    perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
                    
                if np.random.random() < mutation_rate:
                    perm = population[i+1]
                    idx1, idx2 = np.random.choice(self.n, 2, replace=False)
                    perm[idx1], perm[idx2] = perm[idx2], perm[idx1]
            
            if generation % 50 == 0:
                print(f"  Generation {generation}: best_corr = {best_corr:.4f}")
        
        runtime = time.time() - start_time
        
        result = {
            'method': 'genetic_algorithm',
            'permutation': best_perm,
            'correlation': best_corr,
            'runtime': runtime,
            'correlations_history': correlations,
            'parameters': {
                'population_size': population_size,
                'generations': generations,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate
            }
        }
        
        self.results['genetic_algorithm'] = result
        return result
    
    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order crossover for permutations."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child1 = np.full(size, -1)
        child2 = np.full(size, -1)
        
        # Copy segment
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Fill remaining positions
        self._fill_child(child1, parent2, start, end)
        self._fill_child(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_child(self, child: np.ndarray, parent: np.ndarray, start: int, end: int):
        """Helper for order crossover."""
        size = len(child)
        remaining = [x for x in parent if x not in child[start:end]]
        
        # Fill positions after end
        pos = end
        idx = 0
        while pos < size and idx < len(remaining):
            child[pos] = remaining[idx]
            pos += 1
            idx += 1
        
        # Fill positions before start
        pos = 0
        while pos < start and idx < len(remaining):
            child[pos] = remaining[idx]
            pos += 1
            idx += 1
    
    def brute_force(self) -> Dict:
        """
        Brute force search (only feasible for small n).
        """
        if self.n > 8:
            print(f"Brute force not feasible for n={self.n} (too large)")
            return None
            
        print(f"Running brute force search for n={self.n}...")
        start_time = time.time()
        
        best_perm = None
        best_corr = -np.inf
        
        count = 0
        for perm in itertools.permutations(range(self.n)):
            perm = np.array(perm)
            corr = self._compute_correlation(perm)
            
            if corr > best_corr:
                best_corr = corr
                best_perm = perm.copy()
            
            count += 1
            if count % 1000 == 0:
                print(f"  Evaluated {count} permutations, best_corr = {best_corr:.4f}")
        
        runtime = time.time() - start_time
        
        result = {
            'method': 'brute_force',
            'permutation': best_perm,
            'correlation': best_corr,
            'runtime': runtime,
            'total_permutations': count
        }
        
        self.results['brute_force'] = result
        return result
    
    def greedy_heuristic(self) -> Dict:
        """
        Greedy heuristic: iteratively assign points to maximize correlation.
        """
        print("Running greedy heuristic...")
        start_time = time.time()
        
        assigned_opt = set()
        assigned_clr = set()
        perm = np.full(self.n, -1)
        
        for step in range(self.n):
            best_score = -np.inf
            best_i, best_j = None, None
            
            for i in range(self.n):
                if i in assigned_opt:
                    continue
                for j in range(self.n):
                    if j in assigned_clr:
                        continue
                    
                    # Score this assignment based on correlation with already assigned points
                    score = 0
                    count = 0
                    for k in range(self.n):
                        if perm[k] != -1:  # Already assigned
                            # Add correlation between distances
                            score += abs(self.Dopt[i, k] - self.Dclr[j, perm[k]])
                            count += 1
                    
                    # Normalize score (we want to minimize distance differences)
                    if count > 0:
                        score = -score / count  # Negative because we want to minimize
                    else:
                        # For first assignment, use correlation of distance vectors
                        dopt_dists = self.Dopt[i, :]
                        dclr_dists = self.Dclr[j, :]
                        if self.correlation_type == 'pearson':
                            score, _ = pearsonr(dopt_dists, dclr_dists)
                        else:
                            score, _ = spearmanr(dopt_dists, dclr_dists)
                        score = score if not np.isnan(score) else -1
                    
                    if score > best_score:
                        best_score = score
                        best_i, best_j = i, j
            
            # Make assignment
            perm[best_i] = best_j
            assigned_opt.add(best_i)
            assigned_clr.add(best_j)
        
        correlation = self._compute_correlation(perm)
        runtime = time.time() - start_time
        
        result = {
            'method': 'greedy_heuristic',
            'permutation': perm,
            'correlation': correlation,
            'runtime': runtime
        }
        
        self.results['greedy_heuristic'] = result
        return result
    
    def run_all_methods(self, methods: Optional[List[str]] = None) -> Dict:
        """
        Run all available methods and compare results.
        
        Parameters:
        -----------
        methods : List[str], optional
            List of methods to run. If None, runs all appropriate methods.
        """
        if methods is None:
            methods = ['hungarian', 'greedy_heuristic', 'simulated_annealing']
            if self.n <= 8:
                methods.append('brute_force')
            if self.n >= 10:  # GA works better for larger problems
                methods.append('genetic_algorithm')
        
        print(f"Running optimization for {self.n}x{self.n} distance matrices")
        print(f"Correlation type: {self.correlation_type}")
        print(f"Methods to run: {methods}")
        print("="*60)
        
        results = {}
        
        for method in methods:
            try:
                if method == 'hungarian':
                    results[method] = self.hungarian_method()
                elif method == 'simulated_annealing':
                    results[method] = self.simulated_annealing()
                elif method == 'genetic_algorithm':
                    results[method] = self.genetic_algorithm()
                elif method == 'brute_force':
                    results[method] = self.brute_force()
                elif method == 'greedy_heuristic':
                    results[method] = self.greedy_heuristic()
                else:
                    print(f"Unknown method: {method}")
                    continue
                
                print(f"{method}: correlation = {results[method]['correlation']:.4f}, "
                      f"runtime = {results[method]['runtime']:.3f}s")
                
            except Exception as e:
                print(f"Error running {method}: {e}")
                
        print("="*60)
        
        # Find best result
        if results:
            best_method = max(results.keys(), key=lambda k: results[k]['correlation'])
            print(f"Best method: {best_method} (correlation = {results[best_method]['correlation']:.4f})")
            
        return results
    
    def visualize_results(self, save_fig: bool = False, filename: str = None):
        """
        Visualize the optimization results.
        """
        if not self.results:
            print("No results to visualize. Run optimization first.")
            return
        
        # Create subplots
        n_methods = len(self.results)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8))
        
        if n_methods == 1:
            axes = axes.reshape(2, -1)
        
        # Plot original matrices
        im1 = axes[0, 0].imshow(self.Dopt, cmap='viridis')
        axes[0, 0].set_title('Original Dopt')
        axes[0, 0].set_xlabel('Point index')
        axes[0, 0].set_ylabel('Point index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[1, 0].imshow(self.Dclr, cmap='viridis')
        axes[1, 0].set_title('Original Dclr')
        axes[1, 0].set_xlabel('Point index')
        axes[1, 0].set_ylabel('Point index')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Plot results for each method
        for i, (method, result) in enumerate(self.results.items()):
            perm = result['permutation']
            Dclr_perm = self.Dclr[np.ix_(perm, perm)]
            
            im = axes[0, i+1].imshow(Dclr_perm, cmap='viridis')
            axes[0, i+1].set_title(f'{method}\nCorr = {result["correlation"]:.3f}')
            axes[0, i+1].set_xlabel('Point index')
            axes[0, i+1].set_ylabel('Point index')
            plt.colorbar(im, ax=axes[0, i+1])
            
            # Scatter plot of distances
            mask = np.triu(np.ones_like(self.Dopt, dtype=bool), k=1)
            dopt_vec = self.Dopt[mask]
            dclr_vec = Dclr_perm[mask]
            
            axes[1, i+1].scatter(dopt_vec, dclr_vec, alpha=0.6)
            axes[1, i+1].plot([dopt_vec.min(), dopt_vec.max()], 
                             [dopt_vec.min(), dopt_vec.max()], 'r--', alpha=0.8)
            axes[1, i+1].set_xlabel('Dopt distances')
            axes[1, i+1].set_ylabel('Dclr distances (permuted)')
            axes[1, i+1].set_title(f'Distance Correlation\nr = {result["correlation"]:.3f}')
        
        plt.tight_layout()
        
        if save_fig:
            if filename is None:
                filename = f'distance_matrix_optimization_results_{int(time.time())}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        plt.show()
    
    def get_best_result(self) -> Dict:
        """Get the best result across all methods."""
        if not self.results:
            return None
        
        best_method = max(self.results.keys(), key=lambda k: self.results[k]['correlation'])
        return self.results[best_method]
