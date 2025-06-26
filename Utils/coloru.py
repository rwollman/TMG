# Removed colormath dependency due to numpy.asscalar incompatibility
# from colormath.color_objects import sRGBColor, LabColor
# from colormath.color_conversions import convert_color
# from colormath.color_diff import delta_e_cie2000, delta_e_cie1976
from scipy.cluster import hierarchy 
import numpy as np

import random
import itertools

import umap

import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.spatial.distance import pdist

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
        clr_best = np.reshape(clr_best,(sz,3))
    clr_rgb = np.zeros(clr_best.shape)
    for i in range(clr_rgb.shape[0]):
        clr_lab = LabColor(clr_best[i,0]*100,(clr_best[i,1]-0.5)*255,(clr_best[i,2]-0.5)*255)
        clr_rgb[i,:] = np.array(convert_color(clr_lab,sRGBColor).get_value_tuple())

    clr_rgb = np.clip(clr_rgb,0,1)
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