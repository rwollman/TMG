from TMG.Analysis.TissueGraph import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TMG.Utils.coloru import rgb_array_to_hex
import distinctipy
import igraph as ig

def _oblique_project(coords_2d, z, tilt_angle_deg=30, depth_scale=0.45):
    """
    Oblique (cabinet-style) projection of 2D plane-coordinates at height z.
        screen_x = px + py * cos(α) * depth_scale
        screen_y = py * sin(α) * depth_scale + z
    """
    alpha = np.radians(tilt_angle_deg)
    px = coords_2d[:, 0]
    py = coords_2d[:, 1]
    sx = px + py * np.cos(alpha) * depth_scale
    sy = py * np.sin(alpha) * depth_scale + z
    return np.column_stack([sx, sy])


def _normalise_coords(coords, margin=0.06):
    coords = coords.copy()
    for dim in range(2):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        span = hi - lo if hi > lo else 1.0
        coords[:, dim] = margin + (1 - 2 * margin) * (coords[:, dim] - lo) / span
    return coords


def _cell_layout(cmat, min_r_threshold=0.5, niter=500):
    """FR layout on cell-cell correlation graph. Returns (n_cells, 2) coords."""
    n = cmat.shape[0]
    g = ig.Graph()
    g.add_vertices(n)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = cmat[i, j]
            if r > min_r_threshold:
                edges.append((i, j))
                weights.append(float(10 ** r))
    g.add_edges(edges)
    if weights:
        g.es["weight"] = weights
    layout = g.layout_fruchterman_reingold(
        weights="weight" if weights else None, niter=niter)
    return _normalise_coords(np.array(layout.coords), margin=0.05)


def _cluster_cells(cmat, min_r_threshold=0.5):
    """Louvain clustering on cell-cell graph. Returns integer label array."""
    n = cmat.shape[0]
    g = ig.Graph()
    g.add_vertices(n)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = cmat[i, j]
            if r > min_r_threshold:
                edges.append((i, j))
                weights.append(float(r))
    g.add_edges(edges)
    if weights:
        g.es["weight"] = weights
    membership = g.community_multilevel(
        weights="weight" if weights else None).membership
    return np.array(membership)


def _region_layer_layout(region_feature_mat, cell_coords, node_half=0.022,
                         niter=200, margin=0.05):
    """
    Place each region at its weighted-mean cell position, then repel overlaps.
    Returns (n_regions, 2) normalised to [margin, 1-margin]².
    """
    n_regions, n_cells = region_feature_mat.shape
    region_coords = np.zeros((n_regions, 2))
    for r in range(n_regions):
        w = region_feature_mat[r, :]
        total = w.sum()
        if total < 1e-9:
            region_coords[r] = [0.5, 0.5]
        else:
            region_coords[r, 0] = np.dot(w, cell_coords[:, 0]) / total
            region_coords[r, 1] = np.dot(w, cell_coords[:, 1]) / total
    min_sep = node_half * 2.5
    for _ in range(niter):
        moved = False
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                dx = region_coords[j, 0] - region_coords[i, 0]
                dy = region_coords[j, 1] - region_coords[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_sep and dist > 1e-9:
                    push = (min_sep - dist) / 2.0
                    nx_, ny_ = dx / dist, dy / dist
                    region_coords[i, 0] -= nx_ * push
                    region_coords[i, 1] -= ny_ * push
                    region_coords[j, 0] += nx_ * push
                    region_coords[j, 1] += ny_ * push
                    moved = True
        if not moved:
            break
    for dim in range(2):
        lo, hi = region_coords[:, dim].min(), region_coords[:, dim].max()
        span = hi - lo if hi > lo else 1.0
        region_coords[:, dim] = margin + (1 - 2 * margin) * (region_coords[:, dim] - lo) / span
    return region_coords

def _gene_layout(cell_gene_mat, cell_coords, cluster_labels,
                 n_neighbors=10,
                 node_r=0.012, component_scale=0.06, niter=80, margin=0.05):
    """
    Cell-type anchored gene layout.

    Each gene is placed at the weighted-average XY position of the cell-type
    (cluster) centroids it maps to.  Per-cluster weight is the sum of
    cell_gene_mat weights for all cells belonging to that cluster.  Only
    clusters with weight > 0 contribute.  Genes with no positive weight are
    placed on the periphery in a sunflower pattern.
    """
    n_cells, n_genes = cell_gene_mat.shape
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    n_clumps = cluster_labels.max() + 1
    center = np.array([0.5, 0.5], dtype=float)

    # Cluster centroids in the cell plane
    cluster_centers = np.array([
        cell_coords[cluster_labels == k].mean(axis=0)
        for k in range(n_clumps)
    ])

    # Per-gene, per-cluster total weight: shape (n_genes, n_clumps)
    cluster_weights = np.zeros((n_genes, n_clumps), dtype=float)
    for k in range(n_clumps):
        mask = cluster_labels == k
        cluster_weights[:, k] = cell_gene_mat[mask, :].sum(axis=0)

    coords = np.zeros((n_genes, 2), dtype=float)
    unmapped = []

    for g in range(n_genes):
        w = cluster_weights[g]
        total = w.sum()
        if total <= 0:
            unmapped.append(g)
        else:
            coords[g] = (w / total) @ cluster_centers

    # Unmapped genes → sunflower ring on the periphery
    n_unmapped = len(unmapped)
    if n_unmapped > 0:
        r_periph = 0.5 - margin - node_r
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for i, g in enumerate(unmapped):
            theta = i * golden
            coords[g] = center + r_periph * np.array([np.cos(theta), np.sin(theta)])

    print(
        f"  Gene layout: {n_genes - n_unmapped} mapped genes (positioned by cell-type centroid), "
        f"{n_unmapped} unmapped → periphery"
    )
    return np.clip(coords, margin, 1 - margin)

def plot_three_layers(
    region_cell_mat,    # (n_regions, n_cells)
    cell_gene_mat,      # (n_cells, n_genes)
    cmat,               # (n_cells, n_cells) cell-cell correlation matrix
    rgb_region,         # n_regions hex colours
    rgb_cell,           # n_cells hex colours
    rgb_gene=None,      # n_genes hex colours (None -> grey)
    # ── layout ──────────────────────────────────────────────────────────
    min_r_threshold=0.5,
    cell_niter=500,
    cell_scale=0.7,
    gene_niter=80,
    component_scale=0.06,
    gene_n_neighbors=10,
    tilt_angle_deg=66,
    depth_scale=0.35,
    layer_gap=0.4,
    # ── display ─────────────────────────────────────────────────────────
    region_weight_threshold=0.05,
    gene_weight_threshold=0.1,
    hide_unconnected_genes=False,
    node_radius=0.012,      # cell circle radius (canvas [0,1] coords)
    gene_radius=0.005,      # gene circle radius (canvas [0,1] coords)
    region_half=0.012,      # region square half-side (canvas [0,1] coords)
    edge_alpha=0.35,
    edge_width_range=(0.3, 1.5),
    draw_layer_frames=True,
    cell_frame_color="#dde8ff",
    region_frame_color="#ffdde0",
    gene_frame_color="#ddffee",
    frame_alpha=0.3,
    figsize=(10, 12),
    canvas_margin=0.04,
    title=None,
    ax=None,
):
    """
    Three-layer pseudo-3D layout (oblique/cabinet projection):
      - Layer 0 (bottom) : genes   — FR layout on gene-gene co-expression,
                                      seeded by weighted-mean cell positions
      - Layer 1 (middle) : cells   — FR layout on cell-cell correlation
      - Layer 2 (top)    : regions — weighted-mean cell positions + repulsion

    All nodes are solid shapes (no pie / treemap composition).

    Parameters
    ----------
    region_cell_mat       : (n_regions, n_cells)
    cell_gene_mat         : (n_cells, n_genes)
    cmat                  : (n_cells, n_cells) cell-cell correlation matrix
    rgb_region, rgb_cell, rgb_gene : hex colour arrays
    gene_min_r            : correlation threshold for gene-gene FR graph
    gene_weight_threshold : min cell_gene_mat value to draw a cell-gene edge
    """
    region_cell_mat = np.array(region_cell_mat, dtype=float)
    cell_gene_mat   = np.array(cell_gene_mat,   dtype=float)
    cmat            = np.array(cmat,            dtype=float)
    n_regions, n_cells = region_cell_mat.shape
    _nc, n_genes       = cell_gene_mat.shape
    assert n_cells == _nc, f"cell count mismatch: {n_cells} vs {_nc}"
    rgb_region = list(rgb_region)
    rgb_cell   = list(rgb_cell)
    if rgb_gene is None:
        rgb_gene = ["#999999"] * n_genes
    else:
        rgb_gene = list(rgb_gene)

    # ── Stage 1: cell FR layout (middle plane) ───────────────────────────
    print("Stage 1: cell FR layout…")
    cell_coords_local = _cell_layout(cmat, min_r_threshold, cell_niter)
    cx0, cy0 = 0.5, 0.5
    cell_coords = (cell_coords_local - np.array([cx0, cy0])) * cell_scale + np.array([cx0, cy0])

    # Intra-cluster repulsion to reduce crowding
    cluster_labels = _cluster_cells(cmat, min_r_threshold)
    n_clumps = cluster_labels.max() + 1
    print(f"  → {n_clumps} clusters")
    min_sep = node_radius * 2.6
    for _ in range(50):
        moved = False
        for k in range(n_clumps):
            members = np.where(cluster_labels == k)[0]
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    i, j = members[ii], members[jj]
                    dx = cell_coords[j, 0] - cell_coords[i, 0]
                    dy = cell_coords[j, 1] - cell_coords[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < min_sep and dist > 1e-9:
                        push = (min_sep - dist) / 2.0
                        nx_, ny_ = dx / dist, dy / dist
                        cell_coords[i, 0] -= nx_ * push
                        cell_coords[i, 1] -= ny_ * push
                        cell_coords[j, 0] += nx_ * push
                        cell_coords[j, 1] += ny_ * push
                        moved = True
        if not moved:
            break

    # ── Stage 2: region layout (top plane) ───────────────────────────────
    print("Stage 2: region layer layout…")
    region_coords_plane = _region_layer_layout(
        region_cell_mat, cell_coords,
        node_half=region_half, niter=200, margin=0.05)

    # ── Stage 3: gene layout (bottom plane) ──────────────────────────────
    print("Stage 3: gene layer layout (projected edge springs)…")
    gene_coords_plane = _gene_layout(
        cell_gene_mat, cell_coords, cluster_labels,
        n_neighbors=gene_n_neighbors,
        node_r=gene_radius, component_scale=component_scale,
        niter=gene_niter)

    # ── Oblique projection ────────────────────────────────────────────────
    # genes: z=0 (bottom), cells: z=layer_gap (middle), regions: z=2*layer_gap (top)
    gene_screen   = _oblique_project(gene_coords_plane,   z=0,             tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)
    cell_screen   = _oblique_project(cell_coords,         z=layer_gap,     tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)
    region_screen = _oblique_project(region_coords_plane, z=2 * layer_gap, tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)

    plane_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    frame_gene_screen   = _oblique_project(plane_corners, z=0,             tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)
    frame_cell_screen   = _oblique_project(plane_corners, z=layer_gap,     tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)
    frame_region_screen = _oblique_project(plane_corners, z=2 * layer_gap, tilt_angle_deg=tilt_angle_deg, depth_scale=depth_scale)

    # ── Canvas normalisation ──────────────────────────────────────────────
    all_pts = np.vstack([gene_screen, cell_screen, region_screen,
                         frame_gene_screen, frame_cell_screen, frame_region_screen])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = hi - lo
    canvas_scale  = (1 - 2 * canvas_margin) / span.max()
    screen_center = (lo + hi) / 2
    canvas_offset = np.array([0.5, 0.5]) - screen_center * canvas_scale

    def to_canvas(pts):
        return pts * canvas_scale + canvas_offset

    gene_canvas         = to_canvas(gene_screen)
    cell_canvas         = to_canvas(cell_screen)
    region_canvas       = to_canvas(region_screen)
    frame_gene_canvas   = to_canvas(frame_gene_screen)
    frame_cell_canvas   = to_canvas(frame_cell_screen)
    frame_region_canvas = to_canvas(frame_region_screen)

    # Shape transform matrix (oblique projection of node outlines)
    alpha_rad = np.radians(tilt_angle_deg)
    M_shape = canvas_scale * np.array([
        [1.0,  np.cos(alpha_rad) * depth_scale],
        [0.0,  np.sin(alpha_rad) * depth_scale],
    ])
    r_cell_draw = node_radius / canvas_scale
    r_gene_draw = gene_radius / canvas_scale
    half_draw   = region_half / canvas_scale

    # ── Edge lists ────────────────────────────────────────────────────────
    rc_edges, rc_weights = [], []
    for r in range(n_regions):
        for c in range(n_cells):
            w = region_cell_mat[r, c]
            if w > region_weight_threshold:
                rc_edges.append((r, c))
                rc_weights.append(float(w))

    cg_edges, cg_weights = [], []
    for c in range(n_cells):
        for g in range(n_genes):
            w = cell_gene_mat[c, g]
            if w > gene_weight_threshold:
                cg_edges.append((c, g))
                cg_weights.append(float(w))

    connected_genes = {g for (_, g) in cg_edges}
    n_hidden = n_genes - len(connected_genes) if hide_unconnected_genes else 0
    print(f"  Edges: {len(rc_edges)} region-cell, {len(cg_edges)} cell-gene"
          + (f"  ({n_hidden} unconnected genes hidden)" if n_hidden else ""))

    def _lw(w, w_lo, w_span, lw_min, lw_max):
        return lw_min + (lw_max - lw_min) * (w - w_lo) / w_span

    # ── Figure ────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    # ── Layer frames ──────────────────────────────────────────────────────
    if draw_layer_frames:
        from matplotlib.patches import Polygon as MplPolygon
        ax.add_patch(MplPolygon(frame_gene_canvas, closed=True,
                                facecolor=gene_frame_color, edgecolor="#88cc99",
                                linewidth=0.8, alpha=frame_alpha, zorder=0))
        ax.add_patch(MplPolygon(frame_cell_canvas, closed=True,
                                facecolor=cell_frame_color, edgecolor="#8899cc",
                                linewidth=0.8, alpha=frame_alpha, zorder=0))
        ax.add_patch(MplPolygon(frame_region_canvas, closed=True,
                                facecolor=region_frame_color, edgecolor="#cc8899",
                                linewidth=0.8, alpha=frame_alpha, zorder=0))

    # ── Edges: cell -> gene ───────────────────────────────────────────────
    if cg_edges:
        cg_w = np.array(cg_weights)
        w_lo, w_hi = cg_w.min(), cg_w.max()
        w_span = max(w_hi - w_lo, 1e-9)
        lw_min, lw_max = edge_width_range
        for (c, g), w in zip(cg_edges, cg_weights):
            ax.plot(
                [cell_canvas[c, 0], gene_canvas[g, 0]],
                [cell_canvas[c, 1], gene_canvas[g, 1]],
                color=rgb_cell[c],
                linewidth=_lw(w, w_lo, w_span, lw_min, lw_max),
                alpha=edge_alpha, zorder=1)

    # ── Edges: region -> cell ─────────────────────────────────────────────
    if rc_edges:
        rc_w = np.array(rc_weights)
        w_lo, w_hi = rc_w.min(), rc_w.max()
        w_span = max(w_hi - w_lo, 1e-9)
        lw_min, lw_max = edge_width_range
        for (r, c), w in zip(rc_edges, rc_weights):
            ax.plot(
                [region_canvas[r, 0], cell_canvas[c, 0]],
                [region_canvas[r, 1], cell_canvas[c, 1]],
                color=rgb_region[r],
                linewidth=_lw(w, w_lo, w_span, lw_min, lw_max),
                alpha=edge_alpha, zorder=2)

    # ── Gene nodes — bottom layer, solid oblique triangles ────────────────
    # Equilateral triangle, circumradius = 1, pointing upward
    unit_tri = np.array([
        [ 0.0,            1.0 ],
        [-np.sqrt(3)/2,  -0.5 ],
        [ np.sqrt(3)/2,  -0.5 ],
    ])
    for g in range(n_genes):
        if hide_unconnected_genes and g not in connected_genes:
            continue
        cx, cy = gene_canvas[g]
        pts = (r_gene_draw * unit_tri) @ M_shape.T + np.array([cx, cy])
        ax.fill(pts[:, 0], pts[:, 1], color=rgb_gene[g], zorder=3)

    # ── Cell nodes — middle layer, solid oblique ellipses ─────────────────
    thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    unit_circle = np.column_stack([np.cos(thetas), np.sin(thetas)])
    for c in range(n_cells):
        cx, cy = cell_canvas[c]
        pts = (r_cell_draw * unit_circle) @ M_shape.T + np.array([cx, cy])
        ax.fill(pts[:, 0], pts[:, 1], color=rgb_cell[c], zorder=4)

    # ── Region nodes — top layer, solid oblique squares ───────────────────
    corners_unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    for r in range(n_regions):
        cx, cy = region_canvas[r]
        pts = (half_draw * corners_unit) @ M_shape.T + np.array([cx, cy])
        ax.fill(pts[:, 0], pts[:, 1], color=rgb_region[r], zorder=5)

    plt.tight_layout()
    return fig, ax

