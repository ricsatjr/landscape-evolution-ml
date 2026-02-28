#!/usr/bin/env python
# coding: utf-8
"""
02_extract_features.py
======================
Extracts topographic features from synthetic landscape elevation grids for
training the LE-ML inverse model pipeline described in:

    Saturay et al. (2025). [Title]. JGR: Earth Surface. [DOI]

This script implements the feature generation step (Section 3.1, Feature
generation) and operates in two stages that can be run independently:

Stage 1 -- rasnet
    Loads a steady-state elevation grid (.npy), adds measurement noise,
    runs D8 flow routing, constructs the drainage network, and saves an
    intermediate rasnet (.pkl) file containing the Landlab grid, watershed
    mask, NetworkX drainage graph, and LE parameter labels.

    Intermediate rasnet files are saved because Stage 1 is computationally
    expensive (flow routing + network construction). Saving the intermediate
    objects allows Stage 2 (feature computation) to be rerun independently
    without re-executing the flow routing, which is useful when exploring
    alternative feature definitions or aggregation methods.

Stage 2 -- features
    Loads rasnet (.pkl) files and computes the full 39-feature vector for
    each landscape: 28 raster-based features (Table 2 in paper) and 11
    network-based features (Table 3 in paper). Saves label-feature
    DataFrames as .pkl files ready for model training.

Usage
-----
    # Run Stage 1 only (rasnet extraction):
    python 02_extract_features.py --stage rasnet \\
        --data-dir data/landscapes/ --job-id 42 --output-dir data/rasnet/

    # Run Stage 2 only (feature computation from saved rasnets):
    python 02_extract_features.py --stage features \\
        --data-dir data/rasnet/ --job-id 42 --output-dir data/features/

    # Run both stages end-to-end:
    python 02_extract_features.py --stage all \\
        --data-dir data/landscapes/ --job-id 42 --output-dir data/features/

Arguments
---------
    --stage         Pipeline stage to run: 'rasnet', 'features', or 'all'
    --data-dir      Input data directory
    --job-id        Job identifier for filtering files (int, or 'all' for all)
    --output-dir    Directory for output files (default: current directory)
    --elev-err      Elevation measurement error magnitude in meters (default: 10)
    --ts-index      Time step index to extract from elevation time series
                    (default: 99, i.e., the final steady-state snapshot)

Output (Stage 1)
----------------
    rasnet-n{elev_err}-{job_id}-{landscape_idx}-{ts_index}.pkl
        Pickle file containing:
        [le_params, mg, mask, chNet, wsOutlets, wsOutletsDA]

Output (Stage 2)
----------------
    features-{job_id}.pkl
        Pandas DataFrame with LE parameter labels and 39 topographic features
        for all landscapes in the specified job.

Feature Names
-------------
    Raster-based (28 features, Table 2 in paper):
        Z_mean, Z_cv, Z_med, Z_max, Z_skew, Z_kurt
        grd_mean, grd_std, grd_med, grd_max, grd_skew, grd_kurt
        crv_mean, crv_std, crv_min, crv_med, crv_max, crv_skew, crv_kurt
        htcrv_mean, htcrv_std, htcrv_min, htcrv_med, htcrv_max,
        htcrv_skew, htcrv_kurt
        hyp_int
        Ly

    Network-based (11 features, Table 3 in paper):
        n_nodes, Rb, Rl, Rb0, Rl0
        n0, L0, l0_mean, rlf0_mean, grd0_mean, path_max

See DATA_PROVENANCE.md for the elevation noise seed convention used
during feature extraction.

Dependencies
------------
    landlab >= 2.9.2, networkx >= 3.4.2, numpy, scipy, pandas
    See environment.yml for pinned versions.
"""

import sys
import time
import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import ndimage
import scipy.stats as st

from landlab import RasterModelGrid
from landlab.components import (
    FlowAccumulator,
    DepressionFinderAndRouter,
    ChannelProfiler,
)
from landlab.utils.watershed import get_watershed_masks


# =============================================================================
# FIXED CONSTANTS
# =============================================================================

ELEV_ERR   = 10.0  # Default elevation measurement error magnitude (m)
                    # Represents ±10m random noise simulating DEM uncertainty
                    # (Section 3.1, preprocessing; Mercuri et al., 2006)
TS_INDEX   = 99    # Default time step index for steady-state snapshot
                    # Index 99 = final time step (t = T_ss); zero-based
DX         = 30    # Grid cell resolution (m); must match forward model

MEDIAN_FILTER_SIZE = 5   # Size of median filter applied before gradient and
                          # curvature computation (Section 3.1 in paper)
HTCURV_MAX_GRAD    = 20  # Maximum gradient (degrees) for hilltop curvature
                          # cells (Hurst et al., 2012; Section 3.1 in paper)


# =============================================================================
# UTILITY
# =============================================================================

def make_elev_seed(job_id, landscape_idx, ts_index):
    """
    Compute the random seed for elevation measurement noise added during
    feature extraction (Stage 1).

    The seed is constructed by string-concatenating job_id, landscape_idx,
    and ts_index and adding 1:

        elev_seed = int(str(job_id) + str(landscape_idx) + str(ts_index)) + 1

    This convention is deliberately different from the landscape generation
    seed (100 * job_id + landscape_idx) to ensure that the measurement noise
    added during feature extraction is statistically independent of the
    initial elevation noise used during landscape evolution. See
    DATA_PROVENANCE.md for full documentation of both seed conventions.

    Parameters
    ----------
    job_id : int
        Job identifier.
    landscape_idx : int
        Landscape index within the job.
    ts_index : int
        Time step index of the elevation snapshot.

    Returns
    -------
    int
        Random seed for elevation noise.

    Examples
    --------
    >>> make_elev_seed(9916, 1, 99)
    991620
    """
    return int(f"{job_id}{landscape_idx}{ts_index}") + 1


# =============================================================================
# STAGE 1: GRID LOADING AND DRAINAGE NETWORK CONSTRUCTION
# =============================================================================

def load_elev_grid(fp, dx=DX, elev_err=ELEV_ERR,
                   elev_seed=None, add_noise=True):
    """
    Load a steady-state elevation grid from a .npy file into a Landlab
    RasterModelGrid, and optionally add elevation measurement noise.

    The .npy file contains core node elevations only. This function pads the
    array with a 1-cell boundary (zero elevation) to restore the full grid
    including open boundary rows, then sets up the same boundary conditions
    used during landscape generation (closed left/right, open top/bottom).

    Noise simulates DEM measurement uncertainty (Section 3.1 in paper).
    Noise is drawn from a uniform distribution over [-elev_err, +elev_err],
    i.e., (np.random.rand() - 0.5) * elev_err * 2. The resulting negative
    elevations are set to zero to prevent unphysical values.

    Parameters
    ----------
    fp : str or Path
        Path to .npy elevation grid file.
    dx : float
        Grid cell resolution (m). Default: DX (30 m).
    elev_err : float
        Half-range of uniform elevation noise (m). Default: ELEV_ERR (10 m).
    elev_seed : int or None
        Random seed for elevation noise. Use make_elev_seed() to compute.
        See DATA_PROVENANCE.md for seed convention.
    add_noise : bool
        If True, add random elevation noise. Default: True.

    Returns
    -------
    mg : RasterModelGrid
        Landlab grid with 'topographic__elevation' field populated.
    """
    # Load core node elevations and pad with 1-cell zero boundary
    arr = np.load(fp)
    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)

    nY, nX = arr.shape
    mg = RasterModelGrid((nY, nX), dx)

    # Boundary conditions matching landscape generation (01_generate_landscapes.py)
    mg.set_closed_boundaries_at_grid_edges(
        right_is_closed=True,
        top_is_closed=False,
        left_is_closed=True,
        bottom_is_closed=False,
    )

    # Initialize elevation field
    z = mg.add_zeros("topographic__elevation", at="node")
    z += arr.flatten()

    # Add elevation measurement noise (Section 3.1, preprocessing)
    if add_noise:
        np.random.seed(elev_seed)   # See DATA_PROVENANCE.md
        noise = (np.random.rand(mg.number_of_nodes) - 0.5) * elev_err * 2
        z += noise

    # Enforce boundary and non-negative elevation conditions
    z[mg.status_at_node != 0] = 0
    z[:] = np.where(z >= 0, z, 0)
    mg.at_node['topographic__elevation'] = z

    return mg


def extract_channels_from_grid(mg,
                                minimum_cell_contrib_channel=1,
                                minimum_cell_contrib_outlet=1,
                                number_of_watersheds=None):
    """
    Run D8 flow routing and extract the drainage network from a Landlab grid.

    Flow routing uses D8 with depression filling (DepressionFinderAndRouter).
    The ChannelProfiler is run with no drainage area threshold
    (minimum_cell_contrib_channel=1), which extends the network into the
    hillslope domain and enables zero-order edge detection — the key
    methodological choice described in Section 3.1 of the paper.

    Watersheds whose outlets drain into, or are adjacent to, the closed
    left/right grid boundaries are excluded from the network and masked out
    of the raster analysis. These catchments have underestimated drainage
    areas due to the boundary conditions.

    Parameters
    ----------
    mg : RasterModelGrid
        Grid with 'topographic__elevation' field.
    minimum_cell_contrib_channel : int
        Minimum contributing cells to define a channel cell. Default: 1
        (no threshold — extends network into hillslopes).
    minimum_cell_contrib_outlet : int
        Minimum contributing cells to define a watershed outlet. Default: 1.
    number_of_watersheds : int or None
        Number of watersheds to extract. None = all watersheds.

    Returns
    -------
    mg : RasterModelGrid
        Updated grid with drainage fields added at node:
        'drainage_area', 'flow__receiver_node', 'watershed__outlet'.
    rastermask : np.ndarray of bool, shape (nY, nX)
        True for valid core nodes not connected to closed boundaries.
    chNet : nx.DiGraph
        Directed acyclic graph of the drainage network with Strahler orders
        (including zero order) assigned to edges.
    wsOutlets : np.ndarray of int
        Node IDs of valid watershed outlets.
    wsOutletsDA : list of int
        Drainage areas (in cells) of valid watershed outlets.
    """
    # Flow routing
    fac = FlowAccumulator(
        mg,
        flow_director="FlowDirectorD8",
        depression_finder=DepressionFinderAndRouter,
        routing='D8',
    )
    fac.run_one_step()

    # Channel extraction with no drainage area threshold (Section 3.1)
    channels = ChannelProfiler(
        mg,
        number_of_watersheds=number_of_watersheds,
        minimum_outlet_threshold=mg.dx * mg.dy * minimum_cell_contrib_outlet,
        main_channel_only=False,
        minimum_channel_threshold=mg.dx * mg.dy * minimum_cell_contrib_channel,
    )
    channels.run_one_step()

    # Identify and exclude boundary-connected watersheds
    mg.at_node['watershed__outlet'] = get_watershed_masks(mg)
    outlets = mg.at_node['watershed__outlet'].reshape(mg.nodes.shape)
    exclude_outlets = np.unique(np.concatenate([
        outlets[:, 0:2].flatten(),
        outlets[:, -2:].flatten(),
    ]))

    # Build raster mask: True = valid core node, not boundary-connected
    rastermask = (
        (~np.isin(outlets, exclude_outlets)) *
        (np.isin(mg.nodes, mg.core_nodes))
    )

    # Build NetworkX drainage graph and assign modified Strahler orders
    chNet = _build_nx_graph(mg, channels, exclude_outlets)
    chNet = _assign_stream_order(chNet)

    # Valid outlets and their drainage areas
    wsOutlets = np.sort(
        list(set(mg.at_node['watershed__outlet']) - set(exclude_outlets))
    )
    wsOutletsDA = [
        int(da / (mg.dx * mg.dy))
        for da in mg.at_node['drainage_area'][wsOutlets]
    ]

    return mg, rastermask, chNet, wsOutlets, wsOutletsDA


def _build_nx_graph(mg, channels, exclude_outlets):
    """
    Convert a Landlab ChannelProfiler result to a NetworkX DiGraph.

    Each graph edge corresponds to one branch in the channel network.
    Edge attributes: length (m), relief (m), outlet (node ID).

    Parameters
    ----------
    mg : RasterModelGrid
    channels : ChannelProfiler
    exclude_outlets : array-like
        Outlet node IDs to exclude (boundary-connected watersheds).

    Returns
    -------
    G : nx.DiGraph or None
        Directed acyclic graph, or None if cycles are detected.
    """
    G = nx.DiGraph()

    for outlet in channels.data_structure.keys():
        if outlet in exclude_outlets:
            continue
        for branch in channels.data_structure[outlet].keys():
            # Reverse to get upstream → downstream ordering
            ids = list(channels.data_structure[outlet][branch]['ids'][::-1])
            if len(ids) <= 1:
                continue
            distances = channels.data_structure[outlet][branch]['distances'][::-1]

            # Add terminal and outlet nodes with outlet-distance attribute
            G.add_nodes_from([
                (ids[0],  {'o_dst': distances[0]}),
                (ids[-1], {'o_dst': distances[-1]}),
            ])

            # Add edge with geometric and topographic attributes
            elev = mg.at_node['topographic__elevation']
            G.add_edges_from([(
                ids[0], ids[-1],
                {
                    'length':  np.round(distances[0] - distances[-1], 2),
                    'relief':  np.round(elev[ids[0]] - elev[ids[-1]], 2),
                    'outlet':  outlet,
                }
            )])

    # Validate: all components must be directed acyclic graphs
    for component in nx.weakly_connected_components(G):
        if not nx.is_directed_acyclic_graph(G.subgraph(component)):
            print('Warning: channel network contains cycles — returning None')
            return None

    return G


def _assign_stream_order(G):
    """
    Assign modified Strahler stream order to graph edges.

    The standard Strahler ordering is modified so that terminal (source)
    branches are assigned order zero rather than order one. This emphasizes
    that the uppermost drainages may not be true streams but rather hillslope
    flow paths — i.e., zero-order basins (Tsuboyama et al., 2000). This
    extension into the hillslope domain is central to the zero-order drainage
    finding in Section 3.5 of the paper.

    Ordering rules:
      - Source nodes (in-degree = 0): assigned order 0
      - Nodes with one upstream branch: inherit upstream order
      - Nodes where all upstream branches share the same order: order + 1
      - Nodes where upstream branches differ in order: maximum upstream order

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    G : nx.DiGraph
        Graph with 'str_order' attribute added to all edges.
    """
    node_order = {}

    for component in nx.weakly_connected_components(G):
        sG = G.subgraph(component)
        for node in nx.topological_sort(sG):
            if sG.in_degree(node) == 0:
                node_order[node] = 0
            else:
                upstream_orders = np.array(
                    [node_order[pred] for pred in sG.predecessors(node)],
                    dtype=int,
                )
                if len(upstream_orders) == 1:
                    node_order[node] = upstream_orders[0]
                else:
                    max_order = np.max(upstream_orders)
                    if np.all(upstream_orders == max_order):
                        node_order[node] = max_order + 1
                    else:
                        node_order[node] = max_order

    for u, v in G.edges():
        G.edges[u, v]['str_order'] = node_order[u]

    return G


# =============================================================================
# STAGE 2: RASTER FEATURE EXTRACTION
# =============================================================================

def compute_raster_features(mg, mask):
    """
    Compute 28 raster-based topographic features from a Landlab grid.

    Features are derived from four spatial fields — elevation, gradient,
    curvature, and hilltop curvature — each computed over the valid (masked)
    core nodes. Landscape width Ly and hypsometric integral are also included.

    Feature names and symbols follow Table 2 in Saturay et al. (2025):

    Elevation (Z): Z_mean, Z_cv, Z_med, Z_max, Z_skew, Z_kurt
        Coefficient of variation (cv = std/mean) is used instead of std
        for elevation, as it is more indicative of terrain roughness.

    Gradient (grd): grd_mean, grd_std, grd_med, grd_max, grd_skew, grd_kurt
        Gradient magnitude computed from finite differences on a 5×5
        median-filtered elevation grid (Section 3.1).

    Curvature (crv): crv_mean, crv_std, crv_min, crv_med, crv_max,
                     crv_skew, crv_kurt
        Negative Laplacian of the smoothed elevation grid (divided by dx²).
        Convention: positive = convex (ridges), negative = concave (channels).

    Hilltop curvature (htcrv): htcrv_mean, htcrv_std, htcrv_min, htcrv_med,
                                htcrv_max, htcrv_skew, htcrv_kurt
        Subset of curvature for cells with: drainage area = 1 cell (no
        contributing cells), positive curvature, and gradient < 20°.
        After Hurst et al. (2012); Section 3.1 in paper.

    Hypsometric integral (hyp_int): area under the normalized elevation-area
        curve (Strahler, 1952).

    Landscape width (Ly): grid width in the y-direction (m). Included as a
        feature because it is a measurable physical property of the landscape
        that is also an LEM input parameter.

    Parameters
    ----------
    mg : RasterModelGrid
        Grid with 'topographic__elevation' and 'drainage_area' fields.
    mask : np.ndarray of bool, shape (nY, nX)
        Valid node mask from extract_channels_from_grid().

    Returns
    -------
    features : dict
        Dictionary of 28 raster feature values keyed by paper symbol names.
    """
    features = {}

    # --- Compute spatial fields ---
    z   = mg.at_node['topographic__elevation'].copy().reshape(mg.shape)
    da  = (mg.at_node['drainage_area'] / mg.dx ** 2).reshape(mg.shape)

    # Smooth elevation before derivative computation (5×5 median filter)
    zs  = ndimage.median_filter(z, size=MEDIAN_FILTER_SIZE)

    # Gradient: magnitude of finite-difference first derivatives (Section 3.1)
    dz_dy, dz_dx = np.gradient(zs)
    grd = np.sqrt((dz_dx / mg.dx) ** 2 + (dz_dy / mg.dx) ** 2)

    # Curvature: negative Laplacian (convention: +ve = convex; Section 3.1)
    crv = -ndimage.laplace(zs) / mg.dx ** 2

    # Hilltop curvature: convex cells, low gradient, no contributing area
    # (Hurst et al., 2012; Section 3.1)
    htcrv = np.where(da  == 1,   crv,  np.nan)   # no contributing cells
    htcrv = np.where(crv >  0,   htcrv, np.nan)  # convex surfaces only
    htcrv = np.where(
        np.degrees(np.arctan(grd)) < HTCURV_MAX_GRAD, htcrv, np.nan
    )  # gentle gradient (<20°)

    # Apply mask (set boundary/excluded cells to NaN)
    z    = np.where(mask, z,    np.nan)
    grd  = np.where(mask, grd,  np.nan)
    crv  = np.where(mask, crv,  np.nan)
    htcrv = np.where(mask, htcrv, np.nan)

    # --- Compute statistics for each field ---
    # Elevation: cv instead of std (Table 2, footnote)
    features['Z_mean'] = np.nanmean(z)
    features['Z_cv']   = np.nanstd(z) / np.nanmean(z)   # coefficient of variation
    features['Z_med']  = np.nanmedian(z)
    features['Z_max']  = np.nanmax(z)
    features['Z_skew'] = float(st.skew(z, axis=None, nan_policy='omit'))
    features['Z_kurt'] = float(st.kurtosis(z, axis=None, nan_policy='omit'))

    # Gradient
    features['grd_mean'] = np.nanmean(grd)
    features['grd_std']  = np.nanstd(grd)
    features['grd_med']  = np.nanmedian(grd)
    features['grd_max']  = np.nanmax(grd)
    features['grd_skew'] = float(st.skew(grd, axis=None, nan_policy='omit'))
    features['grd_kurt'] = float(st.kurtosis(grd, axis=None, nan_policy='omit'))

    # Curvature (includes min; Table 2)
    features['crv_mean'] = np.nanmean(crv)
    features['crv_std']  = np.nanstd(crv)
    features['crv_min']  = np.nanmin(crv)
    features['crv_med']  = np.nanmedian(crv)
    features['crv_max']  = np.nanmax(crv)
    features['crv_skew'] = float(st.skew(crv, axis=None, nan_policy='omit'))
    features['crv_kurt'] = float(st.kurtosis(crv, axis=None, nan_policy='omit'))

    # Hilltop curvature (includes min; Table 2)
    features['htcrv_mean'] = np.nanmean(htcrv)
    features['htcrv_std']  = np.nanstd(htcrv)
    features['htcrv_min']  = np.nanmin(htcrv)
    features['htcrv_med']  = np.nanmedian(htcrv)
    features['htcrv_max']  = np.nanmax(htcrv)
    features['htcrv_skew'] = float(st.skew(htcrv, axis=None, nan_policy='omit'))
    features['htcrv_kurt'] = float(st.kurtosis(htcrv, axis=None, nan_policy='omit'))

    # Hypsometric integral (Strahler, 1952; Section 3.1)
    features['hyp_int'] = _compute_hypsometric_integral(mg, mask)

    # Landscape width Ly (m) — measurable physical feature (Section 3.1)
    features['Ly'] = (mg.number_of_node_rows - 2) * mg.dx

    return features


def _compute_hypsometric_integral(mg, mask):
    """
    Compute the hypsometric integral from the normalized elevation-area curve.

    The hypsometric integral is the area under the curve of normalized
    cumulative area (x-axis) vs normalized elevation (y-axis), where both
    are scaled to [0, 1] by their respective ranges (Strahler, 1952).

    Parameters
    ----------
    mg : RasterModelGrid
    mask : np.ndarray of bool

    Returns
    -------
    float
        Hypsometric integral in [0, 1].
    """
    elevs = mg.at_node['topographic__elevation'][mask.flatten()]
    e_min, e_max = np.min(elevs), np.max(elevs)
    normed_elev = np.sort((elevs - e_min) / (e_max - e_min))[::-1]
    return float(np.sum(normed_elev * (1.0 / len(elevs))))


# =============================================================================
# STAGE 2: NETWORK FEATURE EXTRACTION
# =============================================================================

def compute_network_features(G, mg):
    """
    Compute 11 network-based topographic features from a drainage graph.

    Features follow Table 3 in Saturay et al. (2025). Catchment-level
    metrics are aggregated to landscape level using catchment-weighted means,
    where the weight for each catchment is its number of essential nodes.

    Landscape-level features:
        n_nodes   Total number of essential nodes in the network
        Rb        Geometric mean bifurcation ratio (all orders)
        Rl        Geometric mean length ratio (all orders)
        Rb0       Bifurcation ratio for zero-order edges specifically
        Rl0       Length ratio for zero-order edges specifically

    Catchment-weighted mean features (zero-order edges):
        n0        Number of zero-order edges per catchment
        L0        Total length of zero-order edges per catchment (m)
        l0_mean   Mean length of zero-order edges (m)
        rlf0_mean Mean relief of zero-order edges (m)
        grd0_mean Mean gradient of zero-order edges (m/m)
        path_max  Maximum path length from any node to catchment outlet (m)

    Parameters
    ----------
    G : nx.DiGraph
        Drainage network graph from extract_channels_from_grid().
    mg : RasterModelGrid
        Grid with 'drainage_area' field.

    Returns
    -------
    features : dict
        Dictionary of 11 network feature values, or empty dict if network
        has no edges or components.
    """
    features = {}

    if not G or not G.edges():
        return features

    components = list(nx.weakly_connected_components(G))
    if not components:
        return features

    # --- Landscape-level features ---
    features['n_nodes'] = len(G.nodes)

    Rb, Rl, Rb0, Rl0 = _compute_bifurcation_length_ratios(G)
    features['Rb']  = Rb
    features['Rl']  = Rl
    features['Rb0'] = Rb0
    features['Rl0'] = Rl0

    # --- Catchment-level features (weighted aggregation) ---
    catchment_props = {}
    for component_nodes in components:
        sG = G.subgraph(component_nodes)
        outlet = [n for n in sG.nodes() if sG.out_degree(n) == 0][0]

        n_zero = len([n for n in sG.nodes() if sG.in_degree(n) == 0])
        zero_edges = [e for e in sG.edges() if sG.edges[e]['str_order'] == 0]

        catchment_props[outlet] = {
            'num_nodes':    len(sG.nodes),
            'n0':           n_zero,
            'L0':           sum(sG.edges[e]['length'] for e in zero_edges),
            'l0_mean':      (sum(sG.edges[e]['length'] for e in zero_edges)
                             / n_zero if n_zero > 0 else np.nan),
            'rlf0_mean':    (np.mean([sG.edges[e]['relief'] for e in zero_edges])
                             if zero_edges else np.nan),
            'grd0_mean':    (np.mean([sG.edges[e]['relief'] / sG.edges[e]['length']
                                      for e in zero_edges if sG.edges[e]['length'] > 0])
                             if zero_edges else np.nan),
            'path_max':     max(
                nx.shortest_path_length(sG, target=outlet, weight='length').values()
            ),
        }

    # Catchment-weighted means (weight = number of nodes per catchment)
    weighted_props = ['n0', 'L0', 'l0_mean', 'rlf0_mean', 'grd0_mean', 'path_max']
    total_weight = sum(p['num_nodes'] for p in catchment_props.values())

    for prop in weighted_props:
        features[prop] = sum(
            p[prop] * p['num_nodes']
            for p in catchment_props.values()
            if not np.isnan(p[prop])
        ) / total_weight

    return features


def _compute_bifurcation_length_ratios(G):
    """
    Compute bifurcation and length ratios for the full drainage network.

    For each stream order o from 0 to max_order-1:
        Bifurcation ratio: N(o) / N(o+1)  where N = number of edges of order o
        Length ratio:      L(o) / L(o+1)  where L = total length of edges of order o

    Landscape-level Rb and Rl are geometric means across all orders.
    Rb0 and Rl0 are the ratios specifically between zero-order and
    first-order edges, which are the most diagnostic features for Kh/Ks
    (Section 3.5 in paper).

    Parameters
    ----------
    G : nx.DiGraph

    Returns
    -------
    Rb, Rl, Rb0, Rl0 : float
        Bifurcation and length ratios. Returns (nan, nan, nan, nan) if
        network contains only zero-order edges.
    """
    max_order = max(G.edges[e]['str_order'] for e in G.edges)
    if max_order == 0:
        return np.nan, np.nan, np.nan, np.nan

    N, L = [], []
    for o in range(max_order + 1):
        edges_o = [e for e in G.edges if G.edges[e]['str_order'] == o]
        N.append(len(edges_o))
        L.append(sum(G.edges[e]['length'] for e in edges_o))

    Rb_per_order = [N[o] / N[o + 1] for o in range(max_order)]
    Rl_per_order = [L[o] / L[o + 1] for o in range(max_order)]

    Rb  = float(np.exp(np.mean(np.log(Rb_per_order))))
    Rl  = float(np.exp(np.mean(np.log(Rl_per_order))))
    Rb0 = float(Rb_per_order[0])
    Rl0 = float(Rl_per_order[0])

    return Rb, Rl, Rb0, Rl0


# =============================================================================
# STAGE 1 MAIN: EXTRACT AND SAVE RASNET FILES
# =============================================================================

def run_stage1_rasnet(data_dir, output_dir, job_id,
                      elev_err=ELEV_ERR, ts_index=TS_INDEX):
    """
    Stage 1: Load elevation grids, run flow routing, build drainage networks,
    and save intermediate rasnet pickle files.

    For each landscape in the job's parameter file, this function:
      1. Locates the steady-state elevation grid (.npy at ts_index)
      2. Loads it into a Landlab grid with measurement noise added
      3. Runs D8 flow routing and extracts the drainage network
      4. Saves [le_params, mg, mask, chNet, wsOutlets, wsOutletsDA] to .pkl

    Parameters
    ----------
    data_dir : Path
        Directory containing elevts-*.npy and params-*.pkl files.
    output_dir : Path
        Directory for rasnet output files.
    job_id : int
        Job identifier for filtering input files.
    elev_err : float
        Elevation noise magnitude (m). Default: ELEV_ERR (10 m).
    ts_index : int
        Time step index of elevation snapshot. Default: TS_INDEX (99).
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_file = data_dir / f'params-{job_id}.pkl'
    if not param_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_file}")

    param_df = pd.read_pickle(param_file)
    print(f"Stage 1 | Job {job_id} | {len(param_df)} landscapes | "
          f"elev_err={elev_err}m | ts_index={ts_index}")

    t_start = time.time()

    for landscape_idx in param_df.index:
        npy_file = data_dir / f'elevts-{job_id}-{landscape_idx}-{ts_index}.npy'

        if not npy_file.exists():
            print(f"  [{landscape_idx}] NOT FOUND: {npy_file.name}")
            continue

        # Elevation seed: see make_elev_seed() and DATA_PROVENANCE.md
        elev_seed = make_elev_seed(job_id, landscape_idx, ts_index)

        t0 = time.time()
        print(f"  [{landscape_idx:>3d}] {npy_file.name} ...", end='', flush=True)

        mg = load_elev_grid(npy_file, dx=DX,
                            elev_err=elev_err, elev_seed=elev_seed,
                            add_noise=True)

        mg, mask, chNet, wsOutlets, wsOutletsDA = extract_channels_from_grid(mg)

        le_params = {
            'u':             param_df.loc[landscape_idx, 'u'],
            'kh':            param_df.loc[landscape_idx, 'kh'],
            'ks':            param_df.loc[landscape_idx, 'ks'],
            'ly':            param_df.loc[landscape_idx, 'ly'],
            'elev_err':      elev_err,
            'job_id':        job_id,
            'landscape_idx': landscape_idx,
            'ts_index':      ts_index,
        }

        out_file = output_dir / f'rasnet-n{int(elev_err)}-{job_id}-{landscape_idx}-{ts_index}.pkl'
        with open(out_file, 'wb') as f:
            pickle.dump([le_params, mg, mask, chNet, wsOutlets, wsOutletsDA], f)

        print(f" {(time.time()-t0)/60:.1f} min")

    print(f"Stage 1 complete | Total: {(time.time()-t_start)/3600:.2f} hrs")


# =============================================================================
# STAGE 2 MAIN: COMPUTE AND SAVE FEATURE DATAFRAMES
# =============================================================================

def run_stage2_features(data_dir, output_dir, job_id):
    """
    Stage 2: Load rasnet files and compute the full 39-feature vector for
    each landscape. Save label-feature DataFrames as .pkl files.

    For each rasnet file, this function:
      1. Loads [le_params, mg, mask, chNet, wsOutlets, wsOutletsDA]
      2. Computes 28 raster features via compute_raster_features()
      3. Computes 11 network features via compute_network_features()
      4. Combines LE parameter labels + features into a single row

    The output DataFrame has one row per landscape with columns for:
      - LE parameter labels: u, kh, ks, ly, job_id, landscape_idx, ts_index
      - Derived labels: u_ks = u/ks, kh_ks = kh/ks  (raw ratios; log-transformed at training time)
      - 39 topographic features (Tables 2-3 in paper)

    Parameters
    ----------
    data_dir : Path
        Directory containing rasnet-*.pkl files.
    output_dir : Path
        Directory for output feature DataFrame.
    job_id : int or str
        Job identifier for filtering rasnet files, or 'all' for all files.
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate rasnet files
    if str(job_id) == 'all':
        rasnet_files = sorted(data_dir.glob('rasnet-*.pkl'))
        out_file = output_dir / 'features-all.pkl'
    else:
        rasnet_files = sorted(data_dir.glob(f'rasnet-n*-{job_id}-*-*.pkl'))
            # Note: if multiple elev_err values are used, the glob pattern will match
            # all rasnet files for a given job_id regardless of noise level, producing
            # multiple rows per landscape in the output DataFrame. In that case, add
            # --elev-err as a CLI argument and filter rasnet files accordingly.
            # For the current dataset (elev_err=10m throughout), this is not an issue.
        out_file = output_dir / f'features-{job_id}.pkl'

    if not rasnet_files:
        raise FileNotFoundError(
            f"No rasnet files found in {data_dir} for job_id={job_id}"
        )

    print(f"Stage 2 | {len(rasnet_files)} rasnet files | output: {out_file.name}")

    all_rows = []
    t_start = time.time()

    for i, rasnet_file in enumerate(rasnet_files):
        print(f"  [{i+1}/{len(rasnet_files)}] {rasnet_file.name} ...",
              end='', flush=True)
        t0 = time.time()

        with open(rasnet_file, 'rb') as f:
            le_params, mg, mask, chNet, wsOutlets, wsOutletsDA = pickle.load(f)

        # Compute features
        raster_features  = compute_raster_features(mg, mask)
        network_features = compute_network_features(chNet, mg)

        # Add derived label columns (raw ratios; log10-transformed at training time in 03_train_models.py)
        le_params['u_ks']  = le_params['u']  / le_params['ks']
        le_params['kh_ks'] = le_params['kh'] / le_params['ks']

        row = {**le_params, **raster_features, **network_features}
        all_rows.append(row)

        print(f" {(time.time()-t0):.1f} s")

    df = pd.DataFrame(all_rows)
    with open(out_file, 'wb') as f:
        pickle.dump(df, f)

    print(f"Stage 2 complete | {len(df)} rows | {len(df.columns)} columns | "
          f"saved: {out_file} | "
          f"Total: {(time.time()-t_start)/60:.1f} min")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract topographic features from synthetic landscape elevation "
            "grids for the LE-ML pipeline (Saturay et al., 2025). "
            "See DATA_PROVENANCE.md for full documentation of file naming "
            "and seed conventions."
        )
    )
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=['rasnet', 'features', 'all'],
        help=(
            "Pipeline stage to run. 'rasnet': load elevation grids, run flow "
            "routing, save intermediate rasnet files. 'features': load rasnet "
            "files, compute 39-feature vectors, save feature DataFrames. "
            "'all': run both stages sequentially."
        )
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help="Input data directory (elevation .npy files for 'rasnet' stage; "
             "rasnet .pkl files for 'features' stage)."
    )
    parser.add_argument(
        '--job-id', type=str, required=True,
        help="Job identifier for filtering input files (integer), or 'all' "
             "to process all available files."
    )
    parser.add_argument(
        '--output-dir', type=str, default='.',
        help="Output directory for rasnet or feature files (default: current)."
    )
    parser.add_argument(
        '--elev-err', type=float, default=ELEV_ERR,
        help=f"Elevation noise magnitude in meters (default: {ELEV_ERR} m)."
    )
    parser.add_argument(
        '--ts-index', type=int, default=TS_INDEX,
        help=f"Time step index of elevation snapshot (default: {TS_INDEX})."
    )

    args = parser.parse_args()

    job_id = args.job_id if args.job_id == 'all' else int(args.job_id)

    if args.stage in ('rasnet', 'all'):
        run_stage1_rasnet(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            job_id=job_id,
            elev_err=args.elev_err,
            ts_index=args.ts_index,
        )

    if args.stage in ('features', 'all'):
        run_stage2_features(
            data_dir=args.output_dir if args.stage == 'all' else args.data_dir,
            output_dir=args.output_dir,
            job_id=job_id,
        )
