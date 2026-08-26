#!/usr/bin/env python3
"""
plot_zero_order_extraction.py
-----------------------------

Generate ``fig:zero-order-extraction`` for the LE-ML manuscript: a two-row
figure contrasting the cell-to-cell flow path network with its reach-based
flow path network.

    Row 1 (synthetic)  A small lattice network, drawn first with every cell
                       node shown, then with through nodes contracted and
                       reaches coloured by modified Strahler order. Defines the
                       representation without the visual density of a real
                       landscape.

    Row 2 (real)       One watershed drawn from a ``rasnet`` file, with the
                       same two representations and the same colour scale.

The cell-to-cell flow path network is built from
``mg.at_node['flow__receiver_node']``, a complete specification of the D8
network: node ``i`` drains to
``receiver[i]``. It is NOT derived from ``chNet``; ``chNet`` supplies only the
node ID of the basin to draw. Keeping the two derivations independent is what
makes the verification step meaningful.

Verification
~~~~~~~~~~~~
``02_extract_features.py`` never instantiates the cell-to-cell flow path
network: it reads ChannelProfiler's already-segmented ``data_structure``
and keeps each branch's
endpoints (``ids[0]``, ``ids[-1]``), discarding interior cells. That is
equivalent to contracting through nodes provided ChannelProfiler segments
only at confluences. This script tests that equivalence directly by
contracting the reconstructed cell-to-cell flow path network and comparing
the result against the archived ``chNet``, reporting any disagreement in
nodes, edges, or
reach lengths.

Notes
~~~~~
``DepressionFinderAndRouter`` rewrites receivers across filled depressions, so
a small number of links may cross flats in ways that look inconsistent with
the hillshade. This is expected and is a property of the routing, not of the
figure.

Usage
~~~~~
    # Diagnostics only: through-node counts, order distribution, basins
    python plot_zero_order_extraction.py \\
        --rasnet data/rasnet/rasnet-n10-42-7-25.pkl --diagnose

    # Render the figure with automatic basin selection
    python plot_zero_order_extraction.py \\
        --rasnet data/rasnet/rasnet-n10-42-7-25.pkl \\
        --output-dir outputs/figures/ --label-tag figz

    # Pin a specific basin once a candidate has been chosen
    python plot_zero_order_extraction.py \\
        --rasnet data/rasnet/rasnet-n10-42-7-25.pkl \\
        --outlet-node 184623 --output-dir outputs/figures/
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap  # noqa: E402

# Row 1 is a fixed schematic: it always shows orders 0-3 regardless of the
# order requested for the real landscape in Row 2.
ROW1_MAX_ORDER = 3

try:
    from pipeline_utils import _git_hash
except ImportError:  # allow running outside pipeline/
    def _git_hash(short=True):
        return 'nogit'


# =============================================================================
# ORDERING AND CONTRACTION
# =============================================================================

def assign_stream_order(G):
    """
    Assign modified Strahler order to edges of a reach-based flow path network.

    Reproduces the rule in ``02_extract_features.py::_assign_stream_order``
    exactly, including the increment condition (all upstream orders equal the
    maximum, rather than at least two of them). Sources are order 0.

    Returns
    -------
    dict
        Mapping ``(u, v) -> order``.
    """
    import networkx as nx

    node_order = {}
    for component in nx.weakly_connected_components(G):
        sG = G.subgraph(component)
        for node in nx.topological_sort(sG):
            if sG.in_degree(node) == 0:
                node_order[node] = 0
            else:
                up = np.array([node_order[p] for p in sG.predecessors(node)],
                              dtype=int)
                if len(up) == 1:
                    node_order[node] = up[0]
                else:
                    mx = up.max()
                    node_order[node] = mx + 1 if np.all(up == mx) else mx
    return {(u, v): node_order[u] for u, v in G.edges()}


def contract_through_nodes(G, coords=None, dx=1.0):
    """
    Contract through nodes, converting a cell-to-cell flow path network
    into a reach-based flow path network.

    Anchors are sources (in-degree 0), junctions (in-degree >= 2), and termini
    (out-degree 0). Each reach runs from one anchor to the next along the
    unique downstream path.

    Parameters
    ----------
    G : nx.DiGraph
        Cell-level graph; every node has out-degree <= 1 under D8.
    coords : dict, optional
        ``node -> (x, y)`` in grid units, for path-integrated length.
    dx : float
        Cell size in metres.

    Returns
    -------
    Gs : nx.DiGraph
        Reach graph with ``length`` (m) on each edge.
    paths : dict
        ``(u, v) -> [node ids along the reach]``, endpoints included.
    """
    import networkx as nx

    anchors = [n for n in G.nodes()
               if G.in_degree(n) == 0 or G.in_degree(n) >= 2
               or G.out_degree(n) == 0]

    Gs = nx.DiGraph()
    Gs.add_nodes_from(anchors)
    paths = {}

    for a in anchors:
        for first in G.successors(a):
            path = [a, first]
            cur = first
            while not (G.in_degree(cur) >= 2 or G.out_degree(cur) == 0):
                nxt = next(iter(G.successors(cur)))
                path.append(nxt)
                cur = nxt
            if coords is not None:
                length = 0.0
                for p, q in zip(path[:-1], path[1:]):
                    (x0, y0), (x1, y1) = coords[p], coords[q]
                    length += dx * np.hypot(x1 - x0, y1 - y0)
            else:
                length = float(len(path) - 1) * dx
            Gs.add_edge(a, cur, length=round(length, 2))
            paths[(a, cur)] = path

    return Gs, paths


# =============================================================================
# ROW 1 -- SYNTHETIC LATTICE NETWORK
# =============================================================================

def _lattice_path(p0, p1):
    """Integer D8 path from p0 to p1: diagonal steps first, then cardinal."""
    (x, y), (x1, y1) = p0, p1
    pts = [(x, y)]
    while (x, y) != (x1, y1):
        x += np.sign(x1 - x)
        y += np.sign(y1 - y)
        pts.append((int(x), int(y)))
    return pts


def _try_synthetic(max_order, dy, spacing, jitter, rng):
    """Attempt one jittered layout; returns (Gcell, coords) or None if invalid."""
    import networkx as nx

    n_src = 2 ** max_order
    level = [((i * 2 + 1) * spacing + int(rng.integers(-jitter, jitter + 1)),
               int(rng.integers(0, jitter + 1)))
              for i in range(n_src)]
    segments = []

    for lvl in range(max_order):
        nxt = []
        for j in range(0, len(level), 2):
            a, b = level[j], level[j + 1]
            jx = (a[0] + b[0]) // 2 + int(rng.integers(-jitter, jitter + 1))
            jy = (lvl + 1) * dy + int(rng.integers(-1, 2))
            jy = max(jy, max(a[1], b[1]) + 2)   # keep flow monotonically downward
            junction = (jx, jy)
            segments.append((a, junction))
            segments.append((b, junction))
            nxt.append(junction)
        level = nxt

    outlet = (level[0][0] + int(rng.integers(-1, 2)),
              level[0][1] + int(1.8 * dy))
    segments.append((level[0], outlet))

    Gcell = nx.DiGraph()
    coords, node_id = {}, {}

    def nid(pt):
        if pt not in node_id:
            node_id[pt] = len(node_id)
            coords[node_id[pt]] = pt
        return node_id[pt]

    for p0, p1 in segments:
        pts = _lattice_path(p0, p1)
        for u, v in zip(pts[:-1], pts[1:]):
            Gcell.add_edge(nid(u), nid(v))

    # Reject layouts where jittered paths collided into unintended junctions:
    # a valid layout has exactly n_src sources and D8 single-receiver structure.
    n_sources = sum(1 for n in Gcell.nodes() if Gcell.in_degree(n) == 0)
    max_out = max(Gcell.out_degree(n) for n in Gcell.nodes())
    if n_sources != n_src or max_out > 1:
        return None
    return Gcell, coords


def build_synthetic_network(max_order=3, dy=5, spacing=3, jitter=2, seed=7):
    """
    Build an irregular lattice network reaching ``max_order``.

    Sources sit at the top, the trunk at the bottom, flow downward. Reaches
    carry interior through nodes at unit lattice spacing, standing in for
    grid cells, and diagonal steps are visibly longer than cardinal ones so
    the D8 length convention is legible without annotation.

    Junction and source positions are jittered from a seeded generator so the
    network reads as a drainage network rather than a balanced binary tree.
    Layouts in which jittered paths collide into unintended junctions are
    rejected and the next seed is tried.

    Returns
    -------
    Gcell, coords : nx.DiGraph, dict
    """
    rng = np.random.default_rng(seed)
    for _ in range(200):
        out = _try_synthetic(max_order, dy, spacing, jitter, rng)
        if out is not None:
            return out
    # Fall back to a clean, unjittered layout.
    return _try_synthetic(max_order, dy, spacing, 0, np.random.default_rng(0))


# =============================================================================
# ROW 2 -- REAL WATERSHED FROM A RASNET FILE
# =============================================================================

def load_rasnet(path):
    """Load a rasnet pickle: [le_params, mg, mask, chNet, wsOutlets, wsOutletsDA]."""
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    le_params, mg, mask, chNet, wsOutlets, wsOutletsDA = payload[:6]
    return le_params, mg, mask, chNet, wsOutlets, wsOutletsDA


def build_donor_map(mg):
    """Invert the receiver field into a donor map. One O(N) pass."""
    recv = np.asarray(mg.at_node['flow__receiver_node'])
    donors = defaultdict(list)
    for i, r in enumerate(recv):
        if r != i:
            donors[int(r)].append(int(i))
    return recv, donors


def contributing_mask(recv):
    """
    Flag cells that receive flow from at least one other cell.

    The extraction admits a cell to the network only if something drains into
    it, so divide cells -- which have no upslope contributing area and cannot
    be a link -- are excluded. This is the minimum possible admission rule
    rather than a drainage-area threshold: no area value is chosen, and every
    cell with any contributing cell is kept.

    Returns
    -------
    ndarray of int
        In-degree of every grid node under D8.
    """
    n = recv.size
    nonself = recv != np.arange(n)
    return np.bincount(recv[nonself], minlength=n)


def build_cell_graph(recv, cells, outlet, indeg, require_donor=True):
    """
    Build the cell-to-cell flow path network on a basin.

    Parameters
    ----------
    require_donor : bool
        If True (default), keep only cells with at least one contributing
        cell, matching the extraction in ``02_extract_features.py``. If False,
        include every basin cell; the resulting graph is dominated by
        divide cells, which contraction cannot merge.
    """
    import networkx as nx

    cellset = set(int(c) for c in cells)
    if require_donor:
        keep = {c for c in cellset if indeg[c] >= 1}
    else:
        keep = cellset
    outlet = int(outlet)
    # Exclude the basin outlet and any self-receiver cell (Landlab marks
    # boundary nodes and unresolved pits with recv[i] == i); an unguarded
    # self-receiver becomes a self-loop and makes the graph cyclic.
    return nx.DiGraph(
        (c, int(recv[c])) for c in keep
        if c != outlet and int(recv[c]) != c and int(recv[c]) in keep
    ), keep


def upstream_cells(outlet, donors):
    """Flood-fill upstream from ``outlet``, returning the contributing cell set."""
    cells, stack = set(), [int(outlet)]
    while stack:
        n = stack.pop()
        if n in cells:
            continue
        cells.add(n)
        stack.extend(donors.get(n, ()))
    return cells


def select_basin(chNet, mg, target_order):
    """
    Choose the smallest basin whose network attains ``target_order``.

    Among edges carrying ``target_order``, take the one with the smallest
    drainage area at its downstream endpoint; that endpoint becomes the basin
    outlet, so the basin contains the full target-order reach.

    Returns
    -------
    outlet : int
    candidates : list of (outlet_node, drainage_area_cells)
    """
    da = np.asarray(mg.at_node['drainage_area'])
    cell_area = mg.dx * mg.dy

    cands = [(int(v), float(da[v]) / cell_area)
             for u, v, d in chNet.edges(data=True)
             if d.get('str_order') == target_order]
    if not cands:
        return None, []
    cands.sort(key=lambda t: t[1])
    return cands[0][0], cands


def node_coords(mg, nodes):
    """Map network node IDs to (col, row) grid coordinates."""
    n_cols = mg.shape[1]
    return {int(n): (int(n) % n_cols, int(n) // n_cols) for n in nodes}


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_against_chnet(Greach, chNet, cells, tol=0.51):
    """
    Compare the contracted cell-to-cell flow path network against the
    archived reach-based flow path network.

    Tests the claim that keeping ChannelProfiler branch endpoints is equivalent
    to contracting through nodes. Compares node sets, edge sets, and reach
    lengths on the induced subgraph.

    Returns
    -------
    dict
        Report with ``match`` plus the specific disagreements found.
    """
    sub = chNet.subgraph([n for n in chNet.nodes() if int(n) in cells])

    ours_e = {(int(u), int(v)) for u, v in Greach.edges()}
    theirs_e = {(int(u), int(v)) for u, v in sub.edges()}

    len_mismatch = []
    for u, v in sorted(ours_e & theirs_e):
        a = Greach.edges[u, v].get('length')
        b = sub.edges[u, v].get('length')
        if a is not None and b is not None and abs(a - b) > tol:
            len_mismatch.append((u, v, a, b))

    return {
        'match': (ours_e == theirs_e) and not len_mismatch,
        'n_edges_ours': len(ours_e),
        'n_edges_chnet': len(theirs_e),
        'only_ours': sorted(ours_e - theirs_e)[:10],
        'only_chnet': sorted(theirs_e - ours_e)[:10],
        'length_mismatch': len_mismatch[:10],
        'n_length_mismatch': len(len_mismatch),
    }


def strahler_divergence_report(chNet):
    """
    Count junctions where the pipeline's ordering rule departs from Strahler.

    ``_assign_stream_order`` increments order only when ALL upstream orders
    equal the maximum. Standard Strahler increments when TWO OR MORE do. The
    rules agree at every bifurcation and can differ only where a junction has
    three or more donors, at least two of them at the maximum and at least one
    below: Strahler increments, the pipeline does not, and counts accumulate in
    the lower order.

    This matters because ``Rb`` is an OLS fit of log(N_w) on w, which assumes a
    log-linear sequence. Suppressed increments distort that sequence.

    Returns
    -------
    dict
        Junction counts by donor count, divergent-junction count, and the
        order histogram with successive bifurcation ratios.
    """
    # Edge (u, v) carries the order of its upstream endpoint u.
    node_order = {}
    for u, _, d in chNet.edges(data=True):
        if d.get('str_order') is not None:
            node_order[u] = d['str_order']

    n_multi, n_diverge = 0, 0
    examples = []
    for n in chNet.nodes():
        up = [node_order[p] for p in chNet.predecessors(n) if p in node_order]
        if len(up) < 3:
            continue
        n_multi += 1
        m = max(up)
        if up.count(m) >= 2 and not all(o == m for o in up):
            n_diverge += 1
            if len(examples) < 5:
                examples.append((int(n), sorted(up, reverse=True)))

    hist = defaultdict(int)
    for _, _, d in chNet.edges(data=True):
        hist[d.get('str_order')] += 1
    orders = sorted(k for k in hist if k is not None)
    ratios = [(w, hist[w], hist[w] / hist[w + 1])
              for w in orders[:-1] if hist.get(w + 1)]

    return {'n_multi': n_multi, 'n_diverge': n_diverge,
            'examples': examples, 'ratios': ratios,
            'hist': {int(k): v for k, v in hist.items() if k is not None}}


def _horton_fit(hist, lengths=None):
    """
    Fit Horton ratios by OLS on log-transformed counts and mean lengths.

    Rb from log(N_w) on w, Rl from log(mean L_w) on w. Returns the ratios with
    the R^2 of each fit, which measures how log-linear the sequence actually
    is -- the assumption Horton's laws make and that a mixture of watersheds
    of differing maximum order will violate.
    """
    orders = sorted(hist)
    out = {'n_orders': len(orders), 'rb': np.nan, 'rb_r2': np.nan,
           'rl': np.nan, 'rl_r2': np.nan}
    if len(orders) < 3:
        return out

    w = np.array(orders, dtype=float)

    def _fit(y):
        m, b = np.polyfit(w, y, 1)
        pred = m * w + b
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return m, r2

    m, r2 = _fit(np.log(np.array([hist[o] for o in orders], dtype=float)))
    out['rb'], out['rb_r2'] = float(np.exp(-m)), r2

    if lengths:
        means = np.array([np.mean(lengths[o]) for o in orders], dtype=float)
        if np.all(means > 0):
            m, r2 = _fit(np.log(means))
            out['rl'], out['rl_r2'] = float(np.exp(m)), r2
    return out


def per_watershed_report(chNet, min_max_order=3):
    """
    Compare Horton fits within individual watersheds against the aggregate.

    ``chNet`` spans many watersheds of differing maximum order. Small basins
    contribute only low orders, so the pooled order histogram is a mixture of
    truncated sequences and need not be log-linear even when every constituent
    basin is. This isolates that effect by refitting within each watershed.

    Watersheds are grouped by the ``outlet`` edge attribute written by
    ``_build_nx_graph``; if absent, weakly connected components are used.
    """
    import networkx as nx

    groups = defaultdict(list)
    have_outlet = False
    for u, v, d in chNet.edges(data=True):
        if d.get('outlet') is not None:
            have_outlet = True
            groups[int(d['outlet'])].append((u, v, d))
    if not have_outlet:
        for comp in nx.weakly_connected_components(chNet):
            sub = chNet.subgraph(comp)
            key = next(iter(comp))
            groups[key] = [(u, v, d) for u, v, d in sub.edges(data=True)]

    per = []
    for key, edges in groups.items():
        hist, lengths = defaultdict(int), defaultdict(list)
        for _, _, d in edges:
            o = d.get('str_order')
            if o is None:
                continue
            hist[int(o)] += 1
            if d.get('length') is not None:
                lengths[int(o)].append(float(d['length']))
        if not hist:
            continue
        mx = max(hist)
        fit = _horton_fit(hist, lengths)
        fit.update({'outlet': key, 'max_order': mx,
                    'n_edges': sum(hist.values())})
        per.append(fit)

    return per, len(groups)


def through_node_report(chNet):
    """Count through nodes surviving in the archived reach-based network."""
    through = [n for n in chNet.nodes()
          if chNet.in_degree(n) == 1 and chNet.out_degree(n) == 1]
    orders = defaultdict(int)
    for _, _, d in chNet.edges(data=True):
        orders[d.get('str_order')] += 1
    return len(through), chNet.number_of_nodes(), dict(sorted(orders.items()))


# =============================================================================
# RENDERING
# =============================================================================

# Okabe-Ito, colourblind-safe. Ordered so that pale yellow falls at high
# orders, where few reaches exist, rather than at order 0 where thin lines
# would disappear against a light hillshade.
_OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
              '#E69F00', '#56B4E9', '#000000', '#F0E442']


def _order_cmap(max_order, palette='okabe-ito'):
    """
    Build a categorical colour scale for stream order.

    Stream order is ordinal, so a categorical palette trades the progression
    cue for discriminability between adjacent orders -- worth it at the thin
    linewidths a real network requires. Use ``--order-lw-step`` to recover the
    ordinal cue through linewidth instead, as stream maps conventionally do.
    """
    n = max_order + 1
    if palette == 'okabe-ito':
        colors = [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(n)]
    else:
        base = plt.get_cmap(palette)
        if hasattr(base, 'colors') and len(base.colors) >= n:
            colors = [base.colors[i] for i in range(n)]
        else:
            colors = [base(i / max(n - 1, 1)) for i in range(n)]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, n + 0.5), n)


def draw_cell_panel(ax, Gcell, coords, style='both',
                    node_size=2.0, lw=0.6, color='0.25'):
    """Draw the cell-to-cell flow path network: one node per admitted cell,
    one edge per D8 step."""
    segs = [[coords[u], coords[v]] for u, v in Gcell.edges()]
    if style in ('lines', 'both'):
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         zorder=3))
    if style in ('nodes', 'both'):
        pts = np.array([coords[n] for n in Gcell.nodes()], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=node_size, c=color,
                   marker='o', linewidths=0, zorder=4)


def draw_node_classes(ax, G, coords, base=14.0, legend=True, fontsize=10,
                      legend_ncol=2):
    """
    Draw nodes of the cell-to-cell network, distinguished by type.

    Under D8 every node drains to exactly one neighbour, so node type is set
    entirely by how many network nodes drain in: none (source), one (through),
    two or more (junction). The outlet is the single node with no downstream
    neighbour and is checked first, since it also has one node draining in.

    Source nodes are fed only by divide cells, which the admission rule
    excludes from the network -- so a source is not a hilltop.
    """
    cls = {'source': [], 'through': [], 'junction': [], 'outlet': []}
    for n in G.nodes():
        if G.out_degree(n) == 0:
            cls['outlet'].append(n)
        elif G.in_degree(n) == 0:
            cls['source'].append(n)
        elif G.in_degree(n) >= 2:
            cls['junction'].append(n)
        else:
            cls['through'].append(n)

    # Marker shape alone carries node type -- no hue. Colour in this figure is
    # reserved for reach order in panels B and D, so a coloured node legend
    # here would compete with it.
    style = {
        'through':  dict(marker='o', s=base * 0.70, facecolors='0.35',
                         edgecolors='none', label='through node', zorder=4),
        'source':   dict(marker='^', s=base * 3.0, facecolors='white',
                         edgecolors='0.1', linewidths=1.0,
                         label='source node', zorder=6),
        'junction': dict(marker='o', s=base * 2.6, facecolors='white',
                         edgecolors='0.1', linewidths=1.1,
                         label='junction node', zorder=6),
        'outlet':   dict(marker='s', s=base * 2.8, facecolors='0.1',
                         edgecolors='0.1', linewidths=1.0,
                         label='outlet node', zorder=7),
    }

    handles = []
    for key in ('through', 'source', 'junction', 'outlet'):
        if not cls[key]:
            continue
        pts = np.array([coords[n] for n in cls[key]], dtype=float)
        h = ax.scatter(pts[:, 0], pts[:, 1], **style[key])
        handles.append(h)

    if legend and handles:
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.01), ncol=legend_ncol,
                  frameon=False, fontsize=fontsize,
                  handletextpad=0.4, columnspacing=1.4,
                  labelspacing=0.5, borderpad=0.2)
    return cls


def draw_reach_panel(ax, Greach, paths, coords, orders, max_order,
                     lw=1.4, show_nodes=True, straight=True,
                     palette='okabe-ito', lw_step=0.0):
    """
    Draw the reach-based flow path network, each reach coloured by stream order.

    With ``straight=True`` (default) a reach is drawn as a single segment
    between its endpoints, which is what the reach-based network is: a
    topological
    object whose geometry is carried in the ``length`` attribute, not in the
    drawn line. Reach chords therefore depart from the traced cell-to-cell path, and
    drawn length is not proportional to reach length -- state this in the
    caption. With ``straight=False`` each reach follows its true cell path,
    making the two panels geometrically identical.
    """
    cmap, norm = _order_cmap(max_order, palette)
    segs, cols, lws = [], [], []
    for (u, v), path in paths.items():
        o = orders.get((u, v))
        if o is None:
            continue
        w = lw + lw_step * float(o)
        if straight:
            segs.append([coords[u], coords[v]])
            cols.append(cmap(norm(o)))
            lws.append(w)
        else:
            pts = [coords[c] for c in path]
            n_seg = len(pts) - 1
            segs.extend([[pts[i], pts[i + 1]] for i in range(n_seg)])
            cols.extend([cmap(norm(o))] * n_seg)
            lws.extend([w] * n_seg)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws,
                                     zorder=3))
    if show_nodes:
        pts = np.array([coords[n] for n in Greach.nodes()], dtype=float)
        ax.scatter(pts[:, 0], pts[:, 1], s=6, facecolors='white',
                   edgecolors='0.15', linewidths=0.5, zorder=5)
    return cmap, norm


def resolve_zoom(zoom, units, extent):
    """
    Translate a --zoom window into axis limits for the Row 2 panels.

    ``extent`` is the imshow extent of the basin crop, ``(left, right, bottom,
    top)``, where ``bottom > top`` because rows increase downward.

    Parameters
    ----------
    zoom : sequence of 4 floats or None
        ``X0 X1 Y0 Y1``. In 'frac' units these are fractions of the basin
        bounding box with Y measured from the top downward, so ``0 1 0 1`` is
        the full basin and ``0.5 1 0 0.5`` is the top-right quadrant. In
        'cells' units they are absolute grid columns and rows.

    Returns
    -------
    (xlim, ylim) : tuple of pairs, ready for set_xlim / set_ylim.
    """
    left, right, bottom, top = extent
    if zoom is None:
        return (left, right), (bottom, top)

    x0, x1, y0, y1 = zoom
    if units == 'frac':
        for name, v in (('X0', x0), ('X1', x1), ('Y0', y0), ('Y1', y1)):
            if not (0.0 <= v <= 1.0):
                sys.exit(f"--zoom {name}={v} out of range for --zoom-units frac "
                         f"(expected 0-1). Use --zoom-units cells for grid "
                         f"coordinates.")
        xa = left + x0 * (right - left)
        xb = left + x1 * (right - left)
        ya = top + y0 * (bottom - top)
        yb = top + y1 * (bottom - top)
    else:
        xa, xb, ya, yb = x0, x1, y0, y1

    if xa >= xb or ya >= yb:
        sys.exit("--zoom requires X0 < X1 and Y0 < Y1 (Y measured downward).")
    return (xa, xb), (yb, ya)


def basemap_crop(mg, cells, kind='hillshade', smooth=5, pad=3):
    """
    Build the Row 2 basemap, computed on the full grid then cropped.

    ``kind``
        'hillshade'  shaded relief. Best for judging whether links follow
                     real convergent hollows, but hillshading differentiates
                     the DEM, and the archived elevation carries the full
                     +/-10 m measurement noise. At 30 m cells the noise
                     gradient (~0.33) exceeds most real hillslope gradients,
                     so an unsmoothed hillshade shows noise, not landform.
                     Median-filter first (``smooth``).
        'elevation'  elevation ramp. Noise-robust -- elevation is the
                     integral, so +/-10 m against hundreds of metres of
                     relief is invisible -- but a smooth ramp renders small
                     convergent hollows weakly.
        'logA'       log10 drainage area. Shows convergence directly, but is
                     circular as corroboration: the network is defined by
                     flow accumulation, so links follow high-A cells by
                     construction. Use as a supplement, not as evidence.

    ``smooth``
        Median filter width in cells applied before hillshading, matching
        ``MEDIAN_FILTER_SIZE`` in ``02_extract_features.py``. 0 disables it.

    Returns
    -------
    (array, extent, cmap, label)
    """
    n_rows, n_cols = mg.shape
    z = np.asarray(mg.at_node['topographic__elevation']).reshape(n_rows, n_cols)

    if kind == 'logA':
        a = np.asarray(mg.at_node['drainage_area']).reshape(n_rows, n_cols)
        field = np.log10(np.maximum(a, mg.dx * mg.dy))
        cmap, label = 'Blues', r'$\log_{10}$ drainage area'
    elif kind == 'elevation':
        field, cmap, label = z, 'gist_earth', 'elevation (m)'
    else:
        zs = z
        if smooth and smooth > 1:
            try:
                from scipy import ndimage
                zs = ndimage.median_filter(z, size=int(smooth))
            except ImportError:
                print("  note: scipy unavailable; hillshading unsmoothed DEM")
        ls = LightSource(azdeg=315, altdeg=45)
        field, cmap, label = (ls.hillshade(zs, vert_exag=1.0,
                                           dx=mg.dx, dy=mg.dy),
                              'gray', 'shaded relief')

    idx = np.array(sorted(cells))
    cs, rs = idx % n_cols, idx // n_cols
    c0, c1 = max(cs.min() - pad, 0), min(cs.max() + pad + 1, n_cols)
    r0, r1 = max(rs.min() - pad, 0), min(rs.max() + pad + 1, n_rows)
    extent = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
    return field[r0:r1, c0:c1], extent, cmap, label


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description=("Generate the zero-order extraction figure: the "
                     "cell-to-cell flow path network versus its reach-based "
                     "form, shown schematically and on a model landscape.")
    )
    p.add_argument('--rasnet', type=str, required=True,
                   help="Path to a rasnet .pkl file (data/rasnet/).")
    p.add_argument('--output-dir', type=str, default='.',
                   help="Directory for the output figure (default: current).")
    p.add_argument('--label-tag', type=str, default='figz',
                   help="Label tag for the output filename (default: figz).")
    p.add_argument('--target-order', type=int, default=3,
                   help=("Stream order the Row 2 basin must attain; the "
                         "smallest such basin is selected. Row 1 is a fixed "
                         f"schematic showing orders 0-{ROW1_MAX_ORDER}. The "
                         "colour scale spans whichever is larger, so order 0 "
                         "is the same colour in both rows. Default: 3."))
    p.add_argument('--zoom', type=float, nargs=4, default=None,
                   metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help=("Zoom window for BOTH Row 2 panels, which share axes. "
                         "Default units are fractions of the basin bounding "
                         "box with Y measured from the top down: '0.5 1 0 0.5' "
                         "is the top-right quadrant. The bounding box is "
                         "printed at run time in both units."))
    p.add_argument('--zoom-units', type=str, default='frac',
                   choices=['frac', 'cells'],
                   help=("Units for --zoom: 'frac' (default) fractions of the "
                         "basin bounding box, or 'cells' absolute grid "
                         "columns and rows."))
    p.add_argument('--outlet-node', type=int, default=None,
                   help=("Pin the Row 2 basin to this Landlab node ID, "
                         "overriding automatic selection."))
    p.add_argument('--cell-graph', type=str, default='contributing',
                   choices=['contributing', 'all'],
                   help=("Which basin cells enter the cell-to-cell network. "
                         "'contributing' (default) keeps only cells with at "
                         "least one contributing cell, matching the "
                         "extraction. 'all' keeps every basin cell, including "
                         "divide cells, which contraction cannot merge."))
    p.add_argument('--basemap', type=str, default='hillshade',
                   choices=['hillshade', 'elevation', 'logA'],
                   help=("Row 2 basemap. 'hillshade' (default) is best for "
                         "judging whether links follow real hollows, but "
                         "must be smoothed since the archived DEM carries "
                         "the full measurement noise. 'elevation' is "
                         "noise-robust but renders small hollows weakly. "
                         "'logA' shows convergence directly but is circular "
                         "as corroboration."))
    p.add_argument('--basemap-smooth', type=int, default=5,
                   help=("Median filter width in cells before hillshading, "
                         "matching MEDIAN_FILTER_SIZE in "
                         "02_extract_features.py. 0 disables. Default: 5."))
    p.add_argument('--basemap-alpha', type=float, default=0.85,
                   help="Basemap opacity (default: 0.85).")
    p.add_argument('--basemap-cbar', type=str, default='auto',
                   choices=['auto', 'on', 'off'],
                   help=("Colourbar for the panel C basemap. 'auto' "
                         "(default) shows it for elevation and logA and "
                         "hides it for hillshade, whose 0-1 shading values "
                         "carry no interpretable units."))
    p.add_argument('--basemap-cbar-label', type=str, default=None,
                   help=("Override the basemap colourbar label. Defaults to "
                         "the basemap's own label."))
    p.add_argument('--basemap-cbar-ticks', type=int, default=4,
                   help="Approximate number of basemap colourbar ticks (default: 4).")
    p.add_argument('--basemap-cbar-fontsize', type=float, default=9.0,
                   help=("Font size of the basemap colourbar tick labels; "
                         "the axis label is one point larger. Default: 7."))
    p.add_argument('--basemap-cbar-shrink', type=float, default=0.45,
                   help=("Basemap colourbar length as a fraction of panel C "
                         "(default: 0.45)."))
    p.add_argument('--cell-style', type=str, default='both',
                   choices=['lines', 'nodes', 'both'],
                   help=("How to render the cell-to-cell panels "
                         "(default: both)."))
    p.add_argument('--per-watershed', action='store_true',
                   help=("Refit Horton ratios within each watershed and "
                         "compare against the aggregate fit, testing whether "
                         "the pooled order histogram is a mixture of "
                         "truncated sequences."))
    p.add_argument('--min-max-order', type=int, default=3,
                   help=("Minimum maximum-order for a watershed to enter the "
                         "per-watershed summary (default: 3). Below 3 there "
                         "are too few points to fit."))
    p.add_argument('--diagnose', action='store_true',
                   help="Print diagnostics and exit without plotting.")
    p.add_argument('--reach-style', type=str, default='straight',
                   choices=['straight', 'path'],
                   help=("How reach-graph edges are drawn. 'straight' (default): "
                         "one segment per reach between its endpoints. 'path': "
                         "follow the traced cell-to-cell path."))
    p.add_argument('--palette', type=str, default='okabe-ito',
                   help=("Categorical colour scale for stream order: "
                         "'okabe-ito' (default, colourblind-safe) or any "
                         "matplotlib qualitative colormap, e.g. tab10, Set1, "
                         "Dark2, Paired."))
    p.add_argument('--order-lw-step', type=float, default=0.0,
                   help=("Linewidth added per stream order in the reach "
                         "panels, recovering the ordinal cue a categorical "
                         "palette drops. Try 0.4 for Row 2. Default: 0.0."))
    p.add_argument('--legend-loc', type=str, default='right',
                   choices=['right', 'left', 'top', 'bottom', 'under-b'],
                   help=("Where the shared order legend sits (default: "
                         "right). The first four anchor it to the figure. "
                         "'under-b' anchors it to subplot B and draws it "
                         "horizontally beneath, so it tracks that panel. "
                         "Orientation follows from this unless "
                         "--legend-orientation overrides it."))
    p.add_argument('--legend-orientation', type=str, default='auto',
                   choices=['auto', 'vertical', 'horizontal'],
                   help=("Legend orientation. 'auto' (default) is vertical "
                         "for left/right and horizontal for top/bottom. "
                         "Matplotlib requires these to agree, so a "
                         "conflicting choice is reported and ignored."))
    p.add_argument('--legend-shrink', type=float, default=0.40,
                   help=("Legend length as a fraction of the figure "
                         "(default: 0.40). Lower to shrink it."))
    p.add_argument('--legend-thickness', type=float, default=0.020,
                   help=("Legend thickness as a fraction of the figure "
                         "(default: 0.020)."))
    p.add_argument('--legend-pad', type=float, default=0.015,
                   help="Gap between the legend and the figure edge.")
    p.add_argument('--legend-offset', type=float, default=0.0,
                   help=("Shift the legend along its long axis, in figure "
                         "fractions. Positive is up (vertical) or right "
                         "(horizontal). Default: 0.0, centred."))
    p.add_argument('--seed', type=int, default=7,
                   help="Seed for the Row 1 synthetic layout (default: 7).")
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()

    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        sys.exit("networkx is required; activate the 'leml' environment.")

    print(f"Loading {args.rasnet}")
    le_params, mg, mask, chNet, wsOutlets, wsOutletsDA = load_rasnet(args.rasnet)

    n_through, n_nodes, order_hist = through_node_report(chNet)
    print("\n--- archived chNet ---")
    print(f"  nodes                : {n_nodes}")
    print(f"  through nodes        : {n_through}"
          f"{'   (endpoint selection == through-node contraction)' if n_through == 0 else '   *** SURVIVING ***'}")
    print(f"  edges per order      : {order_hist}")

    div = strahler_divergence_report(chNet)
    print("\n--- ordering rule check ---")
    print(f"  junctions with >=3 donors        : {div['n_multi']}")
    print(f"  where rule departs from Strahler : {div['n_diverge']}"
          f"{'   (rules agree everywhere)' if div['n_diverge'] == 0 else '   *** see note below ***'}")
    if div['examples']:
        print("  examples (node, upstream orders):")
        for n, up in div['examples']:
            print(f"    {n:>9d}  {up}")
    print("  successive bifurcation ratios N_w / N_w+1:")
    for w, n_w, r in div['ratios']:
        print(f"    w={w}: {n_w:>8d}   ratio {r:5.2f}")
    if div['n_diverge']:
        print("  NOTE: Rb is an OLS fit of log(N_w) on w and assumes a")
        print("        log-linear sequence. Suppressed increments at the")
        print("        junctions above distort it; the ratios listed give")
        print("        the magnitude.")

    if args.per_watershed:
        agg_hist, agg_len = defaultdict(int), defaultdict(list)
        for _, _, d in chNet.edges(data=True):
            o = d.get('str_order')
            if o is None:
                continue
            agg_hist[int(o)] += 1
            if d.get('length') is not None:
                agg_len[int(o)].append(float(d['length']))
        agg = _horton_fit(agg_hist, agg_len)

        per, n_groups = per_watershed_report(chNet, args.min_max_order)
        sel = [q for q in per if q['max_order'] >= args.min_max_order
               and np.isfinite(q['rb'])]

        print("\n--- Horton fits: aggregate vs per-watershed ---")
        print(f"  watersheds in chNet  : {n_groups}")
        print(f"  with max order >= {args.min_max_order}  : {len(sel)}")
        print(f"  AGGREGATE (all edges pooled, as the pipeline computes it):")
        print(f"    Rb {agg['rb']:.3f}  (R^2 {agg['rb_r2']:.3f})"
              f"    Rl {agg['rl']:.3f}  (R^2 {agg['rl_r2']:.3f})")
        if sel:
            rb = np.array([q['rb'] for q in sel])
            rb_r2 = np.array([q['rb_r2'] for q in sel])
            rl = np.array([q['rl'] for q in sel if np.isfinite(q['rl'])])
            rl_r2 = np.array([q['rl_r2'] for q in sel if np.isfinite(q['rl_r2'])])
            print(f"  PER-WATERSHED (refitted within each basin):")
            print(f"    Rb    median {np.median(rb):.3f}   "
                  f"IQR {np.percentile(rb, 25):.3f}-{np.percentile(rb, 75):.3f}")
            print(f"    Rb R^2 median {np.median(rb_r2):.3f}   "
                  f"IQR {np.percentile(rb_r2, 25):.3f}-{np.percentile(rb_r2, 75):.3f}")
            if rl.size:
                print(f"    Rl    median {np.median(rl):.3f}   "
                      f"IQR {np.percentile(rl, 25):.3f}-{np.percentile(rl, 75):.3f}")
                print(f"    Rl R^2 median {np.median(rl_r2):.3f}   "
                      f"IQR {np.percentile(rl_r2, 25):.3f}-{np.percentile(rl_r2, 75):.3f}")
            print("  by maximum order:")
            print("    max_w   n     Rb med   Rb R^2 med")
            for mx in sorted({q['max_order'] for q in sel}):
                g = [q for q in sel if q['max_order'] == mx]
                print(f"    {mx:>5d}   {len(g):>5d}   "
                      f"{np.median([q['rb'] for q in g]):>6.3f}   "
                      f"{np.median([q['rb_r2'] for q in g]):>10.3f}")
            print("  If per-watershed R^2 is high while the aggregate is low,")
            print("  the pooled histogram is a mixture of truncated sequences")
            print("  and the aggregate Rb is a fit to that mixture.")

    outlet, cands = select_basin(chNet, mg, args.target_order)
    if args.outlet_node is not None:
        outlet = args.outlet_node
    if outlet is None:
        sys.exit(f"No edges of order {args.target_order} in this network. "
                 f"Available orders: {sorted(k for k in order_hist if k is not None)}")

    print(f"\n--- basin selection (target order {args.target_order}) ---")
    print(f"  candidates           : {len(cands)}")
    for node, area in cands[:8]:
        mark = ' <-- selected' if node == outlet else ''
        print(f"    node {node:>9d}   {area:>10.0f} cells{mark}")

    recv, donors = build_donor_map(mg)
    cells = upstream_cells(outlet, donors)
    n_cols_grid = mg.shape[1]
    _idx = np.array(sorted(cells))
    _c0, _c1 = int((_idx % n_cols_grid).min()), int((_idx % n_cols_grid).max())
    _r0, _r1 = int((_idx // n_cols_grid).min()), int((_idx // n_cols_grid).max())
    print(f"  basin outlet node    : {outlet}")
    print(f"  basin cells          : {len(cells)}")
    print(f"  bounding box (cells) : cols {_c0}-{_c1}, rows {_r0}-{_r1}  "
          f"({_c1 - _c0 + 1} x {_r1 - _r0 + 1})")
    print(f"  --zoom frac 0 1 0 1 covers the full basin; "
          f"--zoom-units cells expects the ranges above.")

    indeg = contributing_mask(recv)
    Gcell, kept = build_cell_graph(recv, cells, outlet, indeg,
                                   require_donor=(args.cell_graph == 'contributing'))
    coords = node_coords(mg, kept)
    Greach, paths = contract_through_nodes(Gcell, coords=coords, dx=mg.dx)
    orders = assign_stream_order(Greach)

    n_divide = len(cells) - len(kept)
    print(f"\n--- reconstruction ({args.cell_graph}) ---")
    print(f"  basin cells          : {len(cells)}")
    print(f"  divide cells dropped : {n_divide}"
          f"   ({100 * n_divide / max(len(cells), 1):.1f}% -- no contributing cell)")
    print(f"  cell-to-cell nodes   : {Gcell.number_of_nodes()}")
    print(f"  reach-based nodes    : {Greach.number_of_nodes()}")
    print(f"  reduction            : "
          f"{100 * (1 - Greach.number_of_nodes() / max(Gcell.number_of_nodes(), 1)):.1f}%")

    rep = verify_against_chnet(Greach, chNet, kept)
    print(f"\n--- verification against chNet ---")
    print(f"  edges (reconstructed): {rep['n_edges_ours']}")
    print(f"  edges (chNet)        : {rep['n_edges_chnet']}")
    print(f"  MATCH                : {rep['match']}")
    if not rep['match']:
        if rep['only_ours']:
            print(f"  only in reconstruction (first 10): {rep['only_ours']}")
        if rep['only_chnet']:
            print(f"  only in chNet          (first 10): {rep['only_chnet']}")
        if rep['length_mismatch']:
            print(f"  length mismatches ({rep['n_length_mismatch']}), first 10:")
            for u, v, a, b in rep['length_mismatch']:
                print(f"    ({u}, {v})  ours={a}  chNet={b}")

    if args.diagnose:
        return

    # ---- Row 1: synthetic ----
    cbar_max = max(ROW1_MAX_ORDER, args.target_order)
    Gcell_s, coords_s = build_synthetic_network(max_order=ROW1_MAX_ORDER,
                                               seed=args.seed)
    Greach_s, paths_s = contract_through_nodes(Gcell_s, coords=coords_s, dx=1.0)
    orders_s = assign_stream_order(Greach_s)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))

    ax = axes[0, 0]
    draw_cell_panel(ax, Gcell_s, coords_s, style='lines', lw=0.9)
    cls_s = draw_node_classes(ax, Gcell_s, coords_s, base=16.0, legend=True)
    ax.set_title('A', fontsize=12, loc='left')

    ax = axes[0, 1]
    cmap, norm = draw_reach_panel(axes[0, 1], Greach_s, paths_s, coords_s,
                                  orders_s, cbar_max, lw=2.2,
                                  show_nodes=False,
                                  straight=(args.reach_style == 'straight'),
                                  palette=args.palette,
                                  lw_step=args.order_lw_step)
    # Same markers as panel A, so the two panels read against each other.
    # Through nodes are absent here by construction -- that is the contraction.
    draw_node_classes(ax, Greach_s, coords_s, base=16.0, legend=False)
    ax.set_title('B', fontsize=12, loc='left')

    for ax in axes[0]:
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    # ---- Row 2: real watershed ----
    hs, extent, bm_cmap, bm_label = basemap_crop(
        mg, cells, kind=args.basemap, smooth=args.basemap_smooth)
    bm_kw = dict(vmin=0, vmax=1) if args.basemap == 'hillshade' else {}

    im_c = None
    for ax in axes[1]:
        im = ax.imshow(hs, cmap=bm_cmap, extent=extent, origin='upper',
                       alpha=args.basemap_alpha, zorder=1,
                       interpolation='bilinear', **bm_kw)
        if im_c is None:
            im_c = im          # panel C's own mappable, not D's

    # A hillshade colourbar would label arbitrary 0-1 shading values, so it is
    # off by default for that basemap and on for elevation and logA.
    show_bm_cbar = (args.basemap_cbar == 'on' or
                    (args.basemap_cbar == 'auto' and
                     args.basemap in ('elevation', 'logA')))
    if show_bm_cbar:
        from matplotlib.ticker import MaxNLocator
        shr_c = args.basemap_cbar_shrink
        # Anchored to panel C in its own axes coordinates, so it does not take
        # width from the panel and C stays the same size as D.
        cax_c = axes[1, 0].inset_axes(
            [1.03, 0.5 - shr_c / 2, 0.030, shr_c],
            transform=axes[1, 0].transAxes)
        cbar_c = fig.colorbar(im_c, cax=cax_c, orientation='vertical')
        cbar_c.set_label(args.basemap_cbar_label or bm_label,
                         fontsize=args.basemap_cbar_fontsize + 1)
        cbar_c.locator = MaxNLocator(nbins=args.basemap_cbar_ticks)
        cbar_c.update_ticks()
        cbar_c.ax.tick_params(length=0,
                              labelsize=args.basemap_cbar_fontsize)
        cbar_c.outline.set_linewidth(0.5)

    draw_cell_panel(axes[1, 0], Gcell, coords, style=args.cell_style,
                    node_size=1.2, lw=0.35, color='#1f3b73')
    axes[1, 0].set_title('C', fontsize=12, loc='left')

    draw_reach_panel(axes[1, 1], Greach, paths, coords, orders,
                     cbar_max, lw=1.1, show_nodes=False,
                     straight=(args.reach_style == 'straight'),
                     palette=args.palette, lw_step=args.order_lw_step)
    axes[1, 1].set_title('D', fontsize=12, loc='left')

    xlim, ylim = resolve_zoom(args.zoom, args.zoom_units, extent)

    # The two Row 2 panels share axes: one zoom window governs both.
    axes[1, 1].sharex(axes[1, 0])
    axes[1, 1].sharey(axes[1, 0])
    for ax in axes[1]:
        ax.set_aspect('equal')
        ax.axis('off')
    axes[1, 0].set_xlim(*xlim)
    axes[1, 0].set_ylim(*ylim)

    # Scale bar, sized to a quarter of the visible width.
    x0, x1 = xlim
    y_bot, y_top = ylim
    span = x1 - x0
    bar_cells = max(int(round(span * 0.25)), 1)
    inset = 0.03 * span
    y_bar = y_bot - 0.04 * (y_bot - y_top)
    axes[1, 0].plot([x0 + inset, x0 + inset + bar_cells], [y_bar, y_bar],
                    color='k', lw=2, zorder=8, clip_on=False)
    axes[1, 0].text(x0 + inset, y_bar - 0.015 * (y_bot - y_top),
                    f'{bar_cells * mg.dx:.0f} m', fontsize=8, va='bottom',
                    clip_on=False)

    implied = 'vertical' if args.legend_loc in ('right', 'left') else 'horizontal'
    if args.legend_orientation != 'auto' and args.legend_orientation != implied:
        print(f"  note: --legend-orientation {args.legend_orientation} conflicts "
              f"with --legend-loc {args.legend_loc}; using {implied}.")

    loc, shr = args.legend_loc, args.legend_shrink
    thk, lpad, off = args.legend_thickness, args.legend_pad, args.legend_offset

    if loc == 'under-b':
        # Anchored to subplot B in ITS axes coordinates, so it follows that
        # panel wherever the layout puts it. Width and offsets are fractions
        # of B's width; the vertical offset is negative to sit beneath.
        cax = axes[0, 1].inset_axes(
            [0.5 - shr / 2 + off, -(lpad + thk) * 3.0, shr, thk * 1.6],
            transform=axes[0, 1].transAxes)
    else:
        # Anchored to the figure, not to the panel grid, so it keeps its
        # position and size regardless of how the panels are laid out. Space
        # is reserved by shrinking the grid rather than taking it from an axes.
        reserve = lpad + thk + 0.055
        if loc == 'right':
            fig.subplots_adjust(right=1.0 - reserve)
            rect = [1.0 - lpad - thk, 0.5 - shr / 2 + off, thk, shr]
        elif loc == 'left':
            fig.subplots_adjust(left=reserve)
            rect = [lpad, 0.5 - shr / 2 + off, thk, shr]
        elif loc == 'top':
            fig.subplots_adjust(top=1.0 - reserve)
            rect = [0.5 - shr / 2 + off, 1.0 - lpad - thk, shr, thk]
        else:
            fig.subplots_adjust(bottom=reserve)
            rect = [0.5 - shr / 2 + off, lpad, shr, thk]
        cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation=implied,
                        ticks=np.arange(cbar_max + 1))
    cbar.set_label('reach order', fontsize=10)
    # Labels without tick marks.
    cbar.ax.tick_params(length=0, labelsize=10)
    cbar.outline.set_linewidth(0.6)
    if implied == 'vertical':
        # Lowest order at the top, matching panel B: sources above, trunk below.
        cbar.ax.invert_yaxis()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'fig-network-extraction-{args.label_tag}-{_git_hash()}.jpg'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print(f"\nWrote {out}")
    print(f"  Row 2 basin outlet node: {outlet}  ({len(cells)} cells) "
          f"-- record in the caption for reproducibility.")


if __name__ == '__main__':
    main()
