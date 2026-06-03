"""LaPSEE Minicourse — Session 1 helper utilities.

Building blocks shared by ``s01_graphs-and-space.ipynb``:

* IEEE 30-bus loaders and graph constructors,
* DC susceptance Laplacian and a hand-rolled PTDF for §3.1.1 / §3.1.5,
* The four canonical spatial-weight constructions discussed in §3.1.4,
* A Motter & Lai (2002) load–capacity cascade simulator for §3.1.2.

Notebook seed: ``RANDOM_SEED = 2026``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# pandapower 2.14 writes back into result frames; pandas-2 CoW makes the buffer
# read-only and raises on assignment.  Force CoW off at import time so users do
# not have to remember the workaround.
pd.options.mode.copy_on_write = False

import networkx as nx
import pandapower as pp
import pandapower.networks as pn

RANDOM_SEED = 2026


# ---------------------------------------------------------------------------
# 1. Case loading and graph construction
# ---------------------------------------------------------------------------

def load_ieee30():
    """Return a solved IEEE 30-bus pandapower net (DC power flow)."""
    net = pn.case_ieee30()
    pp.rundcpp(net)
    return net


def bus_index_map(net) -> Dict[int, int]:
    """Map pandapower bus IDs to contiguous row indices 0..n-1."""
    return {b: i for i, b in enumerate(sorted(net.bus.index))}


def grid_edges(net):
    """Yield ``(bus_i, bus_j, x_pu, kind)`` for every line and transformer.

    Reactances are converted to per-unit on the system base ``net.sn_mva``.
    """
    base_mva = net.sn_mva
    for _, ln in net.line.iterrows():
        vn = net.bus.loc[int(ln.from_bus), "vn_kv"]
        z_base = vn ** 2 / base_mva
        x_pu = (ln.x_ohm_per_km * ln.length_km) / z_base
        yield int(ln.from_bus), int(ln.to_bus), float(x_pu), "line"
    for _, tr in net.trafo.iterrows():
        x_pu = (tr.vk_percent / 100.0) * (base_mva / tr.sn_mva)
        yield int(tr.hv_bus), int(tr.lv_bus), float(x_pu), "trafo"


def build_topology_graph(net) -> nx.Graph:
    """Unweighted topological graph ``G = (V, E)`` (Pagani & Aiello 2013)."""
    G = nx.Graph()
    G.add_nodes_from(sorted(net.bus.index))
    for i, j, _x, _kind in grid_edges(net):
        G.add_edge(i, j)
    return G


def build_admittance_graph(net) -> nx.Graph:
    """Edge weights are admittance magnitudes ``|y| = 1/|x_pu|`` (Wang 2011)."""
    G = nx.Graph()
    G.add_nodes_from(sorted(net.bus.index))
    for i, j, x_pu, _kind in grid_edges(net):
        if abs(x_pu) < 1e-12:
            continue
        G.add_edge(i, j, weight=1.0 / abs(x_pu))
    return G


def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """Binary adjacency matrix in ``sorted(G.nodes())`` order."""
    nodes = sorted(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes, weight=None, dtype=float)


# ---------------------------------------------------------------------------
# 2. DC power flow as a graph equation  (P = -B θ; Dörfler et al. 2018)
# ---------------------------------------------------------------------------

def susceptance_laplacian(net) -> np.ndarray:
    """B such that ``P_pu = - B θ`` in the lossless DC approximation.

    Line resistances are dropped; only ``x_pu`` enters.  Ordering: the
    sorted pandapower bus index.
    """
    bidx = bus_index_map(net)
    n = len(bidx)
    B = np.zeros((n, n))
    for i, j, x_pu, _kind in grid_edges(net):
        if abs(x_pu) < 1e-12:
            continue
        b = 1.0 / x_pu
        ii, jj = bidx[i], bidx[j]
        B[ii, jj] -= b
        B[jj, ii] -= b
        B[ii, ii] += b
        B[jj, jj] += b
    return B


def nodal_injection_pu(net) -> np.ndarray:
    """Net nodal active-power injection in per-unit (gen − load), excluding slack."""
    bidx = bus_index_map(net)
    P = np.zeros(len(bidx))
    base_mva = net.sn_mva
    for _, ld in net.load.iterrows():
        P[bidx[int(ld.bus)]] -= ld.p_mw / base_mva
    for _, g in net.gen.iterrows():
        if int(g.bus) in bidx:
            P[bidx[int(g.bus)]] += g.p_mw / base_mva
    return P


def solve_dc_angles(net) -> Tuple[np.ndarray, int]:
    """Return ``(theta_deg, slack_idx)`` by solving the reduced ``B θ = P`` system."""
    bidx = bus_index_map(net)
    n = len(bidx)
    B = susceptance_laplacian(net)
    P = nodal_injection_pu(net)
    slack = bidx[int(net.ext_grid.bus.iloc[0])]
    keep = [k for k in range(n) if k != slack]
    P[slack] = -P[keep].sum()  # slack absorbs imbalance
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], P[keep])
    return np.degrees(theta), slack


# ---------------------------------------------------------------------------
# 3. PTDF and electrical betweenness  (§3.1.2, Wang–Scaglione–Thomas 2011)
# ---------------------------------------------------------------------------

def _line_data(net) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Per-edge susceptance ``b_l`` and ``(i,j)`` in 0..n-1 indexing."""
    bidx = bus_index_map(net)
    bs, edges = [], []
    for i, j, x_pu, _kind in grid_edges(net):
        if abs(x_pu) < 1e-12:
            continue
        bs.append(1.0 / x_pu)
        edges.append((bidx[i], bidx[j]))
    return np.array(bs), edges


def ptdf_matrix(net) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """``PTDF[l, k]`` = sensitivity of line-``l`` flow to a unit injection at bus ``k``.

    By construction, with slack ``s``, the column ``PTDF[:, s]`` is zero and the
    rows sum to zero across buses.  Returns ``(PTDF (L × n), edges)``.
    """
    bidx = bus_index_map(net)
    n = len(bidx)
    B = susceptance_laplacian(net)
    slack = bidx[int(net.ext_grid.bus.iloc[0])]
    keep = [k for k in range(n) if k != slack]
    B_red_inv = np.linalg.inv(B[np.ix_(keep, keep)])

    bs, edges = _line_data(net)
    L = len(edges)
    keep_map = {k: idx for idx, k in enumerate(keep)}
    H_red = np.zeros((L, len(keep)))
    for l, (i, j) in enumerate(edges):
        b_l = bs[l]
        if i in keep_map:
            H_red[l, keep_map[i]] += b_l
        if j in keep_map:
            H_red[l, keep_map[j]] -= b_l

    PTDF = np.zeros((L, n))
    PTDF[:, keep] = H_red @ B_red_inv
    return PTDF, edges


def electrical_betweenness(net, threshold: float = 1e-3) -> pd.Series:
    """Bus-level electrical betweenness from PTDF column norms (Wang et al. 2011, simplified).

    For each bus ``k``, sum ``|PTDF[l, k]|`` across lines after thresholding.
    The intuition: a high score means the bus's injections perturb many line
    flows, i.e. the bus is an electrical "hub" in the sense of power-flow
    redistribution.
    """
    PTDF, _ = ptdf_matrix(net)
    bidx = bus_index_map(net)
    rev = {v: k for k, v in bidx.items()}
    P = np.abs(PTDF)
    P[P < threshold] = 0.0
    score = P.sum(axis=0)
    return pd.Series(score, index=[rev[i] for i in range(len(rev))],
                     name="electrical_betweenness").sort_index()


# ---------------------------------------------------------------------------
# 4. The four canonical W constructions  (§3.1.4)
# ---------------------------------------------------------------------------

def W_binary_adjacency(net) -> Tuple[np.ndarray, List[int]]:
    """Choice 1: ``W = A`` (binary adjacency)."""
    bidx = bus_index_map(net)
    n = len(bidx)
    W = np.zeros((n, n))
    for i, j, _x, _kind in grid_edges(net):
        ii, jj = bidx[i], bidx[j]
        W[ii, jj] = W[jj, ii] = 1.0
    return W, sorted(net.bus.index)


def W_electrical_distance(net, eps: float = 1e-9) -> Tuple[np.ndarray, List[int]]:
    """Choice 2: ``W_ij = 1 / Z_eff(i, j)`` for ``i ≠ j``.

    ``Z_eff`` is the effective resistance distance from the pseudo-inverse of
    the susceptance Laplacian (Klein & Randić 1993; Dörfler, Simpson-Porco &
    Bullo 2018).
    """
    B = susceptance_laplacian(net)
    Bp = np.linalg.pinv(B)
    n = B.shape[0]
    diag_b = np.diag(Bp)
    Z = diag_b[:, None] + diag_b[None, :] - 2.0 * Bp
    np.fill_diagonal(Z, 0.0)
    W = np.zeros((n, n))
    mask = Z > eps
    W[mask] = 1.0 / Z[mask]
    return W, sorted(net.bus.index)


def W_ptdf_threshold(net, q: float = 0.75) -> Tuple[np.ndarray, List[int]]:
    """Choice 3: thresholded PTDF-derived coupling.

    For each pair ``(i, j)``, define ``W_ij = max_l |PTDF[l, i] - PTDF[l, j]|``
    — the largest line-flow swing that a 1 MW transaction from ``j`` to ``i``
    induces anywhere on the system.  Keep only the top-``(1-q)`` quantile.
    """
    PTDF, _ = ptdf_matrix(net)
    L, n = PTDF.shape
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            W[i, j] = np.max(np.abs(PTDF[:, i] - PTDF[:, j]))
    pos = W[W > 0]
    if pos.size:
        thr = float(np.quantile(pos, q))
        W[W < thr] = 0.0
    return W, sorted(net.bus.index)


def W_knn_geographic(net, k: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Choice 4: ``k``-nearest neighbours in geographic ``(x, y)`` coordinates."""
    geo = net.bus_geodata
    bidx = bus_index_map(net)
    n = len(bidx)
    coords = np.zeros((n, 2))
    for b, i in bidx.items():
        coords[i] = geo.loc[b, ["x", "y"]].values
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    W = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(D[i])
        for j in order[1:k + 1]:  # skip self
            W[i, j] = 1.0
    W = np.maximum(W, W.T)  # symmetric union of mutual kNN
    return W, sorted(net.bus.index)


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Return ``W̃`` with rows summing to 1; isolated rows are left at zero."""
    rs = W.sum(axis=1)
    safe = np.where(rs == 0, 1.0, rs)
    return W / safe[:, None]


# ---------------------------------------------------------------------------
# 5. Motter & Lai (2002) cascade simulator  (§3.1.2)
# ---------------------------------------------------------------------------

def initial_loads(G: nx.Graph) -> pd.Series:
    """Node load = (unnormalised) betweenness centrality on the live graph."""
    return pd.Series(nx.betweenness_centrality(G, normalized=False),
                     name="load")


def motter_lai_cascade(
    G_full: nx.Graph,
    target_nodes: Iterable[int],
    alpha: float = 1.5,
    max_iters: int = 100,
) -> Dict[str, float]:
    """Iterative load–capacity cascade following Motter & Lai (2002).

    * Initial load ``L_i(0)`` = betweenness centrality on the pristine graph.
    * Capacity ``C_i = α · L_i(0)`` is fixed.
    * Remove ``target_nodes``; recompute loads on the surviving subgraph.
    * Any node with current load above ``C_i`` is tripped; iterate until
      no further trips.

    Returns initial removal count, cascaded count, and the final live fraction.
    """
    nodes = list(G_full.nodes())
    L0 = initial_loads(G_full)
    C = alpha * L0
    targets = set(target_nodes) & set(nodes)
    alive = set(nodes) - targets
    cascaded = 0
    for _ in range(max_iters):
        H = G_full.subgraph(alive).copy()
        if H.number_of_edges() == 0:
            break
        L = nx.betweenness_centrality(H, normalized=False)
        overload = [v for v in alive if L.get(v, 0.0) > C.get(v, 0.0) + 1e-12]
        if not overload:
            break
        alive -= set(overload)
        cascaded += len(overload)
    return {
        "removed_initial": len(targets),
        "cascaded": cascaded,
        "final_alive": len(alive),
        "final_alive_frac": len(alive) / max(1, len(nodes)),
    }


def n_minus_1_cascade_extent(G_full: nx.Graph, alpha: float = 1.5) -> pd.Series:
    """Cascade size (1 − final live fraction) for each single-node removal."""
    sizes = {}
    for v in G_full.nodes():
        r = motter_lai_cascade(G_full, [v], alpha=alpha)
        sizes[v] = 1.0 - r["final_alive_frac"]
    return pd.Series(sizes, name="cascade_extent")
