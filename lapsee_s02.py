"""LaPSEE Minicourse --- Session 2 helper utilities.

Building blocks for ``s02_spatial-prices-siting.ipynb``:

* IEEE 30 / IEEE 118 DC-OPF panel generators with realistic line ratings,
* LMP extraction and the ``LMP = lambda_energy - PTDF^T mu`` decomposition,
* The four W-constructors are imported from :mod:`lapsee_s01` (PTDF reuse too),
* RTS-GMLC loader: reads the GridMod CSVs into a pandapower net for the
  real-ISO mini-case.

All randomness uses ``RANDOM_SEED = 2026`` so the notebook is reproducible
under ``jupyter nbconvert --execute``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = False  # pandapower 2.14 needs writable frames

import networkx as nx
import pandapower as pp
import pandapower.networks as pn

# Re-export the four W constructors from Session 1 so the notebook can import
# everything from a single module.
from lapsee_s01 import (  # noqa: F401
    RANDOM_SEED,
    bus_index_map,
    grid_edges,
    build_topology_graph,
    susceptance_laplacian,
    ptdf_matrix,
    W_binary_adjacency,
    W_electrical_distance,
    W_ptdf_threshold,
    W_knn_geographic,
    row_standardize,
)


# ---------------------------------------------------------------------------
# 1. IEEE 30 / 118 loaders with realistic line ratings
# ---------------------------------------------------------------------------

def _set_realistic_line_ratings(net, mva_hv: float, mva_lv: float) -> None:
    """In-place: assign ``max_i_ka`` from MVA limits + line voltage.

    Lines whose ``from_bus`` is on the HV side (vn_kv > 100) get ``mva_hv``;
    others get ``mva_lv``.  These ratings are chosen to provoke a few binding
    constraints under stress but keep the OPF feasible.
    """
    for i, ln in net.line.iterrows():
        vn = net.bus.loc[int(ln.from_bus), "vn_kv"]
        mva_lim = mva_hv if vn > 100 else mva_lv
        net.line.loc[i, "max_i_ka"] = mva_lim / (np.sqrt(3) * vn)
    net.line["max_loading_percent"] = 100.0
    net.trafo["max_loading_percent"] = 100.0


def load_ieee30_for_opf() -> "pp.pandapowerNet":
    """Return IEEE 30 with realistic line ratings calibrated for spatial OPF.

    Default ratings: 50 MVA on the 132-kV backbone, 16 MVA on the 33-kV ring.
    These tighten the case enough to bind 1--3 lines under base load and
    produce LMP spread of order \\$15/MWh between min and max bus prices.
    """
    net = pn.case_ieee30()
    _set_realistic_line_ratings(net, mva_hv=50.0, mva_lv=16.0)
    return net


def load_ieee118_for_opf() -> "pp.pandapowerNet":
    """Return IEEE 118 with realistic line ratings for siting Lab 2b."""
    net = pn.case118()
    _set_realistic_line_ratings(net, mva_hv=180.0, mva_lv=80.0)
    return net


# ---------------------------------------------------------------------------
# 2. DC-OPF panel: solve OPF over a time series of loads
# ---------------------------------------------------------------------------

def stylized_load_profile(n_hours: int = 200, peak_factor: float = 1.02,
                          trough_factor: float = 0.65,
                          seed: int = RANDOM_SEED) -> np.ndarray:
    """Daily-cyclical load multiplier with mild noise.

    Returns an array of length ``n_hours`` with values in
    ``[trough_factor, peak_factor]``.  The pattern is a sine with a single
    daily peak and a small bit of additive Gaussian noise (std = 0.02),
    seeded for reproducibility.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    daily = 0.5 * (peak_factor + trough_factor) \
            + 0.5 * (peak_factor - trough_factor) * np.sin(2 * np.pi * (t - 6) / 24)
    return daily + rng.normal(scale=0.02, size=n_hours)


def run_dcopf_panel(net, multipliers: np.ndarray,
                    verbose: bool = False) -> pd.DataFrame:
    """Solve the DC-OPF at every ``multipliers[t]`` × base-load, return LMP panel.

    Returns DataFrame with shape ``(len(multipliers), n_bus)`` indexed by hour,
    columns are pandapower bus IDs (sorted).  Hours for which the OPF fails to
    converge appear as NaN.
    """
    base_loads = net.load["p_mw"].values.copy()
    n_bus = len(net.bus)
    bus_ids = sorted(net.bus.index)
    rows = []
    binding = []
    energy_lambdas = []
    for t, m in enumerate(multipliers):
        net.load["p_mw"] = base_loads * m
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pp.rundcopp(net)
            lmp = net.res_bus["lam_p"].reindex(bus_ids).values
            rows.append(lmp)
            binding.append(int((net.res_line["loading_percent"] > 99.9).sum())
                          + int((net.res_trafo["loading_percent"] > 99.9).sum()))
            # Approximate lambda_energy as the LMP at the slack bus.
            energy_lambdas.append(float(net.res_bus.loc[int(net.ext_grid.bus.iloc[0]),
                                                       "lam_p"]))
        except Exception as exc:
            if verbose:
                print(f"hour {t}: OPF failed ({exc})")
            rows.append(np.full(n_bus, np.nan))
            binding.append(np.nan)
            energy_lambdas.append(np.nan)
    # Restore base loads so the net is reusable.
    net.load["p_mw"] = base_loads
    panel = pd.DataFrame(rows, columns=bus_ids)
    panel.index.name = "hour"
    panel.attrs["binding_lines"] = binding
    panel.attrs["lambda_energy"] = energy_lambdas
    return panel


# ---------------------------------------------------------------------------
# 3. LMP decomposition (lambda_energy + congestion via PTDF)
# ---------------------------------------------------------------------------

def lmp_snapshot(net) -> Dict[str, np.ndarray]:
    """Empirical Schweppe decomposition from a solved DC-OPF.

    pandapower 2.14 exposes the nodal duals (``res_bus.lam_p``) but does not
    persist line-flow shadow prices ``mu_l`` in the DC-OPF result frames.
    We therefore split each ``LMP_i`` into the operational quantities that
    *are* exposed:

    * ``lambda_energy`` = LMP at the slack bus.  The PTDF column for the slack
      is zero by construction, so the congestion contribution at that bus is
      zero, and the slack-bus LMP equals the marginal cost of the system-wide
      balance constraint.
    * ``congestion[i]`` = ``LMP_i - lambda_energy`` -- the *empirical*
      congestion component required by the Schweppe (1988) decomposition
      ``LMP_i = lambda_energy - sum_l mu_l * PTDF_{l, i}``.

    Slide A.5 derives the formula; the lab uses the empirical decomposition
    rather than back-solving for ``mu_l``.
    """
    bus_ids = sorted(net.bus.index)
    lmp = net.res_bus["lam_p"].reindex(bus_ids).values
    slack_lmp = float(net.res_bus.loc[int(net.ext_grid.bus.iloc[0]), "lam_p"])
    n_bind = int((net.res_line["loading_percent"] > 99.9).sum()) \
           + int((net.res_trafo["loading_percent"] > 99.9).sum())
    return {
        "lmp": lmp,
        "lambda_energy": slack_lmp,
        "congestion": lmp - slack_lmp,
        "n_binding": n_bind,
        "spread": float(lmp.max() - lmp.min()),
    }


# ---------------------------------------------------------------------------
# 4. Suitability index for DG siting (Lab 2b)
# ---------------------------------------------------------------------------

def suitability_features(net, lmp_panel: pd.DataFrame) -> pd.DataFrame:
    """Build the regressor matrix for DG-siting analysis.

    Columns:
        * ``avg_lmp``                : mean LMP across hours (\\$/MWh)
        * ``betweenness``            : topological betweenness centrality
        * ``demand_density``         : sum of demand within 3 hops
        * ``congestion_exposure``    : fraction of hours with binding lines incident
        * ``renewable_potential``    : a deterministic function of bus coordinates
                                       proxying for solar/wind capacity factor
    """
    bus_ids = sorted(net.bus.index)
    G = build_topology_graph(net)
    bt = nx.betweenness_centrality(G, normalized=True)

    # 3-hop neighbourhood demand density
    base_load_per_bus = (
        net.load.groupby("bus")["p_mw"].sum()
        .reindex(bus_ids, fill_value=0.0)
    )
    dens = {}
    for b in bus_ids:
        ball = nx.single_source_shortest_path_length(G, b, cutoff=3)
        dens[b] = float(base_load_per_bus.reindex(list(ball.keys()), fill_value=0.0).sum())

    # Congestion exposure: fraction of hours with binding lines for which the
    # bus is on either endpoint.  Uses the panel's ``binding_lines`` attribute.
    binding_hours = lmp_panel.attrs.get("binding_lines", [])
    exposure = pd.Series(0.0, index=bus_ids)
    if binding_hours:
        n_hours = len(binding_hours)
        # Approximate exposure as (frac of hours with any binding) * (degree / max_degree)
        any_bind = np.mean([(b or 0) > 0 for b in binding_hours])
        max_deg = max(dict(G.degree()).values()) or 1
        for b in bus_ids:
            exposure[b] = any_bind * G.degree(b) / max_deg

    # Renewable potential: a smooth function of bus_geodata (x, y) coordinates.
    # Higher in the south-west (proxy for solar) — deterministic, no random noise.
    geo = net.bus_geodata.reindex(bus_ids)
    if not geo[["x", "y"]].isna().all().all():
        x = geo["x"].values
        y = geo["y"].values
        x_norm = (x - x.min()) / max(1e-6, x.max() - x.min())
        y_norm = (y - y.min()) / max(1e-6, y.max() - y.min())
        renewable = 0.5 + 0.5 * (1.0 - x_norm) * (1.0 - y_norm)
    else:
        renewable = np.full(len(bus_ids), 0.5)

    avg_lmp = lmp_panel.mean(axis=0).reindex(bus_ids).values

    return pd.DataFrame({
        "avg_lmp": avg_lmp,
        "betweenness": [bt[b] for b in bus_ids],
        "demand_density": [dens[b] for b in bus_ids],
        "congestion_exposure": exposure.values,
        "renewable_potential": renewable,
    }, index=bus_ids)


def suitability_index(features: pd.DataFrame,
                      weights: Tuple[float, float, float, float] = (0.35, 0.20, 0.20, 0.25)
                      ) -> pd.Series:
    """Linear suitability score from standardised features.

    Returns one score per bus (higher = better DG site).  Weights are over
    ``(avg_lmp, demand_density, congestion_exposure, renewable_potential)``.
    """
    w_lmp, w_dem, w_cong, w_ren = weights
    z = (features - features.mean()) / features.std(ddof=0).replace(0, 1.0)
    return (
        w_lmp * z["avg_lmp"]
        + w_dem * z["demand_density"]
        + w_cong * z["congestion_exposure"]
        + w_ren * z["renewable_potential"]
    ).rename("suitability")


# ---------------------------------------------------------------------------
# 5. RTS-GMLC loader  (real-ISO mini-case)
# ---------------------------------------------------------------------------

def load_rts_gmlc(data_dir: Path | str = "data/rts_gmlc",
                  rating_multiplier: float = 1.3) -> "pp.pandapowerNet":
    """Build a pandapower DC-only net from the RTS-GMLC source CSVs.

    Only thermal generators are kept (Coal, Oil, Gas) with linear marginal
    cost computed from ``HR_avg_0`` * ``Fuel Price``.  Renewables and storage
    are dropped to keep the DC-OPF tractable and within the lab's pedagogical
    scope.  Reactive limits are not enforced.

    Source:  https://github.com/GridMod/RTS-GMLC (MIT-licensed).
    """
    data_dir = Path(data_dir)
    bus_df = pd.read_csv(data_dir / "bus.csv")
    branch_df = pd.read_csv(data_dir / "branch.csv")
    gen_df = pd.read_csv(data_dir / "gen.csv")

    net = pp.create_empty_network(sn_mva=100.0)

    # Buses
    pp_idx = {}
    for _, b in bus_df.iterrows():
        pp_idx[int(b["Bus ID"])] = pp.create_bus(
            net, vn_kv=float(b["BaseKV"]), name=str(b["Bus Name"]),
            geodata=(float(b["lng"]), float(b["lat"])),
            zone=int(b["Zone"]) if pd.notna(b["Zone"]) else None,
        )

    # Branches
    for _, br in branch_df.iterrows():
        f, t = int(br["From Bus"]), int(br["To Bus"])
        vn_f = float(bus_df.set_index("Bus ID").loc[f, "BaseKV"])
        vn_t = float(bus_df.set_index("Bus ID").loc[t, "BaseKV"])
        # MVA rating in column "Cont Rating"; convert to kA.
        mva = max(float(br.get("Cont Rating", 100.0)), 1.0)
        if abs(vn_f - vn_t) < 0.1:
            # Line
            r_ohm = float(br["R"]) * (vn_f ** 2) / 100.0
            x_ohm = max(float(br["X"]), 1e-4) * (vn_f ** 2) / 100.0
            pp.create_line_from_parameters(
                net, from_bus=pp_idx[f], to_bus=pp_idx[t], length_km=1.0,
                r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
                c_nf_per_km=0.0, max_i_ka=mva / (np.sqrt(3) * vn_f),
                name=str(br["UID"]),
            )
        else:
            # Transformer
            vk = max(float(br["X"]), 0.001) * 100.0
            pp.create_transformer_from_parameters(
                net, hv_bus=pp_idx[f if vn_f > vn_t else t],
                lv_bus=pp_idx[t if vn_f > vn_t else f],
                sn_mva=mva, vn_hv_kv=max(vn_f, vn_t), vn_lv_kv=min(vn_f, vn_t),
                vk_percent=vk, vkr_percent=0.0, pfe_kw=0.0, i0_percent=0.0,
                name=str(br["UID"]),
            )

    # Loads (per bus from "MW Load")
    for _, b in bus_df.iterrows():
        if float(b["MW Load"]) > 0:
            pp.create_load(net, bus=pp_idx[int(b["Bus ID"])],
                          p_mw=float(b["MW Load"]),
                          controllable=False,
                          name=f"L_{int(b['Bus ID'])}")

    # Generators -- include all, with marginal cost from HR x Fuel Price for
    # thermals and 0 for non-fuel resources (Solar PV, Wind, Hydro).  All
    # generators are dispatchable in the DC-OPF; renewables simply have zero
    # cost and contribute as much as their PMax allows.
    slack_assigned = False
    for _, g in gen_df.iterrows():
        bus_pp = pp_idx[int(g["Bus ID"])]
        pmax = float(g["PMax MW"])
        pmin = max(float(g["PMin MW"]), 0.0)
        if pmax <= 0:
            continue
        fuel = str(g.get("Fuel", ""))
        try:
            hr = float(g["HR_avg_0"])
            fp = float(g["Fuel Price $/MMBTU"])
            mc = hr * fp / 1000.0
        except (ValueError, TypeError):
            hr, fp, mc = 0.0, 0.0, 0.0
        if fuel in {"Solar", "Wind", "Hydro", "Nuclear"} or hr == 0:
            mc = max(mc, 0.0)
        if not slack_assigned and mc > 0 and pmax >= 100:
            # Slack on a large thermal unit so the LMPs anchor sensibly.
            eg = pp.create_ext_grid(net, bus=bus_pp, vm_pu=1.0,
                                    min_p_mw=0.0, max_p_mw=pmax,
                                    name=str(g["GEN UID"]))
            pp.create_poly_cost(net, eg, "ext_grid", cp1_eur_per_mw=mc)
            slack_assigned = True
        else:
            gi = pp.create_gen(net, bus=bus_pp, p_mw=0.0, vm_pu=1.0,
                              max_p_mw=pmax, min_p_mw=0.0,
                              controllable=True,
                              name=str(g["GEN UID"]))
            pp.create_poly_cost(net, gi, "gen", cp1_eur_per_mw=mc)

    # Loosen line ratings: RTS-GMLC's "Cont Rating" reflects nominal
    # steady-state, not OPF-binding limits.  ``rating_multiplier`` scales
    # them; 1.3 produces a few binding lines + negative-LMP renewable
    # curtailment, matching documented real-ISO phenomena.
    net.line["max_i_ka"] *= rating_multiplier
    net.line["max_loading_percent"] = 100.0
    net.trafo["max_loading_percent"] = 100.0
    return net


def rts_gmlc_load_multipliers(data_dir: Path | str = "data/rts_gmlc",
                              n_hours: int = 168,
                              region: int = 1) -> np.ndarray:
    """Return per-hour load multipliers for ``n_hours`` from RTS-GMLC region ``region``.

    Reads ``DAY_AHEAD_regional_Load.csv``, takes the first ``n_hours`` rows
    of the requested region (1, 2, or 3), normalises by the mean to give
    multipliers around 1.0.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "DAY_AHEAD_regional_Load.csv")
    series = df[str(region)].iloc[:n_hours].values
    return series / series.mean()
