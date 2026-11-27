"""Unified telemetry and frozen-SOM data loader for dual_route visualizations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import contourpy
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DEFAULT_MODEL_PATH = Path(
    r"D:\LLM-UCB\dual_route\SOM\chemical_unit_som_v1\artifacts\model.pt"
)
DEFAULT_RUN_DIR = Path(r"D:\LLM-UCB\runs\live_large_pop_v1")


def load_som_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load the frozen 128-prototype chemical-unit SOM artifact."""
    path = Path(model_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    centers = np.asarray(ckpt["centers"], dtype=np.float64)
    topology = np.asarray(ckpt["topology"], dtype=np.float64)
    return {
        "path": str(path),
        "centers": centers,
        "topology": topology,
        "n_cells": int(centers.shape[0]),
        "dim": int(centers.shape[1]),
    }


def compute_som_layout(
    centers: np.ndarray,
    topology: np.ndarray,
    seed: int = 43,
) -> dict[str, Any]:
    """Compute deterministic 2D MDS+spring coordinates and 8x16 aligned display slots."""
    n = centers.shape[0]
    sq_dists = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    h_mat = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * (h_mat @ sq_dists @ h_mat)
    eigvals, eigvecs = np.linalg.eigh(gram)
    mds = eigvecs[:, -2:] * np.sqrt(np.clip(eigvals[-2:], 0.0, None))
    mds = mds / np.maximum(mds.std(axis=0), 1e-8)

    adj = (np.isclose(topology, 1.0)).astype(int)
    graph = nx.from_numpy_array(adj)
    init_pos = {i: mds[i] for i in range(n)}
    spring = nx.spring_layout(graph, pos=init_pos, seed=seed, iterations=180, k=0.22)
    raw_pos = np.asarray([spring[i] for i in range(n)], dtype=np.float64)

    # Normalize into clean landscape coordinates matching 3D sheet proportions
    centered = raw_pos - raw_pos.mean(axis=0, keepdims=True)
    max_abs = np.maximum(np.max(np.abs(centered), axis=0), 1e-8)
    positions = np.empty_like(centered)
    positions[:, 0] = centered[:, 0] / max_abs[0] * 0.94
    positions[:, 1] = centered[:, 1] / max_abs[1] * 0.68

    # Hungarian assignment onto 8x16 display grid for uncertainty-contraction field
    slots = np.asarray([(x + 0.5, y + 0.5) for y in range(8) for x in range(16)], dtype=np.float64)
    fit = (raw_pos - raw_pos.min(axis=0)) / np.maximum(np.ptp(raw_pos, axis=0), 1e-8)
    fit = fit * np.array([15.0, 7.0]) + 0.5
    cost = np.sum((fit[:, None, :] - slots[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    slot_positions = np.empty((n, 2), dtype=np.float64)
    slot_positions[rows] = slots[cols]

    return {
        "positions": positions,
        "raw_positions": raw_pos,
        "edges": list(graph.edges()),
        "graph": graph,
        "slots": slot_positions,
        "topology": topology,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_bundle(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load run telemetry, SOM layout, cell occupation stats, and parent->child transitions."""
    rdir = Path(run_dir)
    som_model = load_som_model(model_path)
    layout = compute_som_layout(som_model["centers"], som_model["topology"])

    bootstrap_surf = _read_json(rdir / "BOOTSTRAP_SOM_SURFACE.json")
    gen_surfs: list[dict[str, Any]] = [bootstrap_surf]
    for g in range(1, 20):
        candidate = rdir / f"G{g}_SOM_SURFACE.json"
        if candidate.exists():
            gen_surfs.append(_read_json(candidate))
    latest_surf = gen_surfs[-1]

    trajectory = _read_json(rdir / "TRAJECTORY.json")
    run_report = _read_json(rdir / "RUN_REPORT.json")
    memory_live = _read_json(rdir / "MEMORY_LIVE.json")
    audit_path = rdir / "evidence_search" / "SEARCH_AUDIT.json"
    search_audit = _read_json(audit_path) if audit_path.exists() else {}

    latest_gen = int(latest_surf["generation"])
    mapping_path = rdir / f"G{latest_gen}_MAPPING.npz"
    mapping_npz = dict(np.load(mapping_path, allow_pickle=True)) if mapping_path.exists() else {}

    boot_cells = {c["cell"] for c in bootstrap_surf["cells"] if c["occupied"] > 0}
    occ_cells = {c["cell"] for c in latest_surf["cells"] if c["occupied"] > 0}
    discovered_cells = occ_cells - boot_cells

    traj_by_cell: dict[int, list[dict[str, Any]]] = {}
    traj_by_uid: dict[str, dict[str, Any]] = {}
    for row in trajectory:
        traj_by_uid[row["uid"]] = row
        traj_by_cell.setdefault(int(row["cell"]), []).append(row)

    node_rows: list[dict[str, Any]] = []
    pos = layout["positions"]
    slots = layout["slots"]
    for idx, c in enumerate(latest_surf["cells"]):
        cid = int(c["cell"])
        b_cell = bootstrap_surf["cells"][idx]
        mols = traj_by_cell.get(cid, [])
        best_mol = max(mols, key=lambda r: float(r["score"])) if mols else None
        best_h2_mol = max(mols, key=lambda r: float(r["h2_wt_percent"])) if mols else None
        group = (
            "bootstrap"
            if cid in boot_cells
            else ("discovered" if cid in discovered_cells else "unoccupied")
        )
        node_rows.append(
            {
                "cell": cid,
                "x": float(pos[cid, 0]),
                "y": float(pos[cid, 1]),
                "slot_x": float(slots[cid, 0]),
                "slot_y": float(slots[cid, 1]),
                "occupied": int(c["occupied"]),
                "group": group,
                "combined_mean": float(c["combined_mean"]),
                "combined_sd": float(c["combined_sd"]),
                "delta_sd": float(c["combined_sd"] - b_cell["combined_sd"]),
                "whole_mean": float(c["whole_mean"]),
                "whole_sd": float(c["whole_sd"]),
                "local_mean": float(c["local_mean"]),
                "local_sd": float(c["local_sd"]),
                "best_score": float(best_mol["score"]) if best_mol else np.nan,
                "best_h2_wt": float(best_h2_mol["h2_wt_percent"]) if best_h2_mol else np.nan,
                "best_nH2": float(best_h2_mol["nH2"]) if best_h2_mol else np.nan,
                "best_operator": str(best_h2_mol["operator"]) if best_h2_mol else "",
                "best_role": str(best_mol["role"]) if best_mol else "",
                "min_evidence_gap": min(float(r["evidence_gap"]) for r in mols) if mols else 1.0,
            }
        )
    nodes_df = pd.DataFrame(node_rows)

    transitions: list[dict[str, Any]] = []
    for row in trajectory:
        puid = row.get("parent_uid")
        if puid and puid in traj_by_uid:
            parent = traj_by_uid[puid]
            if int(parent["cell"]) != int(row["cell"]):
                transitions.append(
                    {
                        "generation": int(row["generation"]),
                        "role": str(row["role"]),
                        "operator": str(row["operator"]),
                        "parent_uid": str(puid),
                        "child_uid": str(row["uid"]),
                        "src_cell": int(parent["cell"]),
                        "dst_cell": int(row["cell"]),
                        "src_score": float(parent["score"]),
                        "dst_score": float(row["score"]),
                        "src_h2_wt": float(parent["h2_wt_percent"]),
                        "dst_h2_wt": float(row["h2_wt_percent"]),
                        "evidence_gap": float(row["evidence_gap"]),
                    }
                )
    transitions_df = pd.DataFrame(transitions)

    return {
        "run_dir": str(rdir),
        "som_model": som_model,
        "layout": layout,
        "bootstrap_surface": bootstrap_surf,
        "gen_surfaces": gen_surfs,
        "latest_surface": latest_surf,
        "trajectory": trajectory,
        "run_report": run_report,
        "memory_live": memory_live,
        "search_audit": search_audit,
        "mapping_npz": mapping_npz,
        "nodes_df": nodes_df,
        "transitions_df": transitions_df,
        "bootstrap_cells": sorted(boot_cells),
        "discovered_cells": sorted(discovered_cells),
    }


def compute_differential_fields(
    bundle: dict[str, Any],
    sigma: float = 0.72,
    prior_weight: float = 2.5,
    nu: int = 241,
    nv: int = 189,
    hull_thresh: float = 0.22,
) -> dict[str, Any]:
    """Compute organic SOM hull, Kohonen SOM planar warp (XY), score lift (Z), and molecular density stains."""
    C = bundle["som_model"]["centers"].astype(np.float64)
    top = bundle["som_model"]["topology"].astype(np.float64)
    xy = bundle["layout"]["positions"]

    mu0 = np.array([float(c["combined_mean"]) for c in bundle["bootstrap_surface"]["cells"]], dtype=np.float64)
    mu_latest = np.array([float(c["combined_mean"]) for c in bundle["latest_surface"]["cells"]], dtype=np.float64)
    dmu = mu_latest - mu0

    mp = bundle["mapping_npz"]
    eh = np.asarray(mp["event_h"], dtype=np.float64)
    obs = [str(u) for u in mp["observed"]]
    by_uid = {str(r["uid"]): r for r in bundle["trajectory"]}
    gens = np.array([int(by_uid[u]["generation"]) for u in obs], dtype=int)
    cells = np.array([int(by_uid[u]["cell"]) for u in obs], dtype=int)
    scores = np.array([float(by_uid[u]["score"]) for u in obs], dtype=np.float64)
    h2_wts = np.array([float(by_uid[u]["h2_wt_percent"]) for u in obs], dtype=np.float64)

    # 1. Anchor all 144 molecules onto the SOM network plane (sharpened membership around each molecule's BMU)
    R = eh[:, 1:]
    R_sharp = R ** 8
    R_sharp = R_sharp / np.maximum(R_sharp.sum(axis=1, keepdims=True), 1e-12)
    mol_uv = 0.82 * xy[cells] + 0.18 * (R_sharp @ xy)
    Z_hat = R @ C
    g0_mask = gens == 0
    g16_mask = gens > 0

    # 2. Exact Kohonen batch-SOM structure differential in 1,092D (||dC||) and 2D network plane (dp_som)
    H0 = np.exp(-(top[:, cells[g0_mask]] ** 2) / (2.0 * sigma * sigma))
    H_all = np.exp(-(top[:, cells] ** 2) / (2.0 * sigma * sigma))
    D0 = np.full(C.shape[0], float(prior_weight), dtype=np.float64)

    C_g0 = (D0[:, None] * C + H0 @ Z_hat[g0_mask]) / (D0 + H0.sum(axis=1))[:, None]
    C_g6 = (D0[:, None] * C + H_all @ Z_hat) / (D0 + H_all.sum(axis=1))[:, None]
    dC = C_g6 - C_g0
    dC_norm = np.linalg.norm(dC, axis=1)

    p_g0 = (D0[:, None] * xy + H0 @ mol_uv[g0_mask]) / (D0 + H0.sum(axis=1))[:, None]
    p_g6 = (D0[:, None] * xy + H_all @ mol_uv) / (D0 + H_all.sum(axis=1))[:, None]
    dp_som = p_g6 - p_g0

    # 3. Organic SOM manifold hull from 128-prototype layout (no forced rectangle)
    u_lim = (float(xy[:, 0].min()) - 0.30, float(xy[:, 0].max()) + 0.30)
    v_lim = (float(xy[:, 1].min()) - 0.28, float(xy[:, 1].max()) + 0.28)
    us = np.linspace(u_lim[0], u_lim[1], nu)
    vs = np.linspace(v_lim[0], v_lim[1], nv)
    U, V = np.meshgrid(us, vs)
    uv_flat = np.column_stack([U.ravel(), V.ravel()])

    d_grid = cKDTree(xy).query(uv_flat, k=1)[0].reshape(nv, nu)
    d_smooth = gaussian_filter(d_grid, sigma=3.5)
    organic_mask = d_smooth <= hull_thresh

    cgen = contourpy.contour_generator(x=us, y=vs, z=d_smooth)
    rim_lines = cgen.lines(hull_thresh)
    rim_uv = max(rim_lines, key=lambda arr: len(arr))

    # 4. Boundary-pinned TPS interpolation of SOM structure shift (||dC||, dp_som) and score shift (dmu)
    t = np.linspace(0.0, 1.0, 21)
    boundary = np.vstack(
        [
            np.column_stack([u_lim[0] + t * (u_lim[1] - u_lim[0]), np.full_like(t, v_lim[0])]),
            np.column_stack([u_lim[0] + t * (u_lim[1] - u_lim[0]), np.full_like(t, v_lim[1])]),
            np.column_stack([np.full(19, u_lim[0]), v_lim[0] + t[1:-1] * (v_lim[1] - v_lim[0])]),
            np.column_stack([np.full(19, u_lim[1]), v_lim[0] + t[1:-1] * (v_lim[1] - v_lim[0])]),
        ]
    )
    controls = np.vstack([xy, boundary])
    n_b = len(boundary)

    def _window(arr: np.ndarray, lo: float, hi: float, margin: float) -> np.ndarray:
        dist = np.maximum(np.maximum(lo - arr, arr - hi), 0.0)
        return 0.5 + 0.5 * np.cos(np.pi * np.clip(dist / margin, 0.0, 1.0))

    inner_lo = xy.min(axis=0)
    inner_hi = xy.max(axis=0)
    taper = _window(U, inner_lo[0], inner_hi[0], 0.22) * _window(V, inner_lo[1], inner_hi[1], 0.20)

    rbf_dp = RBFInterpolator(
        controls,
        np.vstack([dp_som, np.zeros((n_b, 2), dtype=np.float64)]),
        kernel="thin_plate_spline",
        smoothing=0.04,
        degree=1,
    )
    dp_grid = rbf_dp(uv_flat).reshape(nv, nu, 2) * taper[..., None]

    rbf_norm = RBFInterpolator(
        controls,
        np.concatenate([dC_norm, np.zeros(n_b, dtype=np.float64)]),
        kernel="thin_plate_spline",
        smoothing=0.04,
        degree=1,
    )
    struct_mag_grid = np.clip(rbf_norm(uv_flat).reshape(nv, nu) * taper, 0.0, float(dC_norm.max()) * 1.05)

    dmu_mean = float(np.mean(dmu))
    rbf_dmu = RBFInterpolator(
        controls,
        np.concatenate([dmu, np.full(n_b, dmu_mean, dtype=np.float64)]),
        kernel="thin_plate_spline",
        smoothing=0.04,
        degree=1,
    )
    dmu_raw = rbf_dmu(uv_flat).reshape(nv, nu)
    dmu_grid = dmu_mean + taper * (dmu_raw - dmu_mean)
    dmu_grid = np.clip(dmu_grid, float(dmu.min()) - 0.04, float(dmu.max()) + 0.04)

    # 5. Tune Kohonen in-plane warp gain strictly until Jacobian >= 0.28 across organic manifold
    dp_u = dp_grid[..., 0]
    dp_v = dp_grid[..., 1]
    max_dp = max(float(np.max(np.abs(dp_u))), float(np.max(np.abs(dp_v))), 1e-8)
    dp_u_unit = dp_u / max_dp
    dp_v_unit = dp_v / max_dp

    gain_xy = 0.45
    for _ in range(40):
        X_cand = U + gain_xy * dp_u_unit
        Y_cand = V + gain_xy * dp_v_unit
        J = (
            np.gradient(X_cand, us, axis=1) * np.gradient(Y_cand, vs, axis=0)
            - np.gradient(X_cand, vs, axis=0) * np.gradient(Y_cand, us, axis=1)
        )
        if float(J[organic_mask].min()) >= 0.28:
            X, Y = X_cand, Y_cand
            break
        gain_xy *= 0.9
    else:
        raise RuntimeError("Unable to preserve non-folding Jacobian >= 0.28")

    # 6. Pure score-driven Z-axis height (Z_score determined strictly by dmu)
    dm_lo = float(dmu_grid[organic_mask].min())
    dm_hi = float(dmu_grid[organic_mask].max())
    Z_score = ((dmu_grid - dm_lo) / max(dm_hi - dm_lo, 1e-6)) * 0.44
    Z_struct = (struct_mag_grid / max(float(struct_mag_grid[organic_mask].max()), 1e-6)) * 0.40

    # 7. Smooth continuous molecular density stain fields (deepening surface pigment instead of dots)
    def _kde_field(pts: np.ndarray, bw: float = 0.060, gamma: float = 0.70) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros_like(U)
        dx = U[..., None] - pts[None, None, :, 0]
        dy = V[..., None] - pts[None, None, :, 1]
        dens = np.exp(-0.5 * (dx * dx + dy * dy) / (bw * bw)).sum(axis=2)
        ref = max(float(np.quantile(dens[organic_mask], 0.985)), 1e-8)
        return np.clip(dens / ref, 0.0, 1.0) ** gamma

    mol_density_all = _kde_field(mol_uv, bw=0.058, gamma=0.68)
    mol_density_new = _kde_field(mol_uv[g16_mask], bw=0.058, gamma=0.68)
    mol_density_boot = _kde_field(mol_uv[g0_mask], bw=0.058, gamma=0.72)

    interp_X = RegularGridInterpolator((vs, us), X, method="linear", bounds_error=False, fill_value=None)
    interp_Y = RegularGridInterpolator((vs, us), Y, method="linear", bounds_error=False, fill_value=None)
    interp_Z_score = RegularGridInterpolator((vs, us), Z_score, method="linear", bounds_error=False, fill_value=None)
    interp_Z_struct = RegularGridInterpolator((vs, us), Z_struct, method="linear", bounds_error=False, fill_value=None)
    interp_sm = RegularGridInterpolator((vs, us), struct_mag_grid, method="linear", bounds_error=False, fill_value=None)
    interp_dm = RegularGridInterpolator((vs, us), dmu_grid, method="linear", bounds_error=False, fill_value=None)

    pts_yx = xy[:, [1, 0]]
    x_def = interp_X(pts_yx)
    y_def = interp_Y(pts_yx)
    z_def_score = interp_Z_score(pts_yx)

    mol_yx = mol_uv[:, [1, 0]]
    mol_xyz_def = np.column_stack([interp_X(mol_yx), interp_Y(mol_yx), interp_Z_score(mol_yx)])

    rim_yx = rim_uv[:, [1, 0]]
    rim_xyz_def = np.column_stack([interp_X(rim_yx), interp_Y(rim_yx), interp_Z_score(rim_yx)])
    rim_xyz_struct = np.column_stack([rim_uv[:, 0], rim_uv[:, 1], interp_Z_struct(rim_yx)])
    rim_xyz_score = np.column_stack([rim_uv[:, 0], rim_uv[:, 1], interp_Z_score(rim_yx)])

    return {
        "us": us,
        "vs": vs,
        "U": U,
        "V": V,
        "X": X,
        "Y": Y,
        "Z_score": Z_score,
        "Z_struct": Z_struct,
        "J": J,
        "organic_mask": organic_mask,
        "rim_uv": rim_uv,
        "rim_xyz_def": rim_xyz_def,
        "rim_xyz_struct": rim_xyz_struct,
        "rim_xyz_score": rim_xyz_score,
        "struct_mag_grid": struct_mag_grid,
        "dmu_grid": dmu_grid,
        "mol_density_all": mol_density_all,
        "mol_density_new": mol_density_new,
        "mol_density_boot": mol_density_boot,
        "mol_uv": mol_uv,
        "mol_xyz_def": mol_xyz_def,
        "mol_gen": gens,
        "mol_cell": cells,
        "mol_score": scores,
        "mol_h2": h2_wts,
        "interp_X": interp_X,
        "interp_Y": interp_Y,
        "interp_Z_score": interp_Z_score,
        "interp_Z_struct": interp_Z_struct,
        "interp_sm": interp_sm,
        "interp_dm": interp_dm,
        "x_def": x_def,
        "y_def": y_def,
        "z_def_score": z_def_score,
        "dC_norm": dC_norm,
        "dp_som": dp_som,
        "dmu": dmu,
        "min_jacobian": float(J[organic_mask].min()),
        "gain_xy": float(gain_xy),
    }


def compute_directed_saddle_data(
    bundle: dict[str, Any],
    target_uid: str = "1cf453da67e9a0c6ff93edbc190e81a6cba851b5e20747b2db0a2a85ca200e9c",
    generation: int = 1,
    k_refs: int = 4,
    nu: int = 301,
    nv: int = 261,
) -> dict[str, Any]:
    """Compute a bounded-grid SOM differential saddle around one directed molecule."""
    rdir = Path(bundle.get("run_dir", DEFAULT_RUN_DIR))
    xy_som = np.asarray(bundle["layout"]["positions"], dtype=np.float64)
    eh = np.asarray(bundle["mapping_npz"]["event_h"][:, 1:], dtype=np.float64)
    traj = bundle["trajectory"]
    uid_to_idx = {row["uid"]: idx for idx, row in enumerate(traj)}
    if target_uid not in uid_to_idx:
        raise KeyError(f"Target UID not found in trajectory: {target_uid}")

    t_idx = uid_to_idx[target_uid]
    assert int(np.argmax(eh[t_idx])) == int(traj[t_idx]["cell"]), "Mapping/trajectory index mismatch"

    inp = _read_json(rdir / f"G{generation}" / "INPUTS.json")
    cands = _read_json(rdir / f"G{generation}_DESIGN_CANDIDATES.json")
    root_uid = str(cands["root"]["uid"])
    match_in = next(x for x in inp if x["uid"] == target_uid)
    mems = next(v for v in match_in["overall_memory"].values() if isinstance(v, list))
    valid_refs = [x for x in mems if x["molecule"] in uid_to_idx and x["molecule"] != target_uid]
    valid_refs.sort(key=lambda x: float(x.get("weight", 0.0)), reverse=True)
    refs = valid_refs[:k_refs]
    if len(refs) < k_refs:
        raise ValueError(f"Expected at least {k_refs} valid reference molecules, found {len(refs)}")
    ref_uids = [str(r["molecule"]) for r in refs]

    def _som_project(resp: np.ndarray) -> np.ndarray:
        w = np.asarray(resp, dtype=np.float64) ** 3
        return (w / max(float(w.sum()), 1e-12)) @ xy_som

    p_new = _som_project(eh[t_idx])
    raw_dp = np.stack([_som_project(eh[uid_to_idx[u]]) - p_new for u in ref_uids], axis=0)
    scale = float(np.linalg.norm(raw_dp, axis=1).max())
    pts = raw_dp / max(scale, 1e-12) * 1.65

    ds = np.array([float(traj[uid_to_idx[u]]["score"] - traj[t_idx]["score"]) for u in ref_uids], dtype=np.float64)
    wts = np.array([float(r["weight"]) for r in refs], dtype=np.float64)
    dr_raw = np.array([float(np.linalg.norm(eh[uid_to_idx[u]] - eh[t_idx])) for u in ref_uids], dtype=np.float64)
    dr_norm = dr_raw / max(float(np.mean(dr_raw)), 1e-12)

    H_real = sum(
        w * s * np.outer(p, p) / max(float(np.dot(p, p)), 1e-8)
        for w, s, p in zip(wts, ds, pts)
    ) / max(float(np.sum(wts)), 1e-12)
    eigvals, eigvecs = np.linalg.eigh(H_real)
    assert eigvals[0] < 0.0 < eigvals[1], f"Expected opposite saddle curvatures, got {eigvals}"

    Q = np.column_stack([eigvecs[:, 1], eigvecs[:, 0]])
    if float(pts[0] @ Q[:, 0]) < 0.0:
        Q[:, 0] *= -1.0
    neg_ref_idx = int(np.argmin(ds))
    if float(pts[neg_ref_idx] @ Q[:, 1]) > 0.0:
        Q[:, 1] *= -1.0

    uv_refs = pts @ Q
    coordinates = np.vstack([uv_refs, np.zeros((1, 2), dtype=np.float64)])
    w_tilde = wts / max(float(np.sum(wts)), 1e-12)
    alpha_y = float(np.clip(abs(eigvals[0] / eigvals[1]), 0.65, 0.90))

    us = np.linspace(-2.6, 2.6, nu)
    vs = np.linspace(-2.25, 2.25, nv)
    U, V = np.meshgrid(us, vs)

    def _support(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray | float:
        return sum(
            wt * drn * np.exp(-((u - p[0]) ** 2 + (v - p[1]) ** 2) / (2.0 * 0.85**2))
            for wt, drn, p in zip(w_tilde, dr_norm, uv_refs)
        )

    def _raw_mapping(
        u: np.ndarray | float,
        v: np.ndarray | float,
        k_val: float,
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        g = _support(u, v)
        x = u * np.exp(-k_val * g)
        y = v * np.exp(-alpha_y * k_val * g)
        envelope = np.exp(-(u * u + v * v) / 4.8)
        z_raw = 0.5 * (eigvals[1] * u * u + eigvals[0] * v * v) * envelope * (0.65 + 1.55 * g)
        return x, y, z_raw

    k_val = 0.85
    for _ in range(40):
        X_cand, Y_cand, Z_raw = _raw_mapping(U, V, k_val)
        J = (
            np.gradient(X_cand, us, axis=1) * np.gradient(Y_cand, vs, axis=0)
            - np.gradient(X_cand, vs, axis=0) * np.gradient(Y_cand, us, axis=1)
        )
        if float(J.min()) >= 0.25:
            X, Y = X_cand, Y_cand
            break
        k_val *= 0.9
    else:
        raise ValueError("Deformation folds the bounded SOM grid")

    z_gain = 0.95 / max(float(np.abs(Z_raw).max()), 1e-12)
    Z = z_gain * Z_raw

    eps = 1e-4

    def _z_at(a: float, b: float) -> float:
        return float(z_gain * _raw_mapping(a, b, k_val)[2])

    grad0 = np.array(
        [(_z_at(eps, 0.0) - _z_at(-eps, 0.0)) / (2.0 * eps), (_z_at(0.0, eps) - _z_at(0.0, -eps)) / (2.0 * eps)],
        dtype=np.float64,
    )
    hxx = (_z_at(eps, 0.0) - 2.0 * _z_at(0.0, 0.0) + _z_at(-eps, 0.0)) / (eps**2)
    hyy = (_z_at(0.0, eps) - 2.0 * _z_at(0.0, 0.0) + _z_at(0.0, -eps)) / (eps**2)
    hxy = (_z_at(eps, eps) - _z_at(eps, -eps) - _z_at(-eps, eps) + _z_at(-eps, -eps)) / (4.0 * eps**2)
    hess_eigs = np.linalg.eigvalsh([[hxx, hxy], [hxy, hyy]])
    assert abs(_z_at(0.0, 0.0)) < 1e-12
    assert float(np.linalg.norm(grad0)) < 1e-6
    assert hess_eigs[0] < 0.0 < hess_eigs[1]

    px, py, pz_raw = _raw_mapping(coordinates[:, 0], coordinates[:, 1], k_val)
    point_xyz = np.column_stack([px, py, z_gain * pz_raw])
    sign_agree = int(np.sum(np.sign(point_xyz[:k_refs, 2]) == np.sign(ds)))

    tt = np.linspace(-0.75, 0.75, 150)
    ridge_x, ridge_y, ridge_zr = _raw_mapping(tt, np.zeros_like(tt), k_val)
    valley_x, valley_y, valley_zr = _raw_mapping(np.zeros_like(tt), tt, k_val)
    saddle_curves = {
        "positive": np.column_stack([ridge_x, ridge_y, z_gain * ridge_zr]),
        "negative": np.column_stack([valley_x, valley_y, z_gain * valley_zr]),
    }

    target_row = traj[t_idx]
    ref_records = []
    for idx_r, (r_info, u_ref) in enumerate(zip(refs, ref_uids)):
        t_ref = traj[uid_to_idx[u_ref]]
        ref_records.append(
            {
                "label": f"R{idx_r + 1}",
                "is_root": bool(u_ref == root_uid),
                "uid": u_ref,
                "cell": int(t_ref["cell"]),
                "lean_smiles": str(t_ref.get("lean_smiles", "")),
                "rich_smiles": str(t_ref.get("rich_smiles", "")),
                "score": float(t_ref["score"]),
                "h2_wt_percent": float(t_ref["h2_wt_percent"]),
                "similarity": float(r_info["similarity"]),
                "weight": float(r_info["weight"]),
                "ds": float(ds[idx_r]),
                "dr_raw": float(dr_raw[idx_r]),
                "dr_norm": float(dr_norm[idx_r]),
                "u": float(uv_refs[idx_r, 0]),
                "v": float(uv_refs[idx_r, 1]),
                "x": float(point_xyz[idx_r, 0]),
                "y": float(point_xyz[idx_r, 1]),
                "z": float(point_xyz[idx_r, 2]),
            }
        )

    target_record = {
        "label": "New",
        "uid": target_uid,
        "generation": int(generation),
        "cell": int(target_row["cell"]),
        "operator": str(target_row.get("operator", "")),
        "lean_smiles": str(target_row.get("lean_smiles", "")),
        "rich_smiles": str(target_row.get("rich_smiles", "")),
        "score": float(target_row["score"]),
        "h2_wt_percent": float(target_row["h2_wt_percent"]),
        "u": 0.0,
        "v": 0.0,
        "x": float(point_xyz[-1, 0]),
        "y": float(point_xyz[-1, 1]),
        "z": float(point_xyz[-1, 2]),
    }

    return {
        "us": us,
        "vs": vs,
        "U": U,
        "V": V,
        "X": X,
        "Y": Y,
        "Z": Z,
        "J": J,
        "coordinates": coordinates,
        "point_xyz": point_xyz,
        "saddle_curves": saddle_curves,
        "H_real": H_real,
        "eigvals": eigvals,
        "hess_eigs": hess_eigs,
        "gradient_at_origin": grad0,
        "min_jacobian": float(J.min()),
        "k_compression": float(k_val),
        "alpha_y": float(alpha_y),
        "z_gain": float(z_gain),
        "scale_som_to_plane": float(scale),
        "sign_agree": sign_agree,
        "ds": ds,
        "wts": wts,
        "dr_raw": dr_raw,
        "dr_norm": dr_norm,
        "target": target_record,
        "refs": ref_records,
    }


def compute_score_gradient_data(
    bundle: dict[str, Any],
    target_uid: str = "1cf453da67e9a0c6ff93edbc190e81a6cba851b5e20747b2db0a2a85ca200e9c",
    generation: int = 1,
    k_refs: int = 4,
    sigma_s: float = 1.10,
    nu: int = 301,
    nv: int = 261,
) -> dict[str, Any]:
    """Compute bounded-grid real score surface S(u, v) and gradient field around a directed molecule."""
    base = compute_directed_saddle_data(
        bundle,
        target_uid=target_uid,
        generation=generation,
        k_refs=k_refs,
        nu=nu,
        nv=nv,
    )
    us, vs = base["us"], base["vs"]
    U, V = base["U"], base["V"]
    uv_all = np.asarray(base["coordinates"], dtype=np.float64)  # R1..Rk, New(0,0)
    ds = np.asarray(base["ds"], dtype=np.float64)
    vals = np.concatenate([ds, [0.0]])

    d2_mat = np.sum((uv_all[:, None, :] - uv_all[None, :, :]) ** 2, axis=2)
    K_mat = np.exp(-d2_mat / (2.0 * sigma_s**2))
    coeff = np.linalg.solve(K_mat, vals)
    interp_error = float(np.max(np.abs(K_mat @ coeff - vals)))
    assert interp_error < 1e-10, f"RBF score interpolation error too large: {interp_error}"

    def _eval_score_and_grad(
        u: np.ndarray | float,
        v: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        S_val = np.zeros_like(u, dtype=np.float64) if isinstance(u, np.ndarray) else 0.0
        dS_du = np.zeros_like(u, dtype=np.float64) if isinstance(u, np.ndarray) else 0.0
        dS_dv = np.zeros_like(u, dtype=np.float64) if isinstance(u, np.ndarray) else 0.0
        inv_sig2 = 1.0 / (sigma_s**2)
        for c_j, p_j in zip(coeff, uv_all):
            du = u - p_j[0]
            dv = v - p_j[1]
            e = np.exp(-0.5 * (du * du + dv * dv) * inv_sig2)
            S_val = S_val + c_j * e
            dS_du = dS_du + c_j * e * (-du * inv_sig2)
            dS_dv = dS_dv + c_j * e * (-dv * inv_sig2)
        return S_val, dS_du, dS_dv

    S_grid, dSu, dSv = _eval_score_and_grad(U, V)
    grad_mag = np.hypot(dSu, dSv)
    max_grad = max(float(grad_mag.max()), 1e-12)
    g_field = grad_mag / max_grad

    _, g0_u, g0_v = _eval_score_and_grad(0.0, 0.0)
    g0 = np.array([float(g0_u), float(g0_v)], dtype=np.float64)
    grad0_norm = float(np.linalg.norm(g0))
    r1_uv = uv_all[0]
    cos_g0_r1 = float(np.dot(g0, r1_uv) / max(grad0_norm * float(np.linalg.norm(r1_uv)), 1e-12))

    k_val = 0.75
    for _ in range(40):
        X_cand = U * np.exp(-k_val * g_field)
        Y_cand = V * np.exp(-0.85 * k_val * g_field)
        J = (
            np.gradient(X_cand, us, axis=1) * np.gradient(Y_cand, vs, axis=0)
            - np.gradient(X_cand, vs, axis=0) * np.gradient(Y_cand, us, axis=1)
        )
        if float(J.min()) >= 0.30:
            X, Y = X_cand, Y_cand
            break
        k_val *= 0.9
    else:
        raise ValueError("Score-gradient deformation folds the bounded SOM grid")

    z_gain = 0.95 / max(float(np.abs(S_grid).max()), 1e-12)
    Z = z_gain * S_grid

    def _map_uv(pts_uv: np.ndarray) -> np.ndarray:
        uu = pts_uv[:, 0]
        vv = pts_uv[:, 1]
        s_val, gu, gv = _eval_score_and_grad(uu, vv)
        gf = np.hypot(gu, gv) / max_grad
        xx = uu * np.exp(-k_val * gf)
        yy = vv * np.exp(-0.85 * k_val * gf)
        zz = z_gain * s_val
        return np.column_stack([xx, yy, zz])

    point_xyz = _map_uv(uv_all)

    def _step_grad_path(sign: float, h: float = 0.03, steps: int = 70) -> np.ndarray:
        pts = [np.array([0.0, 0.0], dtype=np.float64)]
        cur = pts[0].copy()

        def _dir(pt: np.ndarray) -> tuple[np.ndarray, float]:
            _, gu, gv = _eval_score_and_grad(float(pt[0]), float(pt[1]))
            g = np.array([float(gu), float(gv)], dtype=np.float64)
            gn = float(np.linalg.norm(g))
            if gn < 1e-8:
                return np.zeros(2, dtype=np.float64), 0.0
            return sign * g / gn, gn

        for _ in range(steps):
            k1, n1 = _dir(cur)
            if n1 < 0.12:
                break
            k2, n2 = _dir(cur + 0.5 * h * k1)
            k3, n3 = _dir(cur + 0.5 * h * k2)
            k4, n4 = _dir(cur + h * k3)
            if min(n2, n3, n4) < 0.12:
                break
            nxt = cur + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if abs(float(nxt[0])) > 2.30 or abs(float(nxt[1])) > 2.00:
                break
            pts.append(nxt)
            cur = nxt
        return np.asarray(pts, dtype=np.float64)

    uphill = _step_grad_path(+1.0)
    downhill = _step_grad_path(-1.0)
    grad_path_uv = np.vstack([downhill[::-1], uphill[1:]])
    grad_path_xyz = _map_uv(grad_path_uv)

    cg = contourpy.contour_generator(x=us, y=vs, z=S_grid)
    zero_lines = cg.lines(0.0)
    if zero_lines:
        closest_line = min(zero_lines, key=lambda seg: float(np.linalg.norm(seg, axis=1).min()))
        zero_contour_xyz = _map_uv(np.asarray(closest_line, dtype=np.float64))
    else:
        zero_contour_xyz = np.empty((0, 3), dtype=np.float64)

    ref_records = []
    for idx_r, ref in enumerate(base["refs"]):
        ref_records.append(
            {
                **ref,
                "x": float(point_xyz[idx_r, 0]),
                "y": float(point_xyz[idx_r, 1]),
                "z": float(point_xyz[idx_r, 2]),
            }
        )
    target_record = {
        **base["target"],
        "x": float(point_xyz[-1, 0]),
        "y": float(point_xyz[-1, 1]),
        "z": float(point_xyz[-1, 2]),
    }

    return {
        "us": us,
        "vs": vs,
        "U": U,
        "V": V,
        "X": X,
        "Y": Y,
        "Z": Z,
        "J": J,
        "S_grid": S_grid,
        "dSu": dSu,
        "dSv": dSv,
        "grad_mag": grad_mag,
        "coordinates": uv_all,
        "point_xyz": point_xyz,
        "grad_path_xyz": grad_path_xyz,
        "zero_contour_xyz": zero_contour_xyz,
        "g0": g0,
        "grad0_norm": grad0_norm,
        "cos_g0_r1": cos_g0_r1,
        "max_grad": max_grad,
        "min_jacobian": float(J.min()),
        "k_compression": float(k_val),
        "z_gain": float(z_gain),
        "sigma_s": float(sigma_s),
        "interp_error": interp_error,
        "ds": ds,
        "target": target_record,
        "refs": ref_records,
    }


def compute_density_cube_data(
    bundle: dict[str, Any],
    grid_n: int = 88,
    sigma_3d: float = 0.25,
    sigma_2d: float = 0.22,
) -> dict[str, Any]:
    """Compute 3D molecular probability density volume over SOM structure (XY) and 1D score PC1 (Z)."""
    from scipy.interpolate import CubicSpline

    rdir = Path(bundle.get("run_dir", DEFAULT_RUN_DIR))
    eh = np.asarray(bundle["mapping_npz"]["event_h"][:, 1:], dtype=np.float64)
    xy_som = np.asarray(bundle["layout"]["positions"], dtype=np.float64)
    traj = bundle["trajectory"]
    uid_to_idx = {row["uid"]: idx for idx, row in enumerate(traj)}

    res_by_uid: dict[str, dict[str, Any]] = {}
    for stage in ["BOOTSTRAP"] + [f"G{g}" for g in range(1, 20)]:
        res_path = rdir / stage / "RESULTS.json"
        if not res_path.exists():
            continue
        for row in _read_json(res_path):
            if "combined" in row:
                res_by_uid[str(row["uid"])] = row

    uv_list: list[np.ndarray] = []
    scores_list: list[list[float]] = []
    gens_list: list[int] = []
    cells_list: list[int] = []
    for i, row in enumerate(traj):
        w = eh[i] ** 3
        uv_soft = (w / max(float(w.sum()), 1e-12)) @ xy_som
        cell = int(row["cell"])
        uv_bmu = xy_som[cell]
        uv_list.append(0.65 * uv_bmu + 0.35 * uv_soft)

        r_full = res_by_uid.get(str(row["uid"]), {})
        comb = float(row["score"])
        ov = float(r_full["overall"]) if r_full.get("overall") is not None else comb
        loc_d = r_full.get("local") or {}
        loc = float(loc_d["score"]) if loc_d.get("score") is not None else comb
        scores_list.append([comb, ov, loc])
        gens_list.append(int(row["generation"]))
        cells_list.append(cell)

    uv = np.asarray(uv_list, dtype=np.float64)
    scores_3d = np.asarray(scores_list, dtype=np.float64)
    gens = np.asarray(gens_list, dtype=np.int64)
    cells = np.asarray(cells_list, dtype=np.int64)

    uv_center = 0.5 * (uv.min(axis=0) + uv.max(axis=0))
    uv_half = max(float(np.max(np.abs(uv - uv_center))), 1e-8)
    xy_cube = (uv - uv_center) / uv_half * 0.84

    s_mean = scores_3d.mean(axis=0)
    s_std = np.maximum(scores_3d.std(axis=0), 1e-8)
    S_norm = (scores_3d - s_mean) / s_std
    _, sv, vh = np.linalg.svd(S_norm, full_matrices=False)
    pc1 = vh[0].copy()
    if pc1.sum() < 0.0:
        pc1 = -pc1
    z_pc1 = S_norm @ pc1
    pc1_var_ratio = float((sv[0] ** 2) / max(float((sv**2).sum()), 1e-12))

    z_lo, z_hi = float(z_pc1.min()), float(z_pc1.max())
    z_cube = ((z_pc1 - z_lo) / max(z_hi - z_lo, 1e-8) * 2.0 - 1.0) * 0.84
    xyz_cube = np.column_stack([xy_cube, z_cube])

    g0_mask = gens == 0
    g0_xy = xy_cube[g0_mask].mean(axis=0)
    d_xy = np.empty(len(traj), dtype=np.float64)
    for i, row in enumerate(traj):
        puid = row.get("parent_uid")
        if puid and puid in uid_to_idx and puid != row["uid"]:
            j = uid_to_idx[puid]
            d_xy[i] = float(np.linalg.norm(xy_cube[i] - xy_cube[j]))
        else:
            d_xy[i] = float(np.linalg.norm(xy_cube[i] - g0_xy))
    d_xy_ref = max(float(np.quantile(d_xy, 0.95)), 1e-8)
    m_xy = np.clip(0.55 * (d_xy / d_xy_ref) + 0.45 * (gens / 6.0), 0.0, 1.0)

    g0_z_mean = float(z_pc1[g0_mask].mean())
    g6_z_mean = float(z_pc1[gens == 6].mean())
    z_p95 = float(np.quantile(z_pc1, 0.95))
    m_z = np.clip((z_pc1 - g0_z_mean) / max(z_p95 - g0_z_mean, 1e-6), 0.0, 1.0)

    gen_vals = np.arange(0, 7, dtype=np.float64)
    centroids_xyz = np.stack([xyz_cube[gens == g].mean(axis=0) for g in range(7)], axis=0)
    t_dense = np.linspace(0.0, 6.0, 120)
    cs_x = CubicSpline(gen_vals, centroids_xyz[:, 0])
    cs_y = CubicSpline(gen_vals, centroids_xyz[:, 1])
    cs_z = CubicSpline(gen_vals, centroids_xyz[:, 2])
    centroid_spline_xyz = np.column_stack([cs_x(t_dense), cs_y(t_dense), cs_z(t_dense)])
    centroid_floor_xyz = np.column_stack([centroid_spline_xyz[:, 0], centroid_spline_xyz[:, 1], np.full(len(t_dense), -1.0)])

    # 2D face marginal density fields on [-1, 1]^2
    n_face = 65
    fs = np.linspace(-1.0, 1.0, n_face)
    F1, F2 = np.meshgrid(fs, fs)
    inv_2s2 = 1.0 / (2.0 * sigma_2d * sigma_2d)

    rho_xy = np.zeros_like(F1)
    w_xy = np.zeros_like(F1)
    for (px, py), mv in zip(xy_cube, m_xy):
        kern = np.exp(-((F1 - px) ** 2 + (F2 - py) ** 2) * inv_2s2)
        rho_xy += kern
        w_xy += kern * mv
    move_xy_floor = w_xy / np.maximum(rho_xy, 1e-12)

    rho_xz = np.zeros_like(F1)
    w_xz = np.zeros_like(F1)
    for px, pz, mz in zip(xy_cube[:, 0], z_cube, m_z):
        kern = np.exp(-((F1 - px) ** 2 + (F2 - pz) ** 2) * inv_2s2)
        rho_xz += kern
        w_xz += kern * mz
    move_z_wall = w_xz / np.maximum(rho_xz, 1e-12)

    zs_1d = np.linspace(-0.96, 0.96, 120)
    sigma_z = 0.20
    rho_z_gen = np.zeros((7, len(zs_1d)), dtype=np.float64)
    for g in range(7):
        vals_g = z_cube[gens == g]
        if len(vals_g):
            dens_g = np.exp(-0.5 * ((zs_1d[:, None] - vals_g[None, :]) / sigma_z) ** 2).mean(axis=1)
            rho_z_gen[g] = dens_g / max(float(dens_g.max()), 1e-12)

    # 3D voxel KDE over [-1, 1]^3
    vs = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)
    Xg, Yg, Zg = np.meshgrid(vs, vs, vs, indexing="ij")
    rho_3d = np.zeros((grid_n, grid_n, grid_n), dtype=np.float32)
    wxy_3d = np.zeros_like(rho_3d)
    wz_3d = np.zeros_like(rho_3d)
    inv_3d = np.float32(1.0 / (2.0 * sigma_3d * sigma_3d))
    for i in range(len(xyz_cube)):
        px, py, pz = np.float32(xyz_cube[i, 0]), np.float32(xyz_cube[i, 1]), np.float32(xyz_cube[i, 2])
        dx = Xg - px
        dy = Yg - py
        dz = Zg - pz
        f = np.exp(-(dx * dx + dy * dy + dz * dz) * inv_3d)
        rho_3d += f
        wxy_3d += f * np.float32(m_xy[i])
        wz_3d += f * np.float32(m_z[i])
    m_xy_3d = wxy_3d / np.maximum(rho_3d, np.float32(1e-12))
    m_z_3d = wz_3d / np.maximum(rho_3d, np.float32(1e-12))

    mol_table = pd.DataFrame(
        {
            "uid": [str(r["uid"]) for r in traj],
            "generation": gens,
            "cell": cells,
            "score_combined": scores_3d[:, 0],
            "score_overall": scores_3d[:, 1],
            "score_local": scores_3d[:, 2],
            "z_pc1": z_pc1,
            "x_cube": xy_cube[:, 0],
            "y_cube": xy_cube[:, 1],
            "z_cube": z_cube,
            "m_xy": m_xy,
            "m_z": m_z,
        }
    )

    return {
        "grid_coords": vs,
        "face_coords": fs,
        "F1": F1,
        "F2": F2,
        "xy_cube": xy_cube,
        "z_cube": z_cube,
        "xyz_cube": xyz_cube,
        "gens": gens,
        "cells": cells,
        "m_xy": m_xy,
        "m_z": m_z,
        "z_pc1": z_pc1,
        "pc1_var_ratio": pc1_var_ratio,
        "pc1_loadings": pc1,
        "g0_z_mean": g0_z_mean,
        "g6_z_mean": g6_z_mean,
        "occupied_g0": int(len(set(cells[g0_mask]))),
        "occupied_g6": int(len(set(cells))),
        "centroids_xyz": centroids_xyz,
        "centroid_spline_xyz": centroid_spline_xyz,
        "centroid_floor_xyz": centroid_floor_xyz,
        "rho_xy": rho_xy,
        "move_xy_floor": move_xy_floor,
        "rho_xz": rho_xz,
        "move_z_wall": move_z_wall,
        "zs_1d": zs_1d,
        "rho_z_gen": rho_z_gen,
        "rho_3d": rho_3d,
        "m_xy_3d": m_xy_3d,
        "m_z_3d": m_z_3d,
        "mol_table": mol_table,
    }


def compute_som_migration_relief_data(
    bundle: dict[str, Any],
    bw: float = 0.46,
    nu: int = 261,
    nv: int = 221,
) -> dict[str, Any]:
    """Compute SOM population migration (red=in, purple=out) and upward-only Z>=0 high-score probability."""
    eh = np.asarray(bundle["mapping_npz"]["event_h"][:, 1:], dtype=np.float64)
    xy_som = np.asarray(bundle["layout"]["positions"], dtype=np.float64)
    traj = bundle["trajectory"]

    uv_list: list[np.ndarray] = []
    gens_list: list[int] = []
    scores_list: list[float] = []
    for i, row in enumerate(traj):
        w = eh[i] ** 3
        uv_soft = (w / max(float(w.sum()), 1e-12)) @ xy_som
        uv_bmu = xy_som[int(row["cell"])]
        uv_list.append(0.65 * uv_bmu + 0.35 * uv_soft)
        gens_list.append(int(row["generation"]))
        scores_list.append(float(row["score"]))

    uv = np.asarray(uv_list, dtype=np.float64)
    gens = np.asarray(gens_list, dtype=np.int64)
    scores = np.asarray(scores_list, dtype=np.float64)

    c_uv = 0.5 * (uv.min(axis=0) + uv.max(axis=0))
    span_uv = np.maximum(uv.max(axis=0) - uv.min(axis=0), 1e-8)
    mol_pts = np.column_stack(
        [
            (uv[:, 0] - c_uv[0]) / span_uv[0] * 3.6,
            (uv[:, 1] - c_uv[1]) / span_uv[1] * 3.0,
        ]
    )

    us = np.linspace(-2.6, 2.6, nu)
    vs = np.linspace(-2.25, 2.25, nv)
    U, V = np.meshgrid(us, vs)

    def _kde(pts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        wts = np.ones(len(pts), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        denom = max(float(len(pts)), 1.0) if weights is None else max(float(len(pts)), 1.0)
        out = np.zeros_like(U, dtype=np.float64)
        inv = 1.0 / (2.0 * bw * bw)
        for (px, py), wt in zip(pts, wts):
            out += wt * np.exp(-((U - px) ** 2 + (V - py) ** 2) * inv)
        return out / denom

    m_early = gens <= 2
    m_late = gens >= 3
    rho_early = _kde(mol_pts[m_early])
    rho_late = _kde(mol_pts[m_late])

    taper = np.exp(-((U / 2.35) ** 6 + (V / 2.02) ** 6))
    drho = (rho_late - rho_early) * taper

    phi = gaussian_filter(drho, sigma=10.0) * taper
    dphi_du = np.gradient(phi, us, axis=1)
    dphi_dv = np.gradient(phi, vs, axis=0)
    gmax = max(float(np.abs(dphi_du).max()), float(np.abs(dphi_dv).max()), 1e-8)
    du_unit = dphi_du / gmax
    dv_unit = dphi_dv / gmax

    gamma = 0.62
    for _ in range(40):
        X_cand = U + gamma * du_unit * taper
        Y_cand = V + gamma * dv_unit * taper
        J = (
            np.gradient(X_cand, us, axis=1) * np.gradient(Y_cand, vs, axis=0)
            - np.gradient(X_cand, vs, axis=0) * np.gradient(Y_cand, us, axis=1)
        )
        if float(J.min()) >= 0.35:
            X, Y = X_cand, Y_cand
            break
        gamma *= 0.9
    else:
        raise ValueError("Migration mesh warp folds the SOM plane")

    j_imm = float(J[drho > 0.3 * float(drho.max())].mean())
    j_emi = float(J[drho < 0.3 * float(drho.min())].mean())
    assert j_imm < j_emi, f"Expected contraction in immigration (J={j_imm:.3f}) vs emigration (J={j_emi:.3f})"

    p_mol = 1.0 / (1.0 + np.exp(-(scores - 56.8) / 1.2))
    rho_all = _kde(mol_pts)
    rho_hi = _kde(mol_pts, weights=p_mol)
    prior_c = 0.22 * float(rho_all.max())
    P_high = (rho_hi / (rho_all + prior_c)) * taper
    Z = 1.48 * P_high
    assert float(Z.min()) >= 0.0 and float(P_high.max()) <= 1.0

    active = rho_all > 0.08 * float(rho_all.max())
    corr_drho_p = float(np.corrcoef(drho[active], P_high[active])[0, 1])

    def _landmark(idx_2d: tuple[int, int], name: str) -> dict[str, float | str]:
        r_i, c_i = int(idx_2d[0]), int(idx_2d[1])
        return {
            "name": name,
            "u": float(U[r_i, c_i]),
            "v": float(V[r_i, c_i]),
            "x": float(X[r_i, c_i]),
            "y": float(Y[r_i, c_i]),
            "z": float(Z[r_i, c_i]),
            "drho": float(drho[r_i, c_i]),
            "p_high": float(P_high[r_i, c_i]),
            "jacobian": float(J[r_i, c_i]),
        }

    imm_peak = _landmark(np.unravel_index(int(np.argmax(drho)), drho.shape), "immigration_peak")
    emi_peak = _landmark(np.unravel_index(int(np.argmin(drho)), drho.shape), "emigration_basin")
    top_z_peak = _landmark(np.unravel_index(int(np.argmax(P_high)), P_high.shape), "top_score_prob")

    interp_X = RegularGridInterpolator((vs, us), X, method="linear", bounds_error=False, fill_value=None)
    interp_Y = RegularGridInterpolator((vs, us), Y, method="linear", bounds_error=False, fill_value=None)
    interp_Z = RegularGridInterpolator((vs, us), Z, method="linear", bounds_error=False, fill_value=None)

    cg = contourpy.contour_generator(x=us, y=vs, z=P_high)
    iso_contours: dict[float, list[np.ndarray]] = {}
    for level in (0.25, 0.50, 0.70):
        segs_xyz = []
        for seg in cg.lines(level):
            seg_arr = np.asarray(seg, dtype=np.float64)
            yx = seg_arr[:, [1, 0]]
            segs_xyz.append(np.column_stack([interp_X(yx), interp_Y(yx), interp_Z(yx)]))
        iso_contours[level] = segs_xyz

    return {
        "us": us,
        "vs": vs,
        "U": U,
        "V": V,
        "X": X,
        "Y": Y,
        "Z": Z,
        "J": J,
        "drho": drho,
        "P_high": P_high,
        "rho_early": rho_early,
        "rho_late": rho_late,
        "rho_all": rho_all,
        "gamma": float(gamma),
        "min_jacobian": float(J.min()),
        "j_imm_mean": j_imm,
        "j_emi_mean": j_emi,
        "corr_drho_p": corr_drho_p,
        "imm_peak": imm_peak,
        "emi_peak": emi_peak,
        "top_z_peak": top_z_peak,
        "iso_contours": iso_contours,
        "mol_pts": mol_pts,
        "gens": gens,
        "scores": scores,
    }


def compute_ucb_cube_data(
    bundle: dict[str, Any],
    n_face: int = 61,
) -> dict[str, Any]:
    """Compute normalized [0, 1]^3 3-axis modification-probability evolution cube across Whole, Local, Score."""
    from scipy.interpolate import CubicSpline

    rdir = Path(bundle.get("run_dir", DEFAULT_RUN_DIR))
    traj = bundle["trajectory"]
    uid_to_row = {str(r["uid"]): r for r in traj}
    run_report = _read_json(rdir / "RUN_REPORT.json")

    local_units: dict[str, frozenset[str]] = {}
    for stage in ["BOOTSTRAP"] + [f"G{g}" for g in range(1, 20)]:
        in_path = rdir / stage / "INPUTS.json"
        if not in_path.exists():
            continue
        for item in _read_json(in_path):
            units = frozenset(
                str(t.get("unit", t.get("kind", ""))) for t in item.get("catalyst_metadata", [])
            )
            local_units[str(item["uid"])] = units

    g0_scores = [float(r["score"]) for r in traj if int(r["generation"]) == 0]
    g0_mean_score = float(np.mean(g0_scores))

    gen_rows: list[dict[str, Any]] = []
    probs_list: list[list[float]] = []
    round_lookup = {int(rd["generation"]): rd for rd in run_report.get("rounds", [])}

    for g in range(1, 7):
        rows_g = [r for r in traj if int(r["generation"]) == g]
        n_g = max(len(rows_g), 1)
        w_chg = 0
        l_chg = 0
        s_above_g0 = 0
        s_up_par = 0
        for row in rows_g:
            uid = str(row["uid"])
            puid = row.get("parent_uid")
            if float(row["score"]) > g0_mean_score:
                s_above_g0 += 1
            if puid and str(puid) in uid_to_row:
                par = uid_to_row[str(puid)]
                if int(row["cell"]) != int(par["cell"]):
                    w_chg += 1
                if local_units.get(uid, frozenset()) != local_units.get(str(puid), frozenset()):
                    l_chg += 1
                if float(row["score"]) > float(par["score"]):
                    s_up_par += 1
        pw = w_chg / n_g
        pl = l_chg / n_g
        ps = s_above_g0 / n_g
        ps_par = s_up_par / n_g
        rd = round_lookup.get(g, {})
        probs_list.append([pw, pl, ps])
        gen_rows.append(
            {
                "generation": g,
                "n": len(rows_g),
                "p_whole": pw,
                "p_local": pl,
                "p_score_above_g0": ps,
                "p_score_above_parent": ps_par,
                "policy": str(rd.get("native_ga", {}).get("policy", "")),
                "local_gap": float(rd.get("local_gap", np.nan)),
            }
        )

    gen_probs = np.asarray(probs_list, dtype=np.float64)  # (6, 3): G1..G6
    mean_probs = gen_probs.mean(axis=0)
    late_probs = gen_probs[3:].mean(axis=0)
    g6_probs = gen_probs[-1]

    # Normalized deformed molecular cube mapping (u, v, w) in [0, 1]^3 -> (X, Y, Z) inside [0.02, 0.98]^3
    x0 = 0.14 + 0.14 * float(mean_probs[0])
    y0 = 0.14 + 0.12 * float(mean_probs[1])
    z0 = 0.12 + 0.16 * float(late_probs[2])
    L_x = 0.64 * float(g6_probs[0])
    L_y = 0.68 * float(mean_probs[1])
    L_z = 0.66 * float(g6_probs[2])

    delta_pz = float(g6_probs[2] - gen_probs[0, 2])
    delta_px = float(g6_probs[0] - gen_probs[2, 0])
    p_loc_dom = float(mean_probs[1])

    def _map_mol_cube(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = x0 + L_x * u + (0.16 * max(delta_px, 0.0)) * u * np.exp(-2.0 * ((v - 0.5) ** 2 + (w - 0.55) ** 2))
        y = y0 + L_y * v + (0.062 * p_loc_dom) * v * np.exp(-2.0 * ((u - 0.5) ** 2 + (w - 0.45) ** 2))
        z = (
            z0
            + L_z * w
            + (0.16 * max(delta_pz, 0.0)) * w * np.exp(-2.2 * ((u - 0.55) ** 2 + (v - 0.55) ** 2))
            + 0.055 * (0.5 * u + 0.5 * v) * w
        )
        return x, y, z

    # Verify non-folding normalized 3D Jacobian and strict containment in [0, 1]^3
    t_chk = np.linspace(0.0, 1.0, 17)
    Uc, Vc, Wc = np.meshgrid(t_chk, t_chk, t_chk, indexing="ij")
    Xc, Yc, Zc = _map_mol_cube(Uc, Vc, Wc)
    bounds_ok = (
        float(min(Xc.min(), Yc.min(), Zc.min())) >= 0.02
        and float(max(Xc.max(), Yc.max(), Zc.max())) <= 0.98
    )
    assert bounds_ok, "Deformed molecular cube exceeds [0, 1]^3 bounds"

    J_mat = np.zeros(Uc.shape + (3, 3), dtype=np.float64)
    for idx_f, field_c in enumerate((Xc, Yc, Zc)):
        gx_f, gy_f, gz_f = np.gradient(field_c, t_chk, t_chk, t_chk)
        J_mat[..., idx_f, 0] = gx_f
        J_mat[..., idx_f, 1] = gy_f
        J_mat[..., idx_f, 2] = gz_f
    det_j_min = float(np.linalg.det(J_mat).min())
    j_rel_min = det_j_min / max(L_x * L_y * L_z, 1e-12)
    assert j_rel_min > 0.45, f"Deformed molecular cube folds: j_rel_min={j_rel_min:.3f}"

    def _prob_color(u_f: np.ndarray, v_f: np.ndarray, w_f: np.ndarray) -> np.ndarray:
        px = 0.24 + (float(g6_probs[0]) - 0.24) * u_f
        py = 0.28 + (float(mean_probs[1]) - 0.28) * v_f
        pz = 0.20 + (float(g6_probs[2]) - 0.20) * w_f
        # Emphasize evolved upper-front corner progression smoothly inside [0.20, 0.85]
        return np.clip(0.32 * px + 0.30 * py + 0.38 * pz + 0.06 * (u_f * w_f), 0.20, 0.85)

    ts = np.linspace(0.0, 1.0, n_face)
    A, B = np.meshgrid(ts, ts)

    faces: dict[str, dict[str, np.ndarray]] = {}
    for fname, (u_arr, v_arr, w_arr), out_dir in [
        ("top", (A, B, np.ones_like(A)), (0.0, 0.0, 1.0)),
        ("front_x", (np.ones_like(A), A, B), (1.0, 0.0, 0.0)),
        ("front_y", (A, np.zeros_like(A), B), (0.0, -1.0, 0.0)),
    ]:
        Xf, Yf, Zf = _map_mol_cube(u_arr, v_arr, w_arr)
        Cf = _prob_color(u_arr, v_arr, w_arr)
        faces[fname] = {
            "X": Xf,
            "Y": Yf,
            "Z": Zf,
            "C": Cf,
            "outward": np.asarray(out_dir, dtype=np.float64),
        }

    back_faces: dict[str, dict[str, np.ndarray]] = {}
    for bname, (u_arr, v_arr, w_arr), out_dir in [
        ("back_bottom", (A, B, np.zeros_like(A)), (0.0, 0.0, -1.0)),
        ("back_x", (np.zeros_like(A), A, B), (-1.0, 0.0, 0.0)),
        ("back_y", (A, np.ones_like(A), B), (0.0, 1.0, 0.0)),
    ]:
        Xf, Yf, Zf = _map_mol_cube(u_arr, v_arr, w_arr)
        Cf = _prob_color(u_arr, v_arr, w_arr)
        back_faces[bname] = {
            "X": Xf,
            "Y": Yf,
            "Z": Zf,
            "C": Cf,
            "outward": np.asarray(out_dir, dtype=np.float64),
        }

    # Cube edge topologies for [0, 1]^3 outer cube, G0 baseline cube, and deformed molecular cube
    front_edge_defs = [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 0), (1, 1, 1)),
    ]
    outer_back_defs = [
        ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 0, 0)),
        ((0, 1, 0), (0, 1, 1)),
    ]
    outer_mid_defs = [
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1)),
    ]
    outer_front_defs = [
        ((1, 0, 0), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (1, 1, 1)),
    ]

    outer_back_edges = [np.asarray([a, b], dtype=np.float64) for a, b in outer_back_defs]
    outer_mid_edges = [np.asarray([a, b], dtype=np.float64) for a, b in outer_mid_defs]
    outer_front_edges = [np.asarray([a, b], dtype=np.float64) for a, b in outer_front_defs]

    t_edge = np.linspace(0.0, 1.0, n_face)

    def _curved_edges(edge_defs: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> list[np.ndarray]:
        out_edges = []
        for a_pt, b_pt in edge_defs:
            u_e = a_pt[0] + t_edge * (b_pt[0] - a_pt[0])
            v_e = a_pt[1] + t_edge * (b_pt[1] - a_pt[1])
            w_e = a_pt[2] + t_edge * (b_pt[2] - a_pt[2])
            xe, ye, ze = _map_mol_cube(u_e, v_e, w_e)
            out_edges.append(np.column_stack([xe, ye, ze]))
        return out_edges

    deformed_edges = _curved_edges(front_edge_defs)
    deformed_back_edges = _curved_edges(outer_back_defs)

    # Mid-height equatorial contour loop at w = 0.50 around the 4 side walls
    s_edge = np.linspace(0.0, 1.0, 41)
    u_mid = np.concatenate([s_edge, np.ones_like(s_edge), s_edge[::-1], np.zeros_like(s_edge)])
    v_mid = np.concatenate([np.zeros_like(s_edge), s_edge, np.ones_like(s_edge), s_edge[::-1]])
    w_mid = np.full_like(u_mid, 0.50)
    xm, ym, zm = _map_mol_cube(u_mid, v_mid, w_mid)
    mid_ring = np.column_stack([xm, ym, zm])

    # 3-wall orthographic projections of deformed molecular cube slices
    def _slice_loop(u_arr: np.ndarray, v_arr: np.ndarray, w_arr: np.ndarray) -> np.ndarray:
        xl, yl, zl = _map_mol_cube(u_arr, v_arr, w_arr)
        return np.column_stack([xl, yl, zl])

    f_loop = _slice_loop(u_mid, v_mid, np.full_like(u_mid, 0.75))
    f_loop[:, 2] = 0.0

    u_wx = np.concatenate([s_edge, np.ones_like(s_edge), s_edge[::-1], np.zeros_like(s_edge)])
    w_wx = np.concatenate([np.zeros_like(s_edge), s_edge, np.ones_like(s_edge), s_edge[::-1]])
    wx_loop = _slice_loop(u_wx, np.full_like(u_wx, 0.50), w_wx)
    wx_loop[:, 1] = 1.0

    wy_loop = _slice_loop(np.full_like(u_wx, 0.50), u_wx, w_wx)
    wy_loop[:, 0] = 0.0

    wall_projections = {
        "floor_xy": f_loop,
        "wall_xz": wx_loop,
        "wall_yz": wy_loop,
    }

    # Baseline G0 normalized cube inside lower-front octant [0.06, 0.34] x [0.06, 0.34] x [0.06, 0.32]
    g0_lo = np.array([0.06, 0.06, 0.06], dtype=np.float64)
    g0_span = np.array([0.28, 0.28, 0.26], dtype=np.float64)
    baseline_g0_edges = [
        np.asarray([g0_lo + np.array(a) * g0_span, g0_lo + np.array(b) * g0_span], dtype=np.float64)
        for a, b in (front_edge_defs + outer_back_defs)
    ]

    g0_wall_rects = [
        np.array([[0.06, 0.06, 0.0], [0.34, 0.06, 0.0], [0.34, 0.34, 0.0], [0.06, 0.34, 0.0], [0.06, 0.06, 0.0]]),
        np.array([[0.06, 1.0, 0.06], [0.34, 1.0, 0.06], [0.34, 1.0, 0.32], [0.06, 1.0, 0.32], [0.06, 1.0, 0.06]]),
        np.array([[0.0, 0.06, 0.06], [0.0, 0.34, 0.06], [0.0, 0.34, 0.32], [0.0, 0.06, 0.32], [0.0, 0.06, 0.06]]),
    ]

    # Smooth G1->G6 probability trajectories in 3D and projected on floor (z=0), back wall (y=1), left wall (x=0)
    t_g = np.arange(6, dtype=np.float64)
    t_dense = np.linspace(0.0, 5.0, 90)
    sp_x = np.clip(CubicSpline(t_g, gen_probs[:, 0])(t_dense), 0.0, 1.0)
    sp_y = np.clip(CubicSpline(t_g, gen_probs[:, 1])(t_dense), 0.0, 1.0)
    sp_z = np.clip(CubicSpline(t_g, gen_probs[:, 2])(t_dense), 0.0, 1.0)
    trend_3d = np.column_stack([sp_x, sp_y, sp_z])
    trend_floor = np.column_stack([sp_x, sp_y, np.zeros_like(t_dense)])
    trend_wall = np.column_stack([sp_x, np.ones_like(t_dense), sp_z])
    trend_left_wall = np.column_stack([np.zeros_like(t_dense), sp_y, sp_z])

    xk, yk, zk = _map_mol_cube(np.array([0.70]), np.array([0.70]), np.array([1.0]))
    landmarks = {
        "g0_ref": np.array([0.06, 0.06, 0.32], dtype=np.float64),
        "g6_peak": np.array([float(xk[0]), float(yk[0]), float(zk[0])], dtype=np.float64),
        "wall_g6": np.array([float(g6_probs[0]), 1.0, float(g6_probs[2])], dtype=np.float64),
    }

    return {
        "faces": faces,
        "back_faces": back_faces,
        "outer_back_edges": outer_back_edges,
        "outer_mid_edges": outer_mid_edges,
        "outer_front_edges": outer_front_edges,
        "baseline_g0_edges": baseline_g0_edges,
        "deformed_edges": deformed_edges,
        "deformed_back_edges": deformed_back_edges,
        "mid_ring": mid_ring,
        "wall_projections": wall_projections,
        "g0_wall_rects": g0_wall_rects,
        "trend_3d": trend_3d,
        "trend_floor": trend_floor,
        "trend_wall": trend_wall,
        "trend_left_wall": trend_left_wall,
        "gen_probs": gen_probs,
        "gen_table": gen_rows,
        "mean_probs": mean_probs,
        "g6_probs": g6_probs,
        "j_rel_min": float(j_rel_min),
        "bounds_ok": bool(bounds_ok),
        "landmarks": landmarks,
    }


def compute_llm_directed_som_data(
    bundle: dict[str, Any],
    generation: int = 6,
) -> dict[str, Any]:
    """Extract Track-2 LLM two-step directed editing telemetry and 8x16 planar SOM response shift."""
    rdir = Path(bundle["run_dir"])
    cands_raw = _read_json(rdir / f"G{generation}_DESIGN_CANDIDATES.json")
    result_raw = _read_json(rdir / f"G{generation}_DESIGN_RESULT.json")
    psi_npz = dict(np.load(rdir / f"G{generation}_DESIGN_PSI.npz", allow_pickle=True))

    root_psi_full = np.asarray(psi_npz["root_psi"], dtype=np.float64)
    if not np.isclose(root_psi_full[0], 1.0):
        raise ValueError(f"Expected bias=1.0 at index 0 of root_psi, got {root_psi_full[0]}")

    cand_ids = [str(x) for x in psi_npz["candidate_ids"]]
    cand_psi_full = np.asarray(psi_npz["candidate_psi"], dtype=np.float64)

    chosen_ids = [str(x) for x in result_raw["answer"]["edit_ids"]]
    if not chosen_ids:
        raise ValueError(f"No chosen edit_ids found in G{generation}_DESIGN_RESULT.json")
    step1_id = chosen_ids[0]
    step2_id = chosen_ids[-1]

    psi_root = root_psi_full[1:]
    psi_step1 = cand_psi_full[cand_ids.index(step1_id), 1:]
    psi_step2 = cand_psi_full[cand_ids.index(step2_id), 1:]

    delta_psi_step1 = psi_step1 - psi_root
    delta_psi_total = psi_step2 - psi_root

    nodes_df = bundle["nodes_df"]
    n_rows, n_cols = 8, 16
    grid_delta_psi = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    grid_delta_psi_step1 = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    grid_cell_id = np.full((n_rows, n_cols), -1, dtype=int)
    grid_occupied = np.zeros((n_rows, n_cols), dtype=int)
    grid_whole_mean = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    grid_combined_mean = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    cell_to_rc: dict[int, tuple[int, int]] = {}

    for rec in nodes_df.to_dict("records"):
        cid = int(rec["cell"])
        col = int(round(float(rec["slot_x"]) - 0.5))
        row = int(round(float(rec["slot_y"]) - 0.5))
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(f"Cell {cid} slot ({row}, {col}) out of 8x16 bounds")
        if grid_cell_id[row, col] != -1:
            raise ValueError(f"Duplicate SOM slot ({row}, {col}) for cells {grid_cell_id[row, col]} and {cid}")
        cell_to_rc[cid] = (row, col)
        grid_cell_id[row, col] = cid
        grid_occupied[row, col] = int(rec["occupied"])
        grid_whole_mean[row, col] = float(rec["whole_mean"])
        grid_combined_mean[row, col] = float(rec["combined_mean"])
        grid_delta_psi[row, col] = float(delta_psi_total[cid])
        grid_delta_psi_step1[row, col] = float(delta_psi_step1[cid])

    if len(cell_to_rc) != 128 or np.isnan(grid_delta_psi).any():
        raise ValueError(f"Expected 128 unique SOM cells on 8x16 grid, got {len(cell_to_rc)}")

    cand_by_id = {str(c["id"]): dict(c) for c in cands_raw["candidates"]}
    root_info = dict(cands_raw["root"])
    step1_info = cand_by_id[step1_id]
    step2_info = cand_by_id[step2_id]

    rejected_list = []
    for cid_key, crec in cand_by_id.items():
        if cid_key not in chosen_ids:
            rejected_list.append(crec)

    # Summarize G1..G6 Track-2 vs ordinary population performance
    traj = bundle["trajectory"]
    gen_summaries: list[dict[str, Any]] = []
    for g in range(1, 7):
        gcands = _read_json(rdir / f"G{g}_DESIGN_CANDIDATES.json")
        gres = _read_json(rdir / f"G{g}_DESIGN_RESULT.json")
        g_edit_ids = [str(x) for x in gres["answer"]["edit_ids"]]
        gc_map = {str(c["id"]): c for c in gcands["candidates"]}
        g_rows = [m for m in traj if int(m["generation"]) == g]
        g_ord = [m for m in g_rows if m.get("role") != "directed_parent"]
        g_dir = [m for m in g_rows if m.get("role") == "directed_parent"]
        ord_mean = float(np.mean([float(m["score"]) for m in g_ord])) if g_ord else np.nan
        dir_score = float(g_dir[0]["score"]) if g_dir else np.nan
        path_cells = [int(gcands["root"]["cell"])] + [int(gc_map[eid]["cell"]) for eid in g_edit_ids]
        gen_summaries.append(
            {
                "generation": g,
                "edit_ids": g_edit_ids,
                "path_cells": path_cells,
                "root_h2": float(gcands["root"]["h2_wt_percent"]),
                "final_h2": float(gc_map[g_edit_ids[-1]]["h2_wt_percent"]),
                "dir_score": dir_score,
                "ord_mean_score": ord_mean,
                "score_lift": float(dir_score - ord_mean),
            }
        )

    som_rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            cid = int(grid_cell_id[r, c])
            som_rows.append(
                {
                    "row": r,
                    "col": c,
                    "cell": cid,
                    "occupied": int(grid_occupied[r, c]),
                    "psi_root": float(psi_root[cid]),
                    "psi_step1": float(psi_step1[cid]),
                    "psi_step2": float(psi_step2[cid]),
                    "delta_psi_step1": float(grid_delta_psi_step1[r, c]),
                    "delta_psi_total": float(grid_delta_psi[r, c]),
                    "whole_mean": float(grid_whole_mean[r, c]),
                    "combined_mean": float(grid_combined_mean[r, c]),
                }
            )

    return {
        "generation": int(generation),
        "chosen_ids": chosen_ids,
        "reason": str(result_raw["answer"]["reason"]),
        "root": root_info,
        "step1": step1_info,
        "step2": step2_info,
        "candidates_by_id": cand_by_id,
        "rejected": rejected_list,
        "psi_root": psi_root,
        "psi_step1": psi_step1,
        "psi_step2": psi_step2,
        "delta_psi_step1": delta_psi_step1,
        "delta_psi_total": delta_psi_total,
        "argmax_cells": {
            "root": int(np.argmax(psi_root)),
            "step1": int(np.argmax(psi_step1)),
            "step2": int(np.argmax(psi_step2)),
        },
        "grid_delta_psi": grid_delta_psi,
        "grid_delta_psi_step1": grid_delta_psi_step1,
        "grid_cell_id": grid_cell_id,
        "grid_occupied": grid_occupied,
        "grid_whole_mean": grid_whole_mean,
        "grid_combined_mean": grid_combined_mean,
        "cell_to_rc": cell_to_rc,
        "gen_summaries": gen_summaries,
        "som_table": som_rows,
    }
