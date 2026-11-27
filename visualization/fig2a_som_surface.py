"""Core Plate I (Fig. 2), Panel a: Organic 3-subfigure SOM Structure & Score Differential Manifold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import binary_dilation

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(r"D:\skills\nature-skills\skills\nature-figure\scripts")
for _p in (ROOT_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from audit_panel_alignment import require_matplotlib_panel_alignment
from visualization.loader import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RUN_DIR,
    compute_differential_fields,
    load_run_bundle,
)
from visualization.style import (
    COOL_DARK,
    DARK_INK,
    GRID_HEX,
    GRID_MAJOR_HEX,
    INK,
    MUTED,
    SCORE_DIFF_CMAP,
    STRUCT_DIFF_CMAP,
    WINE,
    apply_nature_som_style,
    make_organic_feathered_shadow,
    shade_surface_white_composited,
)

# Mandatory editable-text configuration for static preflight
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def _project_3d_to_axes(ax: plt.Axes, x: float, y: float, z: float) -> tuple[float, float]:
    """Project a 3D data coordinate to 2D axes-fraction coordinates."""
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp = ax.transData.transform((x2, y2))
    ax_pt = ax.transAxes.inverted().transform(disp)
    return float(ax_pt[0]), float(ax_pt[1])


def _clipped_grid_lines_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    mask: np.ndarray,
    n_u: int,
    n_v: int,
    major_step: int = 4,
    z_offset: float = 0.003,
    n_seg: int = 48,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample uniform-u and uniform-v material lines clipped strictly to the organic mask."""
    nv, nu = X.shape
    minor, major = [], []

    def _split_and_resample(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, m_arr: np.ndarray, is_major: bool) -> None:
        idx = np.flatnonzero(m_arr)
        if len(idx) < 3:
            return
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for run in np.split(idx, splits):
            if len(run) < 3:
                continue
            t_old = np.linspace(0.0, 1.0, len(run))
            assert np.all(np.diff(t_old) > 0)
            t_new = np.linspace(0.0, 1.0, n_seg)
            seg = np.column_stack(
                [
                    np.interp(t_new, t_old, x_arr[run]),
                    np.interp(t_new, t_old, y_arr[run]),
                    np.interp(t_new, t_old, z_arr[run]) + z_offset,
                ]
            )
            (major if is_major else minor).append(seg)

    for k, ci in enumerate(np.linspace(0, nu - 1, n_u).round().astype(int)):
        _split_and_resample(X[:, ci], Y[:, ci], Z[:, ci], mask[:, ci], k % major_step == 0)
    for k, ri in enumerate(np.linspace(0, nv - 1, n_v).round().astype(int)):
        _split_and_resample(X[ri, :], Y[ri, :], Z[ri, :], mask[ri, :], k % major_step == 0)
    return minor, major


def _draped_iso_contours(
    us: np.ndarray,
    vs: np.ndarray,
    scalar_grid: np.ndarray,
    mask: np.ndarray,
    interp_Z,
    levels: np.ndarray,
    z_offset: float = 0.004,
    n_seg: int = 48,
) -> list[np.ndarray]:
    """Generate smooth iso-level gradient contour lines draped on an organic surface (no arrows)."""
    masked_val = np.where(mask, scalar_grid, np.nan)
    cg = contourpy.contour_generator(x=us, y=vs, z=masked_val)
    segments = []
    t_new = np.linspace(0.0, 1.0, n_seg)
    for lvl in levels:
        for line in cg.lines(float(lvl)):
            if len(line) < 5:
                continue
            z_line = interp_Z(line[:, [1, 0]]) + z_offset
            t_old = np.linspace(0.0, 1.0, len(line))
            seg = np.column_stack(
                [
                    np.interp(t_new, t_old, line[:, 0]),
                    np.interp(t_new, t_old, line[:, 1]),
                    np.interp(t_new, t_old, z_line),
                ]
            )
            segments.append(seg)
    return segments


def _draw_gradient_panel(
    ax: plt.Axes,
    fields: dict,
    kind: str,
) -> tuple[Normalize, object]:
    """Draw organic gradient-and-stain 3D manifold (zero arrows, zero discrete dots)."""
    U, V = fields["U"], fields["V"]
    mask = fields["organic_mask"]
    ax.computed_zorder = False

    if kind == "structure":
        val_grid = fields["struct_mag_grid"]
        vmax = max(0.50, float(np.ceil(val_grid[mask].max() * 20.0) / 20.0))
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = STRUCT_DIFF_CMAP
        Z_surf = fields["Z_struct"]
        interp_Z = fields["interp_Z_struct"]
        rim_xyz = fields["rim_xyz_struct"]
        stain = fields["mol_density_new"]
        stain_hex = "#381F60"
        contour_color = "#543B80"
        levels = np.linspace(0.10, float(val_grid[mask].max()) * 0.90, 5)
    else:
        val_grid = fields["dmu_grid"]
        vmin = float(np.floor(val_grid[mask].min() * 20.0) / 20.0)
        vmax = float(np.ceil(val_grid[mask].max() * 20.0) / 20.0)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = SCORE_DIFF_CMAP
        Z_surf = fields["Z_score"]
        interp_Z = fields["interp_Z_score"]
        rim_xyz = fields["rim_xyz_score"]
        stain = fields["mol_density_all"]
        stain_hex = "#5A1228"
        contour_color = "#7D2340"
        levels = np.linspace(vmin + 0.30, vmax - 0.25, 5)

    # Whisper-faint organic ghost mesh at zero plane
    ghost_minor, _ = _clipped_grid_lines_3d(
        U, V, np.zeros_like(Z_surf), mask, n_u=13, n_v=11, major_step=99, z_offset=-0.025
    )
    ax.add_collection3d(Line3DCollection(ghost_minor, colors="#CFC5DC", linewidths=0.20, alpha=0.28, zorder=0))
    ax.plot(
        fields["rim_uv"][:, 0],
        fields["rim_uv"][:, 1],
        np.full(len(fields["rim_uv"]), -0.025),
        color="#C2B6D2",
        lw=0.38,
        alpha=0.45,
        zorder=0,
    )

    # Compute white-composited RGBA on unmasked surface (clean border normals), then crop to organic hull
    rgba = shade_surface_white_composited(
        U,
        V,
        Z_surf,
        val_grid,
        cmap,
        norm,
        opacity_base=0.82,
        edge_taper=False,
        stain_density=stain,
        stain_hex=stain_hex,
        stain_strength=0.48,
    )
    # Dilate plot mask by 1 cell so no stair-step gap opens inside the smooth contourpy rim curve
    plot_mask = binary_dilation(mask, iterations=1)
    Z_masked = np.where(plot_mask, Z_surf, np.nan)
    ax.plot_surface(
        U,
        V,
        Z_masked,
        facecolors=rgba,
        shade=False,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        rcount=Z_surf.shape[0],
        ccount=Z_surf.shape[1],
        rasterized=True,
        zorder=1,
    )

    # Organic-clipped subtle mesh + smooth gradient iso-contours (replaces arrows)
    minor, major = _clipped_grid_lines_3d(U, V, Z_surf, mask, n_u=15, n_v=11, major_step=4, z_offset=0.003)
    ax.add_collection3d(Line3DCollection(minor, colors=GRID_HEX, linewidths=0.16, alpha=0.18, zorder=2))
    ax.add_collection3d(Line3DCollection(major, colors=GRID_MAJOR_HEX, linewidths=0.25, alpha=0.26, zorder=3))

    iso_segs = _draped_iso_contours(fields["us"], fields["vs"], val_grid, mask, interp_Z, levels, z_offset=0.0045)
    if iso_segs:
        ax.add_collection3d(Line3DCollection(iso_segs, colors=contour_color, linewidths=0.42, alpha=0.52, zorder=4))

    ax.plot(rim_xyz[:, 0], rim_xyz[:, 1], rim_xyz[:, 2] + 0.003, color="#837099", lw=0.48, alpha=0.65, zorder=5)

    ax.set_xlim(float(U.min()), float(U.max()))
    ax.set_ylim(float(V.min()), float(V.max()))
    ax.set_zlim(-0.04, 0.48)
    ax.view_init(elev=30, azim=-54)
    ax.set_box_aspect((1.34, 1.0, 0.54), zoom=1.35)
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    for coll in ax.collections:
        coll.set_clip_on(False)
    return norm, cmap


def _draw_coupled_hero(
    fig: plt.Figure,
    ax: plt.Axes,
    fields: dict,
    dmu_norm: Normalize,
) -> None:
    """Draw a3 Hero panel: XY = Kohonen SOM network warp, Z = Score shift (dmu), color = deep molecular pigment."""
    U, V = fields["U"], fields["V"]
    X, Y, Z_score = fields["X"], fields["Y"], fields["Z_score"]
    mask = fields["organic_mask"]
    dmu_grid = fields["dmu_grid"]

    ax.computed_zorder = False
    height = max(float(Z_score[mask].max() - Z_score[mask].min()), 1e-6)
    z_floor = float(Z_score[mask].min()) - 0.22 * height

    # Organic feathered floor shadow matching deformed island silhouette
    shadow = make_organic_feathered_shadow(mask, max_alpha=0.072, sigma=9.0)
    ax.plot_surface(
        X + 0.065,
        Y - 0.055,
        np.full_like(Z_score, z_floor),
        facecolors=shadow,
        shade=False,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        rcount=120,
        ccount=150,
        rasterized=True,
        zorder=0,
    )

    # Faint undeformed organic SOM footprint at floor level so in-plane deformation is unmistakable
    ghost_minor, _ = _clipped_grid_lines_3d(
        U, V, np.full_like(Z_score, z_floor + 0.012), mask, n_u=15, n_v=11, major_step=99, z_offset=0.0
    )
    ax.add_collection3d(Line3DCollection(ghost_minor, colors="#C5BAD4", linewidths=0.22, alpha=0.34, zorder=1))
    ax.plot(
        fields["rim_uv"][:, 0],
        fields["rim_uv"][:, 1],
        np.full(len(fields["rim_uv"]), z_floor + 0.012),
        color="#AEA0BF",
        lw=0.45,
        alpha=0.52,
        zorder=1,
    )

    # White-composited deformed organic surface: Z from dmu, color from dmu + deepened molecular stain (no dots!)
    rgba = shade_surface_white_composited(
        X,
        Y,
        Z_score,
        dmu_grid,
        SCORE_DIFF_CMAP,
        dmu_norm,
        opacity_base=0.80,
        edge_taper=False,
        stain_density=fields["mol_density_all"],
        stain_hex="#551026",
        stain_strength=0.54,
    )
    plot_mask = binary_dilation(mask, iterations=1)
    Z_masked = np.where(plot_mask, Z_score, np.nan)
    ax.plot_surface(
        X,
        Y,
        Z_masked,
        facecolors=rgba,
        shade=False,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        rcount=Z_score.shape[0],
        ccount=Z_score.shape[1],
        rasterized=True,
        zorder=2,
    )

    # 29 constant-u and 23 constant-v deformed material coordinate curves clipped cleanly to organic hull
    minor, major = _clipped_grid_lines_3d(X, Y, Z_score, mask, n_u=29, n_v=23, major_step=4, z_offset=0.004)
    ax.add_collection3d(Line3DCollection(minor, colors=GRID_HEX, linewidths=0.22, alpha=0.35, zorder=3))
    ax.add_collection3d(Line3DCollection(major, colors=GRID_MAJOR_HEX, linewidths=0.35, alpha=0.47, zorder=4))

    # Smooth deformed organic rim curve sealing the curved edge
    rim_d = fields["rim_xyz_def"]
    ax.plot(rim_d[:, 0], rim_d[:, 1], rim_d[:, 2] + 0.004, color="#6F5985", lw=0.58, alpha=0.72, zorder=5)

    ax.set_xlim(float(min(U.min(), X[mask].min())) - 0.08, float(max(U.max(), X[mask].max())) + 0.08)
    ax.set_ylim(float(min(V.min(), Y[mask].min())) - 0.08, float(max(V.max(), Y[mask].max())) + 0.08)
    ax.set_zlim(z_floor, float(Z_score[mask].max()) + 0.14 * height)
    ax.view_init(elev=31, azim=-54)
    ax.set_box_aspect((1.38, 1.0, 0.72), zoom=1.34)
    ax.patch.set_alpha(0.0)
    ax.axis("off")
    for coll in ax.collections:
        coll.set_clip_on(False)

    fig.canvas.draw()

    # Exterior region callouts kept strictly inside ax_a3 bounds (zero page clipping, zero line crossings)
    x_def, y_def, z_def = fields["x_def"], fields["y_def"], fields["z_def_score"]
    callouts = [
        (
            95,
            (0.03, 0.80),
            "left",
            "bottom",
            "High-score molecular ridge\n↑Z Δμ +2.54 · deep pigment",
            WINE,
            6.1,
        ),
        (
            23,
            (0.97, 0.80),
            "right",
            "bottom",
            "High-$\\mathrm{H_2}$ discovery basin\nSOM XY warp · 6.98–7.84 wt%",
            COOL_DARK,
            7.2,
        ),
        (
            105,
            (0.47, 0.09),
            "left",
            "top",
            "Dense evolved molecule pool\nPeak Δμ +2.64 · compressed grid",
            WINE,
            6.1,
        ),
    ]
    for cid, tx_xy, ha, va, label_str, color, fsize in callouts:
        ax_x, ax_y = _project_3d_to_axes(ax, float(x_def[cid]), float(y_def[cid]), float(z_def[cid]) + 0.006)
        ax.annotate(
            label_str,
            xy=(ax_x, ax_y),
            xycoords=ax.transAxes,
            xytext=tx_xy,
            textcoords=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fsize,
            linespacing=1.18,
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.58,
                "alpha": 0.80,
                "shrinkA": 3.5,
                "shrinkB": 4.2,
            },
            zorder=20,
        )


def draw_panel_a_triptych(fig: plt.Figure, bundle: dict, fields: dict) -> dict[str, object]:
    """Assemble the complete 3-subfigure organic Panel A (a1, a2, a3)."""
    ax_a1 = fig.add_axes([0.015, 0.545, 0.365, 0.335], projection="3d", computed_zorder=False)
    ax_a2 = fig.add_axes([0.015, 0.092, 0.365, 0.335], projection="3d", computed_zorder=False)
    ax_a3 = fig.add_axes([0.405, 0.140, 0.582, 0.745], projection="3d", computed_zorder=False)

    norm_s, cmap_s = _draw_gradient_panel(ax_a1, fields, "structure")
    norm_m, cmap_m = _draw_gradient_panel(ax_a2, fields, "score")
    _draw_coupled_hero(fig, ax_a3, fields, norm_m)

    sep = Line2D([0.394, 0.394], [0.055, 0.905], transform=fig.transFigure, color="#E4DCEC", lw=0.7)
    fig.add_artist(sep)

    # Header & sub-panel titles (pure Arial, no missing glyphs)
    fig.text(0.018, 0.972, "a", fontsize=8.5, fontweight="bold", color=DARK_INK, va="top")
    fig.text(
        0.036,
        0.971,
        "SOM-driven planar structural deformation and score-height coupling across the organic molecular manifold",
        fontsize=7.4,
        fontweight="bold",
        color=DARK_INK,
        va="top",
    )
    fig.text(
        0.036,
        0.940,
        "144 anchored molecules (G0 → G6) · Kohonen SOM XY warp (min Jacobian 0.28) · Z-axis driven by Kalman Δμ",
        fontsize=5.9,
        color=MUTED,
        va="top",
    )

    fig.text(0.022, 0.902, "a1", fontsize=7.5, fontweight="bold", color=DARK_INK, va="top")
    fig.text(0.045, 0.901, "SOM structure differential (||ΔC|| gradient)", fontsize=6.4, fontweight="bold", color=INK, va="top")

    fig.text(0.022, 0.448, "a2", fontsize=7.5, fontweight="bold", color=DARK_INK, va="top")
    fig.text(0.045, 0.447, "Kalman score differential (Δμ gradient)", fontsize=6.4, fontweight="bold", color=INK, va="top")

    fig.text(0.415, 0.902, "a3", fontsize=7.5, fontweight="bold", color=DARK_INK, va="top")
    fig.text(0.438, 0.901, "Deformed molecular space: XY = SOM ΔC warp · Z = score Δμ · deep tint = molecules", fontsize=6.4, fontweight="bold", color=INK, va="top")

    cax_a1 = fig.add_axes([0.072, 0.530, 0.245, 0.012])
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_s, cmap=cmap_s),
        cax=cax_a1,
        orientation="horizontal",
        ticks=[0.0, round(norm_s.vmax / 2.0, 2), norm_s.vmax],
    )
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=5.5, length=1.6, width=0.45, pad=1.6, colors=INK)
    cb1.set_label("Structure shift ||ΔC|| (deeper = evolved molecules)", fontsize=5.5, labelpad=1.8, color=INK)

    cax_a2 = fig.add_axes([0.072, 0.062, 0.245, 0.012])
    ticks_m = [round(norm_m.vmin, 2), round(0.5 * (norm_m.vmin + norm_m.vmax), 2), round(norm_m.vmax, 2)]
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m),
        cax=cax_a2,
        orientation="horizontal",
        ticks=ticks_m,
    )
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(labelsize=5.5, length=1.6, width=0.45, pad=1.6, colors=INK)
    cb2.set_label("Score differential Δμ = μ(G6) − μ(G0)", fontsize=5.5, labelpad=1.8, color=INK)

    # Compact key without scatter-dot symbols
    legend_handles = [
        Patch(facecolor="#551026", edgecolor="none", label="Deep pigment: anchored molecules (n = 144)"),
        Line2D([0], [0], color=GRID_MAJOR_HEX, lw=0.95, label="Warped 29×23 grid (SOM Δp)"),
        Line2D([0], [0], color="#AEA0BF", lw=0.70, label="Initial frozen footprint"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.695, 0.114),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.0,
        handletextpad=0.50,
        fontsize=5.6,
        frameon=False,
    )

    cax_a3 = fig.add_axes([0.525, 0.062, 0.340, 0.014])
    cb3 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m),
        cax=cax_a3,
        orientation="horizontal",
        ticks=ticks_m,
    )
    cb3.outline.set_visible(False)
    cb3.ax.tick_params(labelsize=5.7, length=1.8, width=0.5, pad=1.8, colors=INK)
    cb3.set_label("Z elevation & base gradient: Δμ (G6 − G0)  |  XY warp: SOM ΔC", fontsize=5.8, labelpad=2.2, color=INK)

    return {
        "axes": [ax_a1, ax_a2, ax_a3],
        "caxes": [cax_a1, cax_a2, cax_a3],
    }


def export_source_data(bundle: dict, fields: dict, out_dir: Path) -> None:
    """Save traceable node, molecule-anchor, and surface NPZ source data."""
    sdir = out_dir / "source_data"
    sdir.mkdir(parents=True, exist_ok=True)
    df = bundle["nodes_df"].copy()
    df["dC_norm"] = fields["dC_norm"]
    df["dp_som_x"] = fields["dp_som"][:, 0]
    df["dp_som_y"] = fields["dp_som"][:, 1]
    df["dmu"] = fields["dmu"]
    df["x_deformed"] = fields["x_def"]
    df["y_deformed"] = fields["y_def"]
    df["z_score_deformed"] = fields["z_def_score"]
    df.to_csv(sdir / "fig2a_differential_nodes.csv", index=False)

    mol_df = pd.DataFrame(
        {
            "generation": fields["mol_gen"],
            "cell": fields["mol_cell"],
            "score": fields["mol_score"],
            "h2_wt_percent": fields["mol_h2"],
            "u_som": fields["mol_uv"][:, 0],
            "v_som": fields["mol_uv"][:, 1],
            "x_deformed": fields["mol_xyz_def"][:, 0],
            "y_deformed": fields["mol_xyz_def"][:, 1],
            "z_score_deformed": fields["mol_xyz_def"][:, 2],
        }
    )
    mol_df.to_csv(sdir / "fig2a_anchored_molecules.csv", index=False)

    np.savez_compressed(
        sdir / "fig2a_coupled_surface.npz",
        U=fields["U"],
        V=fields["V"],
        X=fields["X"],
        Y=fields["Y"],
        Z_score=fields["Z_score"],
        Z_struct=fields["Z_struct"],
        J=fields["J"],
        organic_mask=fields["organic_mask"],
        struct_mag_grid=fields["struct_mag_grid"],
        dmu_grid=fields["dmu_grid"],
        mol_density_all=fields["mol_density_all"],
        mol_density_new=fields["mol_density_new"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render organic 3-subfigure Fig. 2 Panel A.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "visualization" / "output" / "fig2_panels")
    args = parser.parse_args()

    apply_nature_som_style(font_size=6.4, axes_linewidth=0.65)
    bundle = load_run_bundle(args.run_dir, args.model_path)
    fields = compute_differential_fields(bundle)

    fig = plt.figure(figsize=(183 / 25.4, 114 / 25.4), facecolor="white")
    layout_info = draw_panel_a_triptych(fig, bundle, fields)

    export_source_data(bundle, fields, args.out_dir)

    contract = {
        "figure": "Fig. 2 Panel A (Organic SOM-deformed & score-elevated triptych)",
        "core_conclusion": (
            "As 144 molecules accumulate across 6 closed-loop generations, Kohonen SOM structural differentials "
            "warp the organic molecular plane in (X, Y) while Kalman score differentials elevate Z, with deeper "
            "embedded pigment encoding anchored molecular concentrations without discrete points or arrows."
        ),
        "archetype": "asymmetric mixed-modality figure (two stacked differential gradients + one hero coupled organic manifold)",
        "backend": "Python matplotlib",
        "size_mm": [183, 114],
        "run_dir": str(args.run_dir),
        "som_model": str(args.model_path),
        "metrics": {
            "molecules": len(bundle["trajectory"]),
            "occupied_cells": int((bundle["nodes_df"]["occupied"] > 0).sum()),
            "min_planar_jacobian": round(fields["min_jacobian"], 4),
            "dC_norm_range": [round(float(fields["dC_norm"].min()), 4), round(float(fields["dC_norm"].max()), 4)],
            "dmu_range": [round(float(fields["dmu"].min()), 4), round(float(fields["dmu"].max()), 4)],
        },
        "scope_notes": [
            "Organic boundary follows smoothed distance envelope (d_smooth <= 0.22) around the 128 frozen SOM prototypes; no artificial rectangular clipping.",
            "No arrows or discrete point markers are plotted; gradient progression uses smooth color stops + iso-level contours and molecular anchoring uses continuous KDE pigment deepening.",
            "In a3, planar grid deformation (X, Y) is strictly governed by the Kohonen SOM training-rule update from G0 to G6, and vertical elevation Z is strictly governed by Kalman score differential dmu.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig,
        axes=layout_info["axes"],
        panel_ids=["a1", "a2", "a3"],
        column_groups=[["a1", "a2"]],
        row_groups=[],
        exclude_axes=layout_info["caxes"],
        exemptions=[],
        json_out=args.out_dir / "Fig2a_som_surface.alignment.json",
        overlay_svg=args.out_dir / "Fig2a_som_surface.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )

    fig.savefig(args.out_dir / "Fig2a_som_surface.svg", facecolor="white")
    fig.savefig(args.out_dir / "Fig2a_som_surface.pdf", facecolor="white")
    fig.savefig(args.out_dir / "Fig2a_som_surface.png", dpi=600, facecolor="white")
    fig.savefig(args.out_dir / "Fig2a_som_surface.tiff", dpi=600, facecolor="white")
    (args.out_dir / "Fig2a_som_surface.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plt.close(fig)
    print("Rendered organic 3-subfigure Fig2a_som_surface successfully to", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
