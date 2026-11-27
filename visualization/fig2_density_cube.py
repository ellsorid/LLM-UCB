"""Transparent-cube 3D molecular probability-density volume over SOM structure (XY) and 1D score ascent (Z)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
NATURE_FIG_SCRIPTS = Path(r"D:\skills\nature-skills\skills\nature-figure\scripts")
for _p in (str(THIS_DIR), str(NATURE_FIG_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_panel_alignment import require_matplotlib_panel_alignment
import loader
import style

OUT_DIR = THIS_DIR / "output" / "fig2_saddle"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUT_DIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)


def _camera_basis(
    elev_deg: float = 25.0,
    azim_deg: float = -54.0,
    box_aspect: tuple[float, float, float] = (1.12, 1.00, 0.88),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact shared orthographic camera basis matching Matplotlib's elev/azim."""
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float64)
    view /= np.linalg.norm(view)
    right = np.array([-np.sin(az), np.cos(az), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    assert up[2] > 0.0
    scale = np.asarray(box_aspect, dtype=np.float64)
    return view, right, up, scale


def _project(pts_xyz: np.ndarray, right: np.ndarray, up: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Project cube coordinates [-1, 1]^3 into orthographic screen plane (sx, sy)."""
    pts_s = np.atleast_2d(pts_xyz) * scale[None, :]
    return np.column_stack([pts_s @ right, pts_s @ up])


def _render_volume_rgba(
    cube: dict[str, object],
    view: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    scale: np.ndarray,
    bounds: tuple[float, float, float, float],
    width_px: int = 1120,
    height_px: int = 780,
    depth_steps: int = 112,
) -> np.ndarray:
    """Direct Beer-Lambert 3D ray integration blending planar red (XY) and score purple (Z)."""
    rho = np.asarray(cube["rho_3d"], dtype=np.float32)
    m_xy = np.asarray(cube["m_xy_3d"], dtype=np.float32)
    m_z = np.asarray(cube["m_z_3d"], dtype=np.float32)

    r = rho / max(float(rho.max()), 1e-12)
    grad_x, grad_y, grad_z = np.gradient(rho)
    normal = -np.stack([grad_x, grad_y, grad_z], axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=3, keepdims=True), np.float32(1e-10))

    light = np.array([-0.42, -0.48, 0.77], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.maximum(normal @ light, 0.0)
    half = light + view.astype(np.float32)
    half /= np.linalg.norm(half)
    specular = np.maximum(normal @ half, 0.0) ** 28

    c_red = style.PLANAR_RED_CMAP(np.clip(0.22 + 0.78 * m_xy, 0.0, 1.0))[..., :3].astype(np.float32)
    c_purp = style.SCORE_PURPLE_CMAP(np.clip(0.22 + 0.78 * m_z, 0.0, 1.0))[..., :3].astype(np.float32)
    w_z = np.clip(0.50 + 1.35 * (m_z - 0.85 * m_xy), 0.0, 1.0)[..., None]
    color = (1.0 - w_z) * c_red + w_z * c_purp

    color *= (0.78 + 0.22 * diffuse)[..., None]
    color += (1.0 - color) * (0.14 * specular)[..., None]

    activation = np.clip((r - 0.065) / 0.17, 0.0, 1.0)
    activation = activation * activation * (3.0 - 2.0 * activation)
    boundary_band = np.exp(-0.5 * ((r - 0.105) / 0.032) ** 2)
    outside_fade = np.clip(r / 0.065, 0.0, 1.0)
    outside_fade = outside_fade * outside_fade * (3.0 - 2.0 * outside_fade)
    extinction = 3.5 * (r**0.72) * activation + 0.45 * boundary_band * outside_fade
    color += (1.0 - color) * (0.04 * boundary_band)[..., None]

    field = np.concatenate([np.clip(color, 0.0, 1.0), extinction[..., None]], axis=-1)
    volume = torch.tensor(field.transpose(3, 0, 1, 2)[None].copy(), dtype=torch.float32)

    vx_min, vx_max, vy_min, vy_max = bounds
    extent_d = float(np.linalg.norm(scale))
    gx = torch.linspace(vx_min, vx_max, width_px, dtype=torch.float32)
    gy = torch.linspace(vy_max, vy_min, height_px, dtype=torch.float32)
    depths = torch.linspace(extent_d, -extent_d, depth_steps, dtype=torch.float32)
    step = float(2.0 * extent_d / (depth_steps - 1))

    right_t = torch.tensor(right, dtype=torch.float32)
    up_t = torch.tensor(up, dtype=torch.float32)
    view_t = torch.tensor(view, dtype=torch.float32)
    inv_scale_t = torch.tensor(1.0 / scale, dtype=torch.float32)

    # Verify exact ray-sample back-projection alignment against orthographic helper.
    test_cube = np.array([0.0, 0.0, 0.84], dtype=np.float64)
    test_s = test_cube * scale
    test_sx = float(test_s @ right)
    test_sy = float(test_s @ up)
    test_d = float(test_s @ view)
    reconstructed = (test_sx * right + test_sy * up + test_d * view) / scale
    assert np.allclose(reconstructed, test_cube, atol=1e-10), "Camera ray alignment check failed"

    image = np.zeros((height_px, width_px, 4), dtype=np.float32)
    with torch.inference_mode():
        for y0 in range(0, height_px, 130):
            y1 = min(y0 + 130, height_px)
            rows = y1 - y0
            plane = gx[None, :, None] * right_t + gy[y0:y1, None, None] * up_t
            transmission = torch.ones((rows, width_px), dtype=torch.float32)
            radiance = torch.zeros((3, rows, width_px), dtype=torch.float32)
            optical_depth = torch.zeros((rows, width_px), dtype=torch.float32)
            for start in range(0, depth_steps, 16):
                d = depths[start : start + 16]
                world = plane[None] + d[:, None, None, None] * view_t
                norm_xyz = world * inv_scale_t
                grid = norm_xyz[..., [2, 1, 0]][None]
                sample = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                tau = sample[3] * step
                optical_depth += tau.sum(dim=0)
                alpha = -torch.expm1(-tau)
                cumulative = torch.exp(-torch.cumsum(tau, dim=0))
                before = torch.cat([torch.ones_like(cumulative[:1]), cumulative[:-1]], dim=0)
                contribution = torch.sum(sample[:3] * (alpha * before)[None], dim=1)
                radiance += transmission[None] * contribution
                transmission *= cumulative[-1]
            alpha = -torch.expm1(-optical_depth)
            rgb = (radiance / alpha.clamp_min(1e-8)[None]).permute(1, 2, 0)
            image[y0:y1, :, :3] = rgb.numpy()
            image[y0:y1, :, 3] = alpha.numpy()

    return np.clip(image, 0.0, 1.0)


def build_figure(run_dir: Path | str = loader.DEFAULT_RUN_DIR) -> dict[str, object]:
    """Render Fig2_density_cube cleanly on a 180 x 120 mm Nature canvas."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    bundle = loader.load_run_bundle(run_dir=run_dir)
    cube = loader.compute_density_cube_data(bundle)

    view, right, up, scale = _camera_basis(elev_deg=25.0, azim_deg=-54.0, box_aspect=(1.12, 1.00, 0.88))
    corners_xyz = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corners_2d = _project(corners_xyz, right, up, scale)
    vx_min = float(corners_2d[:, 0].min()) - 0.02
    vx_max = float(corners_2d[:, 0].max()) + 0.02
    vy_min = float(corners_2d[:, 1].min()) - 0.02
    vy_max = float(corners_2d[:, 1].max()) + 0.02

    vol_rgba = _render_volume_rgba(cube, view, right, up, scale, (vx_min, vx_max, vy_min, vy_max))

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor="white")
    fig.text(
        0.045,
        0.950,
        "3D molecular probability density across SOM structure and score",
        fontsize=11.0,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.045,
        0.915,
        "Transparent space cube: gradient red tracks SOM planar movement (XY), gradient purple tracks 1D score ascent (Z)",
        fontsize=7.5,
        color=style.MUTED,
    )

    ax = fig.add_axes([0.04, 0.145, 0.92, 0.74])
    ax.patch.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. Frosted transparent back & floor glass faces
    back_faces = [
        np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=np.float64),  # floor z=-1
        np.array([[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64),      # back-right wall y=+1
        np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]], dtype=np.float64),  # back-left wall x=-1
    ]
    for idx_f, quad in enumerate(back_faces):
        q2d = _project(quad, right, up, scale)
        ax.add_patch(
            Polygon(
                q2d,
                closed=True,
                facecolor="#F8F5FA" if idx_f == 0 else "#F7F6FA",
                edgecolor="none",
                alpha=0.38 if idx_f == 0 else 0.28,
                zorder=1,
            )
        )

    # 2. Marginal density maps on floor (z=-1: gradient red) & back wall (y=+1: gradient purple)
    F1, F2 = cube["F1"], cube["F2"]
    floor_3d = np.column_stack([F1.ravel(), F2.ravel(), np.full(F1.size, -1.0)])
    floor_2d = _project(floor_3d, right, up, scale)
    SX_f = floor_2d[:, 0].reshape(F1.shape)
    SY_f = floor_2d[:, 1].reshape(F1.shape)

    rho_xy = cube["rho_xy"] / max(float(cube["rho_xy"].max()), 1e-12)
    red_val = np.clip(0.18 * rho_xy + 0.82 * cube["move_xy_floor"], 0.0, 1.0)
    rgba_floor = style.PLANAR_RED_CMAP(red_val).copy()
    rgba_floor[..., 3] = np.clip((rho_xy**0.85) * 0.72, 0.0, 0.72)
    rgba_floor_cell = 0.25 * (
        rgba_floor[:-1, :-1] + rgba_floor[1:, :-1] + rgba_floor[:-1, 1:] + rgba_floor[1:, 1:]
    )
    ax.pcolormesh(SX_f, SY_f, rgba_floor_cell, shading="flat", edgecolors="none", rasterized=True, zorder=2)

    wall_3d = np.column_stack([F1.ravel(), np.full(F1.size, 1.0), F2.ravel()])
    wall_2d = _project(wall_3d, right, up, scale)
    SX_w = wall_2d[:, 0].reshape(F1.shape)
    SY_w = wall_2d[:, 1].reshape(F1.shape)

    rho_xz = cube["rho_xz"] / max(float(cube["rho_xz"].max()), 1e-12)
    purp_val = np.clip(0.18 * rho_xz + 0.82 * cube["move_z_wall"], 0.0, 1.0)
    rgba_wall = style.SCORE_PURPLE_CMAP(purp_val).copy()
    rgba_wall[..., 3] = np.clip((rho_xz**0.85) * 0.56, 0.0, 0.56)
    rgba_wall_cell = 0.25 * (
        rgba_wall[:-1, :-1] + rgba_wall[1:, :-1] + rgba_wall[:-1, 1:] + rgba_wall[1:, 1:]
    )
    ax.pcolormesh(SX_w, SY_w, rgba_wall_cell, shading="flat", edgecolors="none", rasterized=True, zorder=2)

    # 3. Pale gray coordinate grids on floor & back wall + layered 1D Z-score densities on x=-1 wall
    grid_vals = np.linspace(-1.0, 1.0, 9)
    for gv in grid_vals:
        p_f1 = _project(np.array([[-1.0, gv, -1.0], [1.0, gv, -1.0]]), right, up, scale)
        p_f2 = _project(np.array([[gv, -1.0, -1.0], [gv, 1.0, -1.0]]), right, up, scale)
        ax.plot(p_f1[:, 0], p_f1[:, 1], color=style.UNDEF_GRID, lw=0.25, alpha=0.45, zorder=3)
        ax.plot(p_f2[:, 0], p_f2[:, 1], color=style.UNDEF_GRID, lw=0.25, alpha=0.45, zorder=3)

        p_w1 = _project(np.array([[-1.0, 1.0, gv], [1.0, 1.0, gv]]), right, up, scale)
        p_w2 = _project(np.array([[gv, 1.0, -1.0], [gv, 1.0, 1.0]]), right, up, scale)
        ax.plot(p_w1[:, 0], p_w1[:, 1], color=style.UNDEF_GRID, lw=0.23, alpha=0.35, zorder=3)
        ax.plot(p_w2[:, 0], p_w2[:, 1], color=style.UNDEF_GRID, lw=0.23, alpha=0.35, zorder=3)

    zs_1d = cube["zs_1d"]
    rho_z_gen = cube["rho_z_gen"]
    for g in range(7):
        y_base = 0.94 - g * (1.42 / 6.0)
        y_peak = y_base - 0.36 * rho_z_gen[g]
        curve_pts = np.column_stack([np.full(len(zs_1d), -1.0), y_peak, zs_1d])
        base_pts = np.column_stack([np.full(len(zs_1d), -1.0), np.full(len(zs_1d), y_base), zs_1d[::-1]])
        poly_2d = _project(np.vstack([curve_pts, base_pts]), right, up, scale)
        col_g = style.SCORE_PURPLE_CMAP(0.22 + 0.78 * (g / 6.0))
        ax.add_patch(Polygon(poly_2d, closed=True, facecolor=col_g, edgecolor=col_g, lw=0.40, alpha=0.42, zorder=3))

    # Floor planar centroid drift trajectory (gradient red, zero arrows)
    floor_pts_2d = _project(cube["centroid_floor_xyz"], right, up, scale)
    segs_f = np.stack([floor_pts_2d[:-1], floor_pts_2d[1:]], axis=1)
    lc_f = LineCollection(segs_f, colors=style.PLANAR_RED_CMAP(np.linspace(0.25, 0.95, len(segs_f))), linewidths=1.15, alpha=0.88, zorder=4)
    ax.add_collection(lc_f)

    # 4. Hidden back wireframe edges converging on (-1, 1, -1)
    back_edges = [
        (np.array([-1, 1, -1]), np.array([1, 1, -1])),
        (np.array([-1, 1, -1]), np.array([-1, -1, -1])),
        (np.array([-1, 1, -1]), np.array([-1, 1, 1])),
    ]
    for a, b in back_edges:
        e2d = _project(np.vstack([a, b]), right, up, scale)
        ax.plot(e2d[:, 0], e2d[:, 1], color="#BDB3C7", lw=0.45, ls="--", alpha=0.60, zorder=4)

    # 5. Direct Beer-Lambert 3D volumetric density cloud
    ax.imshow(
        vol_rgba,
        extent=[vx_min, vx_max, vy_min, vy_max],
        origin="upper",
        interpolation="bilinear",
        zorder=5,
    )

    # 6. 3D Population centroid trajectory inside the density cloud (red -> purple gradient, zero arrows)
    spine_2d = _project(cube["centroid_spline_xyz"], right, up, scale)
    segs_s = np.stack([spine_2d[:-1], spine_2d[1:]], axis=1)
    t_cols = np.linspace(0.0, 1.0, len(segs_s))[:, None]
    s_red = style.PLANAR_RED_CMAP(np.full(len(segs_s), 0.78))[:, :3]
    s_purp = style.SCORE_PURPLE_CMAP(np.full(len(segs_s), 0.88))[:, :3]
    spine_cols = (1.0 - t_cols) * s_red + t_cols * s_purp
    ax.add_collection(LineCollection(segs_s, colors=spine_cols, linewidths=1.45, alpha=0.92, zorder=6))

    c_start_2d = _project(cube["centroids_xyz"][0], right, up, scale)[0]
    c_end_2d = _project(cube["centroids_xyz"][-1], right, up, scale)[0]
    c_floor_end = _project(cube["centroid_floor_xyz"][-1], right, up, scale)[0]
    ax.plot([c_end_2d[0], c_floor_end[0]], [c_end_2d[1], c_floor_end[1]], color="#766083", lw=0.60, ls=":", alpha=0.70, zorder=6)
    ax.scatter(c_start_2d[0], c_start_2d[1], s=40, color="#BA4B73", edgecolor="white", linewidth=0.85, zorder=8)
    ax.scatter(c_end_2d[0], c_end_2d[1], s=110, marker="*", color="#51348A", edgecolor="white", linewidth=0.80, zorder=8)

    # 7. Front transparent cube glass wireframe (9 outer/front edges)
    front_edges = [
        ([-1, -1, -1], [1, -1, -1]),
        ([1, -1, -1], [1, 1, -1]),
        ([1, -1, -1], [1, -1, 1]),
        ([-1, -1, 1], [1, -1, 1]),
        ([1, -1, 1], [1, 1, 1]),
        ([1, 1, 1], [-1, 1, 1]),
        ([-1, 1, 1], [-1, -1, 1]),
        ([-1, -1, -1], [-1, -1, 1]),
        ([1, 1, -1], [1, 1, 1]),
    ]
    for a, b in front_edges:
        e2d = _project(np.vstack([a, b]), right, up, scale)
        ax.plot(e2d[:, 0], e2d[:, 1], color="#7C6D8A", lw=0.68, alpha=0.78, zorder=7)

    # Wide clean margins around volumetric cube extent so NO label touches vol_rgba bbox or strokes
    ax.set_xlim(vx_min - 0.85, vx_max + 0.85)
    ax.set_ylim(vy_min - 0.34, vy_max + 0.30)

    # Axis labels cleanly outside the cube edges
    p_som1 = _project(np.array([0.0, -1.0, -1.0]), right, up, scale)[0]
    p_som2 = _project(np.array([1.0, 0.0, -1.0]), right, up, scale)[0]
    p_zax = _project(np.array([-1.0, -1.0, 1.0]), right, up, scale)[0]
    ax.text(p_som1[0] - 0.25, p_som1[1] - 0.22, "SOM Dim 1 (structure)", ha="right", va="top", fontsize=6.6, color="#5C4F6B", zorder=10)
    ax.text(p_som2[0] + 0.24, p_som2[1] - 0.22, "SOM Dim 2 (structure)", ha="left", va="top", fontsize=6.6, color="#5C4F6B", zorder=10)
    ax.text(p_zax[0] - 0.08, p_zax[1] + 0.06, "1D Score Axis (PC1)", ha="right", va="bottom", fontsize=6.6, color="#51348A", fontweight="bold", zorder=10)

    # Margin callouts strictly outside volumetric extent [vx_min..vx_max, vy_min..vy_max]
    fig.canvas.draw()
    z_wall_pt = _project(np.array([0.0, 1.0, 0.50]), right, up, scale)[0]
    floor_pt = _project(np.array([0.42, -0.30, -1.0]), right, up, scale)[0]

    margin_callouts = [
        ("G6 high-score core (z = +0.79)", c_end_2d, (vx_max + 0.10, vy_max - 0.12), "left", "#51348A", True),
        ("Z-axis score density (PC1)", z_wall_pt, (vx_max + 0.10, z_wall_pt[1] + 0.02), "left", "#6E52A3", False),
        ("SOM planar movement (red)", floor_pt, (vx_max + 0.12, vy_min + 0.22), "left", "#A13058", False),
        ("G0 bootstrap base (z = -1.09)", c_start_2d, (vx_min - 0.12, vy_min + 0.56), "right", "#BA4B73", False),
    ]
    for label, target_xy, (lx, ly), ha, col, is_bold in margin_callouts:
        dx = lx - target_xy[0]
        dy = ly - target_xy[1]
        dist = max(float(np.hypot(dx, dy)), 1e-6)
        sx = target_xy[0] + dx / dist * 0.05
        sy = target_xy[1] + dy / dist * 0.05
        ex = lx - 0.03 if ha == "left" else lx + 0.03
        ax.add_artist(mlines.Line2D([sx, ex], [sy, ly], color=col, lw=0.55, alpha=0.72, zorder=9))
        ax.text(lx, ly, label, ha=ha, va="center", fontsize=6.7, fontweight="bold" if is_bold else "normal", color=col, zorder=10)

    # Dual horizontal colorbars at bottom (planar red vs score purple)
    cax_red = fig.add_axes([0.10, 0.095, 0.34, 0.012])
    cb_red = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=style.PLANAR_RED_CMAP),
        cax=cax_red,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    cb_red.ax.set_xticklabels(["Low", "SOM planar movement (XY)", "High"])
    cb_red.ax.tick_params(labelsize=5.8, length=1.5, pad=2, colors=style.INK)
    cb_red.outline.set_visible(False)

    cax_purp = fig.add_axes([0.56, 0.095, 0.34, 0.012])
    cb_purp = fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap=style.SCORE_PURPLE_CMAP),
        cax=cax_purp,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    cb_purp.ax.set_xticklabels(["G0 base", "Z-axis score ascent (PC1)", "G6 high"])
    cb_purp.ax.tick_params(labelsize=5.8, length=1.5, pad=2, colors=style.INK)
    cb_purp.outline.set_visible(False)

    var_pct = float(cube["pc1_var_ratio"]) * 100.0
    g0z = float(cube["g0_z_mean"])
    g6z = float(cube["g6_z_mean"])
    fig.text(
        0.045,
        0.032,
        (
            f"Real 3D KDE density volume (live_large_pop_v1, n = {len(cube['gens'])}, G0-G6): "
            f"XY = frozen SOM 2D structure ({cube['occupied_g0']} to {cube['occupied_g6']} cells); "
            f"Z = 1D score PC1 ({var_pct:.1f}% var, {g0z:+.2f} to {g6z:+.2f})."
        ),
        fontsize=5.8,
        color="#7E6E88",
    )

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["a"],
        exclude_axes=[cax_red, cax_purp],
        require_panel_labels=False,
        strict=True,
    )

    svg_path = OUT_DIR / "Fig2_density_cube.svg"
    pdf_path = OUT_DIR / "Fig2_density_cube.pdf"
    png_path = OUT_DIR / "Fig2_density_cube.png"
    tiff_path = OUT_DIR / "Fig2_density_cube.tiff"

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")
    plt.close(fig)

    cube["mol_table"].to_csv(SRC_DIR / "Fig2_density_cube_molecules.csv", index=False)
    np.savez_compressed(
        SRC_DIR / "Fig2_density_cube_arrays.npz",
        grid_coords=cube["grid_coords"],
        rho_3d=cube["rho_3d"],
        m_xy_3d=cube["m_xy_3d"],
        m_z_3d=cube["m_z_3d"],
        rho_xy=cube["rho_xy"],
        move_xy_floor=cube["move_xy_floor"],
        rho_xz=cube["rho_xz"],
        move_z_wall=cube["move_z_wall"],
        centroids_xyz=cube["centroids_xyz"],
        centroid_spline_xyz=cube["centroid_spline_xyz"],
    )

    contract = {
        "figure": "Fig2_density_cube",
        "width_mm": 180.0,
        "height_mm": 120.0,
        "run_dir": str(run_dir),
        "n_molecules": int(len(cube["gens"])),
        "occupied_cells": [cube["occupied_g0"], cube["occupied_g6"]],
        "score_pc1_var_ratio": float(cube["pc1_var_ratio"]),
        "score_pc1_loadings": cube["pc1_loadings"].tolist(),
        "z_pc1_gen_means": [float(cube["z_pc1"][cube["gens"] == g].mean()) for g in range(7)],
        "planar_movement_mean": float(cube["m_xy"].mean()),
        "z_ascent_mean": float(cube["m_z"].mean()),
        "arrows": False,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2_density_cube.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = build_figure()
    print(
        json.dumps(
            {
                "n_molecules": summary["n_molecules"],
                "occupied_cells": summary["occupied_cells"],
                "score_pc1_var_ratio": summary["score_pc1_var_ratio"],
                "z_pc1_gen_means": [round(x, 3) for x in summary["z_pc1_gen_means"]],
            },
            indent=2,
        ),
        flush=True,
    )
