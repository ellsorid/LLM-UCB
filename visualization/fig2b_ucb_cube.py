"""Translucent glass-shell 3D evolution cube + 3-wall orthographic projections across Whole, Local, and Score."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
NATURE_FIG_SCRIPTS = Path(r"D:\skills\nature-skills\skills\nature-figure\scripts")
for _p in (str(THIS_DIR), str(NATURE_FIG_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from audit_panel_alignment import require_matplotlib_panel_alignment
import loader
import style

OUT_DIR = THIS_DIR / "output" / "fig2_ucb"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUT_DIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)


def _shade_oriented_face(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    values: np.ndarray,
    norm: TwoSlopeNorm,
    outward_dir: np.ndarray,
    alpha: float = 0.50,
    specular_strength: float = 0.14,
) -> np.ndarray:
    """Apply diffuse + glass specular satin lighting aligned to each face outward normal."""
    xyz = np.stack([X, Y, Z], axis=-1)
    normal = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    if float(np.nanmean(normal @ outward_dir)) < 0.0:
        normal = -normal
    normal = normal / np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-10)

    light = np.array([-0.4, -0.5, 0.77], dtype=np.float64)
    light = light / np.linalg.norm(light)
    diffuse = np.maximum(normal @ light, 0.0)

    half = light + np.array([0.4, -0.6, 0.7], dtype=np.float64)
    half = half / np.linalg.norm(half)
    specular = np.maximum(normal @ half, 0.0) ** 28

    rgba = style.MIGRATION_DIV_CMAP(norm(values)).copy()
    rgba[..., :3] *= (0.83 + 0.17 * diffuse)[..., None]
    rgba[..., :3] += (1.0 - rgba[..., :3]) * (specular_strength * specular)[..., None]
    rgba[..., 3] = alpha
    return np.clip(rgba, 0.0, 1.0)


def build_figure(run_dir: Path | str = loader.DEFAULT_RUN_DIR) -> dict[str, object]:
    """Render Fig2b_ucb_cube cleanly with Option-2 glass perspective on a 180 x 120 mm Nature canvas."""
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
    cube = loader.compute_ucb_cube_data(bundle)
    norm = TwoSlopeNorm(vmin=0.20, vcenter=0.52, vmax=0.85)

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor="white")
    fig.text(
        0.045,
        0.950,
        "3D molecular evolution cube driven by axis modification probability",
        fontsize=11.0,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.045,
        0.915,
        "Translucent glass perspective + 3-wall projections across Whole structure (X), Local structure (Y), and Score (Z)",
        fontsize=7.1,
        color=style.MUTED,
    )

    ax = fig.add_axes([-0.02, 0.19, 1.04, 0.69], projection="3d", computed_zorder=False)
    ax.patch.set_alpha(0.0)

    # 1. Outer transparent [0, 1]^3 back glass walls, 6x6 grids, and back dashed edges
    back_quads = [
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)],
        [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)],
    ]
    ax.add_collection3d(
        Poly3DCollection(back_quads, facecolors=["#F7F4FA", "#F8F6FB", "#F6F4FA"], edgecolors="none", alpha=0.22, zorder=0.0),
        autolim=False,
    )

    grid_lines = []
    for gv in np.linspace(0.0, 1.0, 6):
        grid_lines.append(np.array([[0.0, gv, 0.0], [1.0, gv, 0.0]]))
        grid_lines.append(np.array([[gv, 0.0, 0.0], [gv, 1.0, 0.0]]))
        grid_lines.append(np.array([[0.0, 1.0, gv], [1.0, 1.0, gv]]))
        grid_lines.append(np.array([[gv, 1.0, 0.0], [gv, 1.0, 1.0]]))
        grid_lines.append(np.array([[0.0, 0.0, gv], [0.0, 1.0, gv]]))
        grid_lines.append(np.array([[0.0, gv, 0.0], [0.0, gv, 1.0]]))
    ax.add_collection3d(
        Line3DCollection(grid_lines, colors="#C2B9CC", linewidths=0.24, alpha=0.36, zorder=0.2),
        autolim=False,
    )
    ax.add_collection3d(
        Line3DCollection(cube["outer_back_edges"], colors="#B5ABC2", linewidths=0.48, ls="--", alpha=0.58, zorder=0.4),
        autolim=False,
    )

    # 2. 3-Wall orthographic deformation footprints + G0 baseline rectangles + G1->G6 probability curves
    w_proj = cube["wall_projections"]
    wall_styles = [
        ("floor_xy", "#C87696", "#A14369"),
        ("wall_xz", "#846DB8", "#63489D"),
        ("wall_yz", "#9E7EA8", "#7A558E"),
    ]
    for wname, fill_col, rim_col in wall_styles:
        loop_pts = w_proj[wname]
        poly_w = Poly3DCollection([loop_pts], facecolors=fill_col, edgecolors="none", alpha=0.16, zorder=0.6)
        ax.add_collection3d(poly_w, autolim=False)
        ax.plot(loop_pts[:, 0], loop_pts[:, 1], loop_pts[:, 2], color=rim_col, lw=0.68, alpha=0.68, zorder=0.7)

    ax.add_collection3d(
        Line3DCollection(cube["g0_wall_rects"], colors="#8F84A3", linewidths=0.48, ls=":", alpha=0.60, zorder=0.75),
        autolim=False,
    )

    floor_tr = cube["trend_floor"]
    wall_tr = cube["trend_wall"]
    left_tr = cube["trend_left_wall"]
    ax.plot(floor_tr[:, 0], floor_tr[:, 1], floor_tr[:, 2], color="#A14369", lw=0.95, alpha=0.72, zorder=0.85)
    ax.scatter(
        cube["gen_probs"][:, 0],
        cube["gen_probs"][:, 1],
        np.zeros(len(cube["gen_probs"])),
        s=10,
        color="#A14369",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=0.9,
    )
    ax.plot(wall_tr[:, 0], wall_tr[:, 1], wall_tr[:, 2], color="#63489D", lw=0.95, alpha=0.72, zorder=0.85)
    ax.scatter(
        cube["gen_probs"][:, 0],
        np.ones(len(cube["gen_probs"])),
        cube["gen_probs"][:, 2],
        s=10,
        color="#63489D",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=0.9,
    )
    ax.plot(left_tr[:, 0], left_tr[:, 1], left_tr[:, 2], color="#7A558E", lw=0.95, alpha=0.68, zorder=0.85)
    ax.scatter(
        np.zeros(len(cube["gen_probs"])),
        cube["gen_probs"][:, 1],
        cube["gen_probs"][:, 2],
        s=10,
        color="#7A558E",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.80,
        zorder=0.9,
    )

    # 3. Inner dashed pale-gray baseline G0 molecular cube
    ax.add_collection3d(
        Line3DCollection(cube["baseline_g0_edges"], colors="#9286A6", linewidths=0.58, ls="--", alpha=0.65, zorder=1.1),
        autolim=False,
    )

    # 4. Hidden/back 3 curved faces of deformed molecular cube (translucent pale tint + dashed curved mesh)
    back_faces = cube["back_faces"]
    for bname, z_ord in [("back_bottom", 1.35), ("back_x", 1.45), ("back_y", 1.55)]:
        fdict = back_faces[bname]
        Xf, Yf, Zf, Cf = fdict["X"], fdict["Y"], fdict["Z"], fdict["C"]
        nv_f, nu_f = Xf.shape
        rgba_b = _shade_oriented_face(Xf, Yf, Zf, Cf, norm, fdict["outward"], alpha=0.22, specular_strength=0.06)
        ax.plot_surface(
            Xf,
            Yf,
            Zf,
            facecolors=rgba_b,
            shade=False,
            linewidth=0.0,
            edgecolor="none",
            antialiased=False,
            rcount=nv_f,
            ccount=nu_f,
            rasterized=True,
            zorder=z_ord,
        )
        back_grid = []
        for r in np.linspace(0, nv_f - 1, 7).astype(int):
            back_grid.append(np.column_stack([Xf[r], Yf[r], Zf[r]]))
        for c in np.linspace(0, nu_f - 1, 7).astype(int):
            back_grid.append(np.column_stack([Xf[:, c], Yf[:, c], Zf[:, c]]))
        ax.add_collection3d(
            Line3DCollection(back_grid, colors="#86739C", linewidths=0.22, ls="--", alpha=0.36, zorder=z_ord + 0.05),
            autolim=False,
        )
    ax.add_collection3d(
        Line3DCollection(cube["deformed_back_edges"], colors="#76628C", linewidths=0.54, ls="--", alpha=0.56, zorder=1.75),
        autolim=False,
    )

    # 5. Internal 3D G1->G6 probability evolution spine + mid-height equatorial contour loop (visible through glass)
    mid_ring = cube["mid_ring"]
    ax.plot(mid_ring[:, 0], mid_ring[:, 1], mid_ring[:, 2], color="#FFFFFF", lw=0.60, alpha=0.52, zorder=2.0)

    t3 = cube["trend_3d"]
    ax.plot(t3[:, 0], t3[:, 1], t3[:, 2], color="#8F1F49", lw=1.40, alpha=0.90, zorder=2.2)
    ax.scatter(t3[0, 0], t3[0, 1], t3[0, 2], s=26, color="#6D509D", edgecolor="white", linewidth=0.75, zorder=2.3)
    ax.scatter(t3[-1, 0], t3[-1, 1], t3[-1, 2], s=44, marker="*", color="#8F1F49", edgecolor="white", linewidth=0.65, zorder=2.3)

    # 6. Front-facing 3 translucent tinted glass faces of deformed molecular cube
    faces = cube["faces"]
    for fname, z_ord, f_alpha in [("front_y", 2.7, 0.48), ("front_x", 2.9, 0.50), ("top", 3.1, 0.54)]:
        fdict = faces[fname]
        Xf, Yf, Zf, Cf = fdict["X"], fdict["Y"], fdict["Z"], fdict["C"]
        nv_f, nu_f = Xf.shape
        rgba = _shade_oriented_face(Xf, Yf, Zf, Cf, norm, fdict["outward"], alpha=f_alpha, specular_strength=0.14)
        ax.plot_surface(
            Xf,
            Yf,
            Zf,
            facecolors=rgba,
            shade=False,
            linewidth=0.0,
            edgecolor="none",
            antialiased=False,
            rcount=nv_f,
            ccount=nu_f,
            rasterized=True,
            zorder=z_ord,
        )
        face_lines = []
        for r in np.linspace(0, nv_f - 1, 9).astype(int):
            face_lines.append(np.column_stack([Xf[r], Yf[r], Zf[r]]))
        for c in np.linspace(0, nu_f - 1, 9).astype(int):
            face_lines.append(np.column_stack([Xf[:, c], Yf[:, c], Zf[:, c]]))
        ax.add_collection3d(
            Line3DCollection(face_lines, colors="#66517A", linewidths=0.23, alpha=0.38, zorder=z_ord + 0.5),
            autolim=False,
        )

    ax.add_collection3d(
        Line3DCollection(cube["deformed_edges"], colors="#533E68", linewidths=0.85, alpha=0.88, zorder=4.0),
        autolim=False,
    )

    # 7. Outer transparent [0, 1]^3 glass wireframe framing the entire scene
    ax.add_collection3d(
        Line3DCollection(cube["outer_mid_edges"], colors="#847696", linewidths=0.78, alpha=0.82, zorder=5.0),
        autolim=False,
    )
    ax.add_collection3d(
        Line3DCollection(cube["outer_front_edges"], colors="#8E80A0", linewidths=0.62, alpha=0.55, zorder=6.0),
        autolim=False,
    )

    lm = cube["landmarks"]
    ax.scatter(*lm["g0_ref"], s=28, marker="o", color="#7B6B91", edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter(*lm["g6_peak"], s=84, marker="*", color="#8F1F49", edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter(*lm["wall_g6"], s=26, marker="o", color="#593D94", edgecolor="white", linewidth=0.75, zorder=7)

    ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), zlim=(-0.04, 1.04))
    ax.view_init(elev=25, azim=-54)
    ax.set_box_aspect((1.05, 1.05, 0.96), zoom=1.08)
    ax.axis("off")
    for coll in ax.collections:
        coll.set_clip_on(False)

    cax = fig.add_axes([0.26, 0.095, 0.48, 0.012])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=style.MIGRATION_DIV_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[0.20, 0.52, 0.85],
    )
    cb.ax.set_xticklabels(["Low prob. (0.20)", "Mid (0.52)", "High prob. (0.85)"])
    cb.ax.tick_params(labelsize=5.8, length=1.5, pad=2, colors=style.INK)
    cb.outline.set_visible(False)

    fig.canvas.draw()
    proj_m = ax.get_proj()

    def _to_fig(pt3: np.ndarray) -> tuple[float, float]:
        x2, y2, _ = proj3d.proj_transform(float(pt3[0]), float(pt3[1]), float(pt3[2]), proj_m)
        fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x2, y2)))
        return float(fx), float(fy)

    # 8. External axis titles cleanly outside the 3 outer [0, 1]^3 edges (0 fail, 0 warn)
    px_mid = _to_fig(np.array([0.45, 0.0, 0.0]))
    py_mid = _to_fig(np.array([1.0, 0.50, 0.0]))
    pz_top = _to_fig(np.array([0.0, 0.0, 0.88]))
    fig.text(px_mid[0] - 0.032, px_mid[1] - 0.028, "Whole structure (P_whole)", ha="right", va="top", fontsize=6.5, color="#5C4F6B", zorder=10)
    fig.text(py_mid[0] + 0.032, py_mid[1] - 0.028, "Local structure (P_local)", ha="left", va="top", fontsize=6.5, color="#5C4F6B", zorder=10)
    fig.text(pz_top[0] - 0.024, pz_top[1] + 0.012, "Score (P_score)", ha="right", va="center", fontsize=6.6, fontweight="bold", color="#8F1F49", zorder=10)

    # 9. White-margin arrowless callouts
    g6p = cube["g6_probs"]
    callouts = [
        (
            lm["g6_peak"],
            f"Translucent evolved cube\n(G6: P_w={g6p[0]:.2f}, P_l={g6p[1]:.2f}, P_s={g6p[2]:.2f})",
            (0.750, 0.785),
            "left",
            "#8F1F49",
            True,
        ),
        (
            lm["wall_g6"],
            "3-wall projections &\nG1->G6 score prob. (0.30 -> 0.75)",
            (0.755, 0.495),
            "left",
            "#593D94",
            False,
        ),
        (
            lm["g0_ref"],
            "Hidden back mesh &\nG0 baseline (via glass)",
            (0.235, 0.525),
            "right",
            "#6E5D7E",
            False,
        ),
    ]
    for pt, text, (tx, ty), ha, col, is_bold in callouts:
        fx0, fy0 = _to_fig(pt)
        lx_end = tx - 0.008 if ha == "left" else tx + 0.008
        dx, dy = lx_end - fx0, ty - fy0
        dist = max(float(np.hypot(dx, dy)), 1e-6)
        sx = fx0 + dx / dist * 0.009
        sy = fy0 + dy / dist * 0.009
        fig.add_artist(
            mlines.Line2D(
                [sx, lx_end],
                [sy, ty],
                transform=fig.transFigure,
                color=col,
                linewidth=0.55,
                alpha=0.75,
                zorder=9,
            )
        )
        fig.text(
            tx,
            ty,
            text,
            ha=ha,
            va="center",
            fontsize=6.5,
            fontweight="bold" if is_bold else "normal",
            color=col,
            zorder=10,
        )

    fig.text(0.06, 0.135, "Dashed inner mesh: occluded back faces & G0 base", fontsize=6.4, color="#8F859B")
    fig.text(0.50, 0.135, "3 walls: orthogonal shape & G1-G6 probability projections", fontsize=6.4, color="#6D5A82")

    mp = cube["mean_probs"]
    fig.text(
        0.045,
        0.032,
        (
            "Real G1-G6 transition probabilities (live_large_pop_v1, n = 120): "
            f"mean P_whole = {mp[0]:.2f}, P_local = {mp[1]:.2f}, "
            f"P_score = {cube['gen_probs'][0, 2]:.2f} to {g6p[2]:.2f}; Option-2 glass + 3-wall projections."
        ),
        fontsize=5.8,
        color="#7E6E88",
    )

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["b"],
        exclude_axes=[cax],
        require_panel_labels=False,
        strict=True,
    )

    svg_path = OUT_DIR / "Fig2b_ucb_cube.svg"
    pdf_path = OUT_DIR / "Fig2b_ucb_cube.pdf"
    png_path = OUT_DIR / "Fig2b_ucb_cube.png"
    tiff_path = OUT_DIR / "Fig2b_ucb_cube.tiff"

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")
    plt.close(fig)

    pd.DataFrame(cube["gen_table"]).to_csv(SRC_DIR / "Fig2b_ucb_cube_probs.csv", index=False)
    np.savez_compressed(
        SRC_DIR / "Fig2b_ucb_cube_arrays.npz",
        gen_probs=cube["gen_probs"],
        mean_probs=cube["mean_probs"],
        g6_probs=cube["g6_probs"],
        trend_3d=cube["trend_3d"],
        top_X=faces["top"]["X"],
        top_Y=faces["top"]["Y"],
        top_Z=faces["top"]["Z"],
        top_C=faces["top"]["C"],
    )

    contract = {
        "figure": "Fig2b_ucb_cube",
        "perspective_mode": "option2_glass_shell_plus_3wall_projections",
        "width_mm": 180.0,
        "height_mm": 120.0,
        "run_dir": str(run_dir),
        "axes": ["Whole structure (P_whole)", "Local structure (P_local)", "Score (P_score)"],
        "mean_probs": [round(float(x), 4) for x in cube["mean_probs"]],
        "g6_probs": [round(float(x), 4) for x in cube["g6_probs"]],
        "j_rel_min": float(cube["j_rel_min"]),
        "bounds_ok": bool(cube["bounds_ok"]),
        "gen_table": cube["gen_table"],
        "arrows": False,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2b_ucb_cube.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = build_figure()
    print(
        json.dumps(
            {
                "mean_probs": summary["mean_probs"],
                "g6_probs": summary["g6_probs"],
                "j_rel_min": round(summary["j_rel_min"], 4),
                "bounds_ok": summary["bounds_ok"],
            },
            indent=2,
        ),
        flush=True,
    )
