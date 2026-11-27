"""Bounded flat SOM grid deformed into a centered saddle by real gradient differentials."""
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

OUT_DIR = THIS_DIR / "output" / "fig2_saddle"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUT_DIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)


def build_figure(run_dir: Path | str = loader.DEFAULT_RUN_DIR) -> dict[str, object]:
    """Render Fig2_directed_saddle cleanly on a 180 x 120 mm Nature canvas."""
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
    saddle = loader.compute_directed_saddle_data(bundle)

    U, V = saddle["U"], saddle["V"]
    X, Y, Z = saddle["X"], saddle["Y"], saddle["Z"]
    nv, nu = U.shape
    limit = float(np.abs(Z).max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    rgba = style.shade_satin_diverging_surface(X, Y, Z, Z, norm, alpha=0.96)

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor="white")
    fig.text(
        0.045,
        0.950,
        "SOM-differential deformation around a directed molecule",
        fontsize=11.0,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.045,
        0.915,
        "Bounded flat SOM grid deformed by real neighbor differentials (G1-A4, Cell 86) into a centered saddle",
        fontsize=7.5,
        color=style.MUTED,
    )

    ax = fig.add_axes([-0.03, 0.13, 1.06, 0.75], projection="3d", computed_zorder=False)
    ax.patch.set_alpha(0.0)

    # Preserve exact material mesh topology between the flat initial plane and the deformed surface.
    flat_lines = []
    deformed_lines = []
    for r in np.linspace(0, nv - 1, 23).astype(int):
        flat_lines.append(np.column_stack([U[r], V[r], np.zeros(nu, dtype=np.float64)]))
        deformed_lines.append(np.column_stack([X[r], Y[r], Z[r]]))
    for c in np.linspace(0, nu - 1, 29).astype(int):
        flat_lines.append(np.column_stack([U[:, c], V[:, c], np.zeros(nv, dtype=np.float64)]))
        deformed_lines.append(np.column_stack([X[:, c], Y[:, c], Z[:, c]]))

    ax.add_collection3d(
        Line3DCollection(flat_lines, colors=style.UNDEF_GRID, linewidths=0.27, alpha=0.24, zorder=0),
        autolim=False,
    )
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=rgba,
        shade=False,
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        rcount=nv,
        ccount=nu,
        rasterized=True,
        zorder=2,
    )
    ax.add_collection3d(
        Line3DCollection(deformed_lines, colors=style.COOL_DARK, linewidths=0.23, alpha=0.39, zorder=3),
        autolim=False,
    )

    # Crosshair curves highlight opposing principal curvatures meeting at the centered molecule.
    for key in ("positive", "negative"):
        curve = saddle["saddle_curves"][key]
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="#A45C81", lw=0.80, alpha=0.70, zorder=5)

    point_xyz = saddle["point_xyz"]
    ref_colors = ["#9E486E", "#9478B3", "#8774B1", "#A798B8"]
    center_color = "#963F65"
    all_colors = ref_colors + [center_color]

    for j, pt in enumerate(point_xyz):
        is_center = j == len(point_xyz) - 1
        ax.scatter(
            pt[0],
            pt[1],
            pt[2],
            s=135 if is_center else 42,
            marker="*" if is_center else "o",
            color=all_colors[j],
            edgecolor="white",
            linewidth=0.85,
            zorder=8 if is_center else 7,
        )

    ax.set(
        xlim=(-2.65, 2.65),
        ylim=(-2.30, 2.30),
        zlim=(float(Z.min()) - 0.12, float(Z.max()) + 0.12),
    )
    ax.view_init(elev=25, azim=-54)
    ax.set_box_aspect((1.25, 1.0, 0.64), zoom=1.40)
    ax.axis("off")
    for coll in ax.collections:
        coll.set_clip_on(False)

    cax = fig.add_axes([0.30, 0.095, 0.40, 0.012])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=style.SADDLE_SATIN_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[-limit, 0.0, limit],
    )
    cb.ax.set_xticklabels(["Downward (\u0394s < 0)", "Saddle plane (Z = 0)", "Upward (\u0394s > 0)"])
    cb.ax.tick_params(labelsize=5.8, length=1.5, pad=2, colors=style.INK)
    cb.outline.set_visible(False)

    fig.canvas.draw()

    refs = saddle["refs"]
    # Place labels cleanly in clear white margin pockets outside the grid mesh, joined by thin hairlines.
    callout_specs = [
        (f"R1 (root, \u0394s = {refs[0]['ds']:+.2f})", (0.685, 0.740), "left", "#8B395E"),
        (f"R2 (\u0394s = {refs[1]['ds']:+.2f})", (0.755, 0.315), "left", "#6E5A8A"),
        (f"R3 (\u0394s = {refs[2]['ds']:+.2f})", (0.245, 0.285), "right", "#67558B"),
        (f"R4 (\u0394s = {refs[3]['ds']:+.2f})", (0.485, 0.215), "center", "#736386"),
        ("Directed molecule \u00b7 saddle", (0.475, 0.805), "center", "#8E355C"),
    ]
    for j, pt in enumerate(point_xyz):
        px2d, py2d, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], ax.get_proj())
        disp_xy = ax.transData.transform((px2d, py2d))
        fx0, fy0 = fig.transFigure.inverted().transform(disp_xy)
        text, (tx, ty), ha, txt_color = callout_specs[j]
        is_center = j == len(point_xyz) - 1

        # Compute hairline endpoints stopping slightly outside marker and just before label anchor
        lx_end = tx - 0.008 if ha == "left" else (tx + 0.008 if ha == "right" else tx)
        ly_end = ty + (0.013 if ty < fy0 else (-0.013 if ha == "center" else 0.0))
        dx, dy = lx_end - fx0, ly_end - fy0
        dist = max(float(np.hypot(dx, dy)), 1e-6)
        pad_start = 0.012 if is_center else 0.009
        sx = fx0 + dx / dist * pad_start
        sy = fy0 + dy / dist * pad_start

        fig.add_artist(
            mlines.Line2D(
                [sx, lx_end],
                [sy, ly_end],
                transform=fig.transFigure,
                color=txt_color,
                linewidth=0.60 if is_center else 0.50,
                alpha=0.72,
                zorder=9,
            )
        )
        fig.text(
            tx,
            ty,
            text,
            ha=ha,
            va="center",
            fontsize=7.0 if is_center else 6.8,
            fontweight="bold" if is_center else "normal",
            color=txt_color,
            zorder=10,
        )

    fig.text(0.08, 0.140, "Gray grid: initial bounded SOM plane (Z = 0)", fontsize=6.5, color="#8F859B")
    fig.text(0.56, 0.140, "Colored grid: SOM-differential deformed space", fontsize=6.5, color="#7F688D")

    lam0, lam1 = float(saddle["eigvals"][0]), float(saddle["eigvals"][1])
    min_j = float(saddle["min_jacobian"])
    fig.text(
        0.045,
        0.032,
        (
            "Real SOM projection (live_large_pop_v1, G1-A4, Cell 86): "
            f"top-4 gradient references yield measured saddle curvature \u03bb = ({lam0:+.2f}, {lam1:+.2f}) "
            f"and min J = {min_j:.2f}."
        ),
        fontsize=5.8,
        color="#7E6E88",
    )

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["a"],
        exclude_axes=[cax],
        require_panel_labels=False,
        strict=True,
    )

    svg_path = OUT_DIR / "Fig2_directed_saddle.svg"
    pdf_path = OUT_DIR / "Fig2_directed_saddle.pdf"
    png_path = OUT_DIR / "Fig2_directed_saddle.png"
    tiff_path = OUT_DIR / "Fig2_directed_saddle.tiff"

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")
    plt.close(fig)

    pts_df = pd.DataFrame(saddle["refs"] + [saddle["target"]])
    pts_df.to_csv(SRC_DIR / "Fig2_directed_saddle_points.csv", index=False)
    np.savez_compressed(
        SRC_DIR / "Fig2_directed_saddle_arrays.npz",
        U=U,
        V=V,
        X=X,
        Y=Y,
        Z=Z,
        J=saddle["J"],
        coordinates=saddle["coordinates"],
        point_xyz=point_xyz,
        H_real=saddle["H_real"],
        eigvals=saddle["eigvals"],
        hess_eigs=saddle["hess_eigs"],
        ds=saddle["ds"],
        wts=saddle["wts"],
        dr_raw=saddle["dr_raw"],
    )

    contract = {
        "figure": "Fig2_directed_saddle",
        "width_mm": 180.0,
        "height_mm": 120.0,
        "run_dir": str(run_dir),
        "target": saddle["target"],
        "references": saddle["refs"],
        "measured_H_eigs": saddle["eigvals"].tolist(),
        "origin_hessian_eigs": saddle["hess_eigs"].tolist(),
        "origin_gradient_norm": float(np.linalg.norm(saddle["gradient_at_origin"])),
        "minimum_jacobian": min_j,
        "compression_k": float(saddle["k_compression"]),
        "alpha_y": float(saddle["alpha_y"]),
        "sign_agreement": f"{saddle['sign_agree']}/{len(refs)}",
        "arrows": False,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2_directed_saddle.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = build_figure()
    print(
        json.dumps(
            {
                "measured_H_eigs": summary["measured_H_eigs"],
                "origin_hessian_eigs": summary["origin_hessian_eigs"],
                "origin_gradient_norm": summary["origin_gradient_norm"],
                "minimum_jacobian": summary["minimum_jacobian"],
                "sign_agreement": summary["sign_agreement"],
            },
            indent=2,
        ),
        flush=True,
    )
