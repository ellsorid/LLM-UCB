"""SOM population-migration mesh warp (red=immigration, purple=emigration) with strictly upward Z>=0 high-score probability."""
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
    """Render Fig2_som_migration_relief cleanly on a 180 x 120 mm Nature canvas."""
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
    relief = loader.compute_som_migration_relief_data(bundle)

    U, V = relief["U"], relief["V"]
    X, Y, Z = relief["X"], relief["Y"], relief["Z"]
    drho = relief["drho"]
    nv, nu = U.shape

    lim = float(np.max(np.abs(drho)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    rgba = style.shade_satin_diverging_surface(X, Y, Z, drho, norm, alpha=0.88, cmap=style.MIGRATION_DIV_CMAP)

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor="white")
    fig.text(
        0.045,
        0.950,
        "SOM population migration and high-score probability relief",
        fontsize=11.0,
        fontweight="bold",
        color=style.INK,
    )
    fig.text(
        0.045,
        0.915,
        "Planar mesh warps with structure migration (red: immigration, purple: emigration); upward Z >= 0 encodes high-score probability",
        fontsize=7.3,
        color=style.MUTED,
    )

    ax = fig.add_axes([-0.03, 0.190, 1.06, 0.70], projection="3d", computed_zorder=False)
    ax.patch.set_alpha(0.0)

    # Flat pale-gray initial SOM plane (Z = 0) beneath the translucent upward score-probability surface.
    flat_lines = []
    deformed_lines = []
    for r in np.linspace(0, nv - 1, 23).astype(int):
        flat_lines.append(np.column_stack([U[r], V[r], np.zeros(nu, dtype=np.float64)]))
        deformed_lines.append(np.column_stack([X[r], Y[r], Z[r]]))
    for c in np.linspace(0, nu - 1, 29).astype(int):
        flat_lines.append(np.column_stack([U[:, c], V[:, c], np.zeros(nv, dtype=np.float64)]))
        deformed_lines.append(np.column_stack([X[:, c], Y[:, c], Z[:, c]]))

    ax.add_collection3d(
        Line3DCollection(flat_lines, colors="#9E94AA", linewidths=0.28, alpha=0.38, zorder=0),
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
        Line3DCollection(deformed_lines, colors="#665378", linewidths=0.23, alpha=0.38, zorder=3),
        autolim=False,
    )

    # Subtle iso-probability contours on the upward relief make Z elevation levels immediately readable.
    for level, segs in relief["iso_contours"].items():
        if segs:
            ax.add_collection3d(
                Line3DCollection(segs, colors="#FFFFFF", linewidths=0.50, alpha=0.52, zorder=4),
                autolim=False,
            )

    ax.set(
        xlim=(-2.65, 2.65),
        ylim=(-2.30, 2.30),
        zlim=(-0.04, float(Z.max()) + 0.16),
    )
    ax.view_init(elev=25, azim=-54)
    ax.set_box_aspect((1.25, 1.0, 0.62), zoom=1.40)
    ax.axis("off")
    for coll in ax.collections:
        coll.set_clip_on(False)

    cax = fig.add_axes([0.26, 0.095, 0.48, 0.012])
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=style.MIGRATION_DIV_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[-lim, 0.0, lim],
    )
    cb.ax.set_xticklabels(
        ["Emigration (\u0394\u03c1 < 0)", "Zero shift (\u0394\u03c1 = 0)", "Immigration (\u0394\u03c1 > 0)"]
    )
    cb.ax.tick_params(labelsize=5.8, length=1.5, pad=2, colors=style.INK)
    cb.outline.set_visible(False)

    fig.canvas.draw()

    imm = relief["imm_peak"]
    emi = relief["emi_peak"]
    top_z = relief["top_z_peak"]

    # Subtle surface anchor rings for key peaks/basins (zero scatter clutter, zero arrows).
    anchors = [
        (imm, "#8F1F49", f"Immigration peak (\u0394\u03c1 > 0, P = {imm['p_high']:.2f})", (0.715, 0.660), "left", True),
        (emi, "#593D94", f"Emigration basin (\u0394\u03c1 < 0, P = {emi['p_high']:.2f})", (0.295, 0.755), "right", False),
        (top_z, "#6A2B55", f"High-score peak (Z >= 0, P = {top_z['p_high']:.2f})", (0.640, 0.810), "left", True),
    ]
    for lm, col, text, (tx, ty), ha, is_bold in anchors:
        ax.scatter(
            lm["x"],
            lm["y"],
            lm["z"],
            s=30,
            marker="o",
            color=col,
            edgecolor="white",
            linewidth=0.85,
            zorder=7,
        )
        px2d, py2d, _ = proj3d.proj_transform(lm["x"], lm["y"], lm["z"], ax.get_proj())
        fx0, fy0 = fig.transFigure.inverted().transform(ax.transData.transform((px2d, py2d)))
        lx_end = tx - 0.008 if ha == "left" else tx + 0.008
        dx, dy = lx_end - fx0, ty - fy0
        dist = max(float(np.hypot(dx, dy)), 1e-6)
        sx = fx0 + dx / dist * 0.008
        sy = fy0 + dy / dist * 0.008
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
            fontsize=6.8,
            fontweight="bold" if is_bold else "normal",
            color=col,
            zorder=10,
        )

    fig.text(0.08, 0.140, "Gray grid: initial bounded SOM plane (Z = 0)", fontsize=6.5, color="#8F859B")
    fig.text(0.55, 0.140, "Colored mesh: migration-warped SOM, Z >= 0 score prob.", fontsize=6.5, color="#6D5A82")

    corr_val = float(relief["corr_drho_p"])
    min_j = float(relief["min_jacobian"])
    fig.text(
        0.045,
        0.032,
        (
            "Real SOM migration (live_large_pop_v1, G0-G2 vs G3-G6, n = 144): "
            f"XY contracts in red immigration zones (r = {corr_val:+.2f} with P_high), Z >= 0 only, min J = {min_j:.2f}."
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

    svg_path = OUT_DIR / "Fig2_som_migration_relief.svg"
    pdf_path = OUT_DIR / "Fig2_som_migration_relief.pdf"
    png_path = OUT_DIR / "Fig2_som_migration_relief.png"
    tiff_path = OUT_DIR / "Fig2_som_migration_relief.tiff"

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")
    plt.close(fig)

    pd.DataFrame([imm, emi, top_z]).to_csv(SRC_DIR / "Fig2_som_migration_relief_landmarks.csv", index=False)
    np.savez_compressed(
        SRC_DIR / "Fig2_som_migration_relief_arrays.npz",
        U=U,
        V=V,
        X=X,
        Y=Y,
        Z=Z,
        J=relief["J"],
        drho=drho,
        P_high=relief["P_high"],
        rho_early=relief["rho_early"],
        rho_late=relief["rho_late"],
        mol_pts=relief["mol_pts"],
        gens=relief["gens"],
        scores=relief["scores"],
    )

    contract = {
        "figure": "Fig2_som_migration_relief",
        "width_mm": 180.0,
        "height_mm": 120.0,
        "run_dir": str(run_dir),
        "n_molecules": int(len(relief["gens"])),
        "z_min": float(Z.min()),
        "z_max": float(Z.max()),
        "p_high_max": float(relief["P_high"].max()),
        "corr_drho_p_high": corr_val,
        "j_immigration_mean": float(relief["j_imm_mean"]),
        "j_emigration_mean": float(relief["j_emi_mean"]),
        "minimum_jacobian": min_j,
        "gamma_warp": float(relief["gamma"]),
        "landmarks": {"immigration_peak": imm, "emigration_basin": emi, "top_score_prob": top_z},
        "arrows": False,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2_som_migration_relief.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = build_figure()
    print(
        json.dumps(
            {
                "z_min": summary["z_min"],
                "z_max": summary["z_max"],
                "p_high_max": summary["p_high_max"],
                "corr_drho_p_high": summary["corr_drho_p_high"],
                "j_immigration_mean": summary["j_immigration_mean"],
                "j_emigration_mean": summary["j_emigration_mean"],
                "minimum_jacobian": summary["minimum_jacobian"],
            },
            indent=2,
        ),
        flush=True,
    )
