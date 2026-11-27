"""Molecule-centric two-step LLM directed editing coupled with an 8x16 flattened SOM stimulation heatmap."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

THIS_DIR = Path(__file__).resolve().parent
NATURE_FIG_SCRIPTS = Path(r"D:\skills\nature-skills\skills\nature-figure\scripts")
for _p in (str(THIS_DIR), str(NATURE_FIG_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader
import style
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT_DIR = THIS_DIR / "output" / "fig2_ucb"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = OUT_DIR / "source_data"
SRC_DIR.mkdir(parents=True, exist_ok=True)


def _quad_bezier(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], n: int = 80) -> np.ndarray:
    """Evaluate a smooth quadratic Bezier curve from p0 through control p1 to p2."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1.0 - t) ** 2 * np.asarray(p0) + 2.0 * (1.0 - t) * t * np.asarray(p1) + t ** 2 * np.asarray(p2)
    return pts


def _prepare_aligned_mols(smiles_root: str, smiles_step1: str, smiles_step2: str) -> dict[str, object]:
    """Build Kekulized RDKit mols with exact shared parent-scaffold 2D coordinates derived from Step 2."""
    m0_raw = Chem.MolFromSmiles(smiles_root)
    m1_raw = Chem.MolFromSmiles(smiles_step1)
    m2_raw = Chem.MolFromSmiles(smiles_step2)
    if m0_raw is None or m1_raw is None or m2_raw is None:
        raise ValueError("Invalid SMILES passed to _prepare_aligned_mols")

    # Compute 2D coords once on m2 and align m1 -> m2 and m0 -> m1 via exact ancestry
    AllChem.Compute2DCoords(m2_raw)
    m1_in_m2 = m2_raw.GetSubstructMatch(m1_raw)
    m0_in_m1 = m1_raw.GetSubstructMatch(m0_raw)
    if len(m1_in_m2) != m1_raw.GetNumAtoms() or len(m0_in_m1) != m0_raw.GetNumAtoms():
        raise RuntimeError("Failed chained substructure alignment m0 -> m1 -> m2")
    m0_in_m2 = tuple(m1_in_m2[i] for i in m0_in_m1)

    conf2 = m2_raw.GetConformer()
    pts2 = np.asarray(
        [[conf2.GetAtomPosition(i).x, conf2.GetAtomPosition(i).y] for i in range(m2_raw.GetNumAtoms())],
        dtype=np.float64,
    )
    pts1 = pts2[list(m1_in_m2)]
    pts0 = pts2[list(m0_in_m2)]

    # Kekulize copies for crisp explicit single/double ring bonds
    m0 = Chem.Mol(m0_raw)
    m1 = Chem.Mol(m1_raw)
    m2 = Chem.Mol(m2_raw)
    for m in (m0, m1, m2):
        Chem.Kekulize(m, clearAromaticFlags=True)

    new1_in_m1 = sorted(set(range(m1.GetNumAtoms())) - set(m0_in_m1))
    new2_in_m2 = sorted(set(range(m2.GetNumAtoms())) - set(m1_in_m2))
    step1_in_m2 = [m1_in_m2[i] for i in new1_in_m1]

    return {
        "m0": m0,
        "m1": m1,
        "m2": m2,
        "pts0": pts0,
        "pts1": pts1,
        "pts2": pts2,
        "m0_in_m1": m0_in_m1,
        "m1_in_m2": m1_in_m2,
        "m0_in_m2": m0_in_m2,
        "new1_in_m1": new1_in_m1,
        "new2_in_m2": new2_in_m2,
        "step1_in_m2": step1_in_m2,
        "site0_m0": 0,
        "site2_m0": 2,
        "site0_m1": m0_in_m1[0],
        "site2_m1": m0_in_m1[2],
        "site2_m2": m0_in_m2[2],
        "xlim": (float(pts2[:, 0].min() - 1.35), float(pts2[:, 0].max() + 1.35)),
        "ylim": (float(pts2[:, 1].min() - 1.25), float(pts2[:, 1].max() + 1.25)),
    }


def _draw_bond_segment(
    ax: plt.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    order: float,
    color: str,
    lw: float,
    zorder: int,
    ring_center: np.ndarray | None = None,
    glow_color: str | None = None,
) -> None:
    """Draw single, double, or triple covalent bonds cleanly in vector SVG/PDF."""
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return
    u = vec / length
    perp = np.array([-u[1], u[0]], dtype=np.float64)

    if glow_color is not None:
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=glow_color,
            lw=lw + 4.4,
            alpha=0.14,
            solid_capstyle="round",
            zorder=zorder - 1,
        )

    if order == 1.0:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    elif order == 2.0:
        if ring_center is not None:
            mid = 0.5 * (p1 + p2)
            inward = ring_center - mid
            if np.dot(inward, perp) < 0:
                perp = -perp
            off = 0.155
            s1, s2 = p1 + perp * off + u * 0.13, p2 + perp * off - u * 0.13
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
            ax.plot([s1[0], s2[0]], [s1[1], s2[1]], color=color, lw=max(lw * 0.78, 0.95), solid_capstyle="round", zorder=zorder)
        else:
            off = 0.115
            for sign in (-1.0, 1.0):
                q1, q2 = p1 + sign * off * perp, p2 + sign * off * perp
                ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=color, lw=min(lw * 0.76, 1.35), solid_capstyle="round", zorder=zorder)
    elif order == 3.0:
        off = 0.175
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=min(lw * 0.74, 1.25), solid_capstyle="round", zorder=zorder)
        for sign in (-1.0, 1.0):
            q1 = p1 + sign * off * perp + u * 0.08
            q2 = p2 + sign * off * perp - u * 0.08
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=color, lw=min(lw * 0.62, 1.00), solid_capstyle="round", zorder=zorder)


def _draw_aligned_molecule(
    ax: plt.Axes,
    mol: Chem.Mol,
    pts: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    base_color: str,
    bond_overrides: dict[tuple[int, int], tuple[str, float, str | None]],
    atom_halos: dict[int, tuple[str, float, str | None]],
    site_tags: list[tuple[int, str, tuple[float, float], str]],
) -> None:
    """Render an aligned RDKit molecule purely as crisp Matplotlib lines and heteroatom glyphs."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    def _ring_center_for_bond(a_idx: int, b_idx: int) -> np.ndarray | None:
        for ring in atom_rings:
            if a_idx in ring and b_idx in ring:
                return pts[list(ring)].mean(axis=0)
        return None

    # Soft halos behind highlighted edited atoms
    for aidx, (hcol, rad, _) in atom_halos.items():
        ax.add_patch(
            Circle(
                (pts[aidx, 0], pts[aidx, 1]),
                radius=rad,
                facecolor=hcol,
                edgecolor="none",
                alpha=0.19,
                zorder=2,
            )
        )

    # Bonds (trimmed cleanly outside heteroatom label clearances so strokes never cross text)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(i, j), max(i, j))
        order = float(bond.GetBondTypeAsDouble())
        rc = _ring_center_for_bond(i, j)
        p1 = pts[i].copy()
        p2 = pts[j].copy()
        dvec = p2 - p1
        dlen = float(np.linalg.norm(dvec))
        if dlen > 1e-6:
            du = dvec / dlen
            ai = mol.GetAtomWithIdx(i)
            aj = mol.GetAtomWithIdx(j)
            if ai.GetSymbol() != "C":
                trim_i = 0.56 if ai.GetTotalNumHs() > 0 else 0.48
                p1 = p1 + du * min(trim_i, 0.42 * dlen)
            if aj.GetSymbol() != "C":
                trim_j = 0.56 if aj.GetTotalNumHs() > 0 else 0.48
                p2 = p2 - du * min(trim_j, 0.42 * dlen)
        if key in bond_overrides:
            bcol, blw, glow = bond_overrides[key]
            bz = 6
        else:
            bcol, blw, glow = base_color, 1.20, None
            bz = 4
        _draw_bond_segment(ax, p1, p2, order=order, color=bcol, lw=blw, zorder=bz, ring_center=rc, glow_color=glow)

    # Highlight crisp rings on targeted atoms (if specified)
    for aidx, (hcol, rad, edge_ls) in atom_halos.items():
        if edge_ls is not None:
            ax.add_patch(
                Circle(
                    (pts[aidx, 0], pts[aidx, 1]),
                    radius=rad * 0.72,
                    facecolor="none",
                    edgecolor=hcol,
                    lw=0.95,
                    linestyle=edge_ls,
                    alpha=0.88,
                    zorder=7,
                )
            )

    # Heteroatom labels with clean white masking pad
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == "C":
            continue
        nh = atom.GetTotalNumHs()
        label = f"{sym}H{nh}" if nh > 1 else (f"{sym}H" if nh == 1 else sym)
        ax.text(
            pts[idx, 0],
            pts[idx, 1],
            label,
            fontsize=5.8,
            fontweight="bold",
            color=style.DARK_INK,
            ha="center",
            va="center",
            zorder=8,
        )

    # Site callout tags (S0 / S2)
    for aidx, tag, (dx, dy), tcol in site_tags:
        tx, ty = float(pts[aidx, 0] + dx), float(pts[aidx, 1] + dy)
        ax.text(
            tx,
            ty,
            tag,
            fontsize=5.5,
            fontweight="bold",
            color=tcol,
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            zorder=9,
        )


def build_figure(run_dir: Path | str = loader.DEFAULT_RUN_DIR) -> dict[str, object]:
    """Render Fig2b_llm_directed on a 180 x 120 mm Nature canvas."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    bundle = loader.load_run_bundle(run_dir=run_dir)
    data = loader.compute_llm_directed_som_data(bundle, generation=6)
    root = data["root"]
    s1 = data["step1"]
    s2 = data["step2"]

    mols = _prepare_aligned_mols(root["lean_smiles"], s1["lean_smiles"], s2["lean_smiles"])

    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), facecolor="white")

    # Header
    fig.text(
        0.045,
        0.952,
        "LLM-directed two-step molecular editing on the 2D SOM plane",
        fontsize=10.5,
        fontweight="bold",
        color=style.DARK_INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.045,
        0.913,
        "G6 chain (A1 -> A1B2): site-selective grafting drives cross-cell leap (C95 -> C51 -> C78) and 1.94x H2 gain",
        fontsize=6.6,
        color=style.MUTED,
        ha="left",
        va="top",
    )

    # Top Row: 3 Equal-Width Rounded Cards
    card_specs = [
        {"x0": 0.045, "w": 0.262, "edge": "#C8BDDD", "lw": 0.85, "title": "Root · Cell 95", "title_col": style.DARK_INK, "foot": f"2 H2 · {root['h2_wt_percent']:.2f} wt% · Score 62.47"},
        {"x0": 0.364, "w": 0.262, "edge": "#CC7D9C", "lw": 0.95, "title": "Step 1 (A1) · Cell 51", "title_col": "#B3446D", "foot": f"3 H2 · {s1['h2_wt_percent']:.2f} wt% · dr +{s1['destination_response_delta']:.4f}"},
        {"x0": 0.683, "w": 0.262, "edge": "#8F1F49", "lw": 1.15, "title": "Step 2 (A1B2) · Cell 78", "title_col": "#8F1F49", "foot": f"5 H2 · {s2['h2_wt_percent']:.2f} wt% · dr +{s2['destination_response_delta']:.4f}"},
    ]
    y_card, h_card = 0.568, 0.305

    bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    bg_ax.set_xlim(0.0, 1.0)
    bg_ax.set_ylim(0.0, 1.0)
    bg_ax.axis("off")

    card_centers_bottom = []
    for cs in card_specs:
        patch = FancyBboxPatch(
            (cs["x0"], y_card),
            cs["w"],
            h_card,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor="#FCFAFE" if cs["edge"] == "#C8BDDD" else ("#FDF8FB" if cs["edge"] == "#CC7D9C" else "#FCF4F7"),
            edgecolor=cs["edge"],
            linewidth=cs["lw"],
            zorder=1,
        )
        bg_ax.add_patch(patch)
        cx = cs["x0"] + 0.5 * cs["w"]
        fig.text(
            cx,
            y_card + h_card - 0.014,
            cs["title"],
            fontsize=6.5,
            fontweight="bold",
            color=cs["title_col"],
            ha="center",
            va="top",
            zorder=10,
        )
        fig.text(
            cx,
            y_card + 0.014,
            cs["foot"],
            fontsize=5.8,
            color=style.INK,
            ha="center",
            va="bottom",
            zorder=10,
        )
        card_centers_bottom.append((cx, y_card))

    # Sleek step transitions between cards (no arrows)
    trans_specs = [
        (0.312, 0.359, "+vinyl", "@ Site 0", "#BE557A"),
        (0.631, 0.678, "+ethynyl", "@ Site 2", "#8F1F49"),
    ]
    y_mid = y_card + 0.5 * h_card
    for x_a, x_b, l1, l2, col in trans_specs:
        bg_ax.plot([x_a, x_b], [y_mid, y_mid], color=col, lw=1.4, solid_capstyle="round", zorder=3)
        bg_ax.scatter([x_b], [y_mid], s=14, color=col, edgecolors="white", linewidths=0.5, zorder=4)
        xm = 0.5 * (x_a + x_b)
        fig.text(xm, y_mid + 0.016, l1, fontsize=6.0, fontweight="bold", color=col, ha="center", va="bottom", zorder=10)
        fig.text(xm, y_mid - 0.016, l2, fontsize=5.6, color=style.MUTED, ha="center", va="top", zorder=10)

    # Molecule Axes inside the 3 cards
    mol_ax_rects = [
        [cs["x0"] + 0.008, y_card + 0.044, cs["w"] - 0.016, h_card - 0.092]
        for cs in card_specs
    ]
    ax_m0 = fig.add_axes(mol_ax_rects[0], zorder=5)
    ax_m1 = fig.add_axes(mol_ax_rects[1], zorder=5)
    ax_m2 = fig.add_axes(mol_ax_rects[2], zorder=5)

    # Card 1: Root with S0 and S2 targets highlighted
    _draw_aligned_molecule(
        ax_m0,
        mols["m0"],
        mols["pts0"],
        mols["xlim"],
        mols["ylim"],
        base_color="#6E607E",
        bond_overrides={},
        atom_halos={
            mols["site0_m0"]: ("#BE557A", 0.68, "--"),
            mols["site2_m0"]: ("#8F1F49", 0.68, "--"),
        },
        site_tags=[
            (mols["site0_m0"], "S0", (0.55, 0.98), "#BE557A"),
            (mols["site2_m0"], "S2", (-0.62, -0.92), "#8F1F49"),
        ],
    )

    # Card 2: Step 1 (A1) with grafted vinyl on Site 0 highlighted
    m1_site0 = mols["site0_m1"]  # atom 2 in m1
    _draw_aligned_molecule(
        ax_m1,
        mols["m1"],
        mols["pts1"],
        mols["xlim"],
        mols["ylim"],
        base_color="#786A89",
        bond_overrides={
            (0, 1): ("#BE557A", 2.05, "#BE557A"),
            (1, m1_site0): ("#BE557A", 2.05, "#BE557A"),
        },
        atom_halos={
            0: ("#BE557A", 0.62, None),
            1: ("#BE557A", 0.62, None),
            mols["site2_m1"]: ("#8F1F49", 0.66, "--"),
        },
        site_tags=[
            (1, "S0", (0.15, 0.92), "#BE557A"),
            (mols["site2_m1"], "S2", (-0.62, -0.92), "#8F1F49"),
        ],
    )

    # Card 3: Step 2 (A1B2) with Step 1 vinyl softly rose + Step 2 ethynyl deep wine
    _draw_aligned_molecule(
        ax_m2,
        mols["m2"],
        mols["pts2"],
        mols["xlim"],
        mols["ylim"],
        base_color="#786A89",
        bond_overrides={
            (4, 5): ("#CC7898", 1.75, None),
            (5, 6): ("#CC7898", 1.75, None),
            (0, 1): ("#8F1F49", 2.25, "#8F1F49"),
            (1, 2): ("#8F1F49", 2.25, "#8F1F49"),
        },
        atom_halos={
            5: ("#CC7898", 0.56, None),
            6: ("#CC7898", 0.56, None),
            0: ("#8F1F49", 0.66, None),
            1: ("#8F1F49", 0.66, None),
        },
        site_tags=[
            (5, "S0", (0.15, 0.92), "#BE557A"),
            (1, "S2", (1.05, -0.10), "#8F1F49"),
        ],
    )

    # Bottom Hero Panel: Flattened 8x16 SOM Response Shift Heatmap
    hm_rect = [0.062, 0.105, 0.806, 0.375]
    ax_h = fig.add_axes(hm_rect, zorder=5)
    grid = data["grid_delta_psi"]
    occ = data["grid_occupied"]
    cell_to_rc = data["cell_to_rc"]

    norm = TwoSlopeNorm(vmin=-0.0035, vcenter=0.0, vmax=0.0096)
    ax_h.imshow(
        grid,
        cmap=style.MIGRATION_DIV_CMAP,
        norm=norm,
        origin="upper",
        extent=(0.0, 16.0, 8.0, 0.0),
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )

    # Clean white tile separator grid (16x8)
    for cx in range(1, 16):
        ax_h.axvline(cx, color="white", lw=0.75, alpha=0.85, zorder=2)
    for ry in range(1, 8):
        ax_h.axhline(ry, color="white", lw=0.75, alpha=0.85, zorder=2)

    # Subtle dot on occupied SOM cells
    occ_r, occ_c = np.where(occ > 0)
    occ_dark_bg = grid[occ_r, occ_c] > 0.0068
    ax_h.scatter(
        occ_c[~occ_dark_bg] + 0.83,
        occ_r[~occ_dark_bg] + 0.80,
        s=5.0,
        color="#564568",
        alpha=0.58,
        edgecolors="none",
        zorder=3,
    )
    ax_h.scatter(
        occ_c[occ_dark_bg] + 0.83,
        occ_r[occ_dark_bg] + 0.80,
        s=5.0,
        color="#F8F4FA",
        alpha=0.85,
        edgecolors="none",
        zorder=3,
    )

    def _tile_center(cid: int) -> tuple[float, float]:
        r, c = cell_to_rc[cid]
        return (c + 0.5, r + 0.5)

    p95 = _tile_center(95)
    p51 = _tile_center(51)
    p78 = _tile_center(78)
    p37 = _tile_center(37)

    # Rejected candidate jump (A2B2 -> C37)
    rej_arc = _quad_bezier(p95, (6.0, 4.10), p37, n=70)
    ax_h.plot(rej_arc[:, 0], rej_arc[:, 1], color="white", lw=2.3, alpha=0.72, zorder=4)
    ax_h.plot(rej_arc[:, 0], rej_arc[:, 1], color="#635473", lw=1.15, ls="--", alpha=0.92, zorder=5)
    ax_h.add_patch(
        Rectangle((p37[0] - 0.5, p37[1] - 0.5), 1.0, 1.0, facecolor="none", edgecolor="#5B4D6B", lw=1.05, ls="--", zorder=6)
    )
    ax_h.scatter([p37[0]], [p37[1]], s=24, facecolor="white", edgecolor="#5B4D6B", linewidths=1.1, zorder=8)
    ax_h.text(
        p37[0] + 0.66,
        p37[1],
        "A2B2\nC37",
        fontsize=5.6,
        fontweight="bold",
        color="#524463",
        ha="left",
        va="center",
        linespacing=0.95,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        zorder=9,
    )

    # Trapped local candidate ring at C95 (A2/A4 stay inside C95)
    ax_h.add_patch(
        Circle(p95, radius=0.40, facecolor="none", edgecolor="#5B4D6B", lw=0.85, ls=":", alpha=0.85, zorder=6)
    )

    # Selected Step 1 arc (C95 -> C51)
    arc1 = _quad_bezier(p95, (4.0, 1.48), p51, n=80)
    ax_h.plot(arc1[:, 0], arc1[:, 1], color="white", lw=3.2, alpha=0.88, solid_capstyle="round", zorder=6)
    ax_h.plot(arc1[:, 0], arc1[:, 1], color="#BE557A", lw=1.95, solid_capstyle="round", zorder=7)

    # Selected Step 2 arc (C51 -> C78)
    arc2 = _quad_bezier(p51, (9.4, 0.50), p78, n=90)
    ax_h.plot(arc2[:, 0], arc2[:, 1], color="white", lw=3.6, alpha=0.90, solid_capstyle="round", zorder=6)
    ax_h.plot(arc2[:, 0], arc2[:, 1], color="#8F1F49", lw=2.35, solid_capstyle="round", zorder=7)

    # Tile borders and landmark markers for C95, C51, C78
    landmarks_hm = [
        (95, p95, "#493D54", "o", 38, "Root\nC95", (p95[0] - 0.68, p95[1]), "right", "center"),
        (51, p51, "#BE557A", "o", 40, "A1\nC51", (p51[0], p51[1] - 0.68), "center", "bottom"),
        (78, p78, "#8F1F49", "*", 92, "A1B2\nC78", (p78[0] + 0.66, p78[1] + 0.04), "left", "center"),
    ]
    for cid, (px, py), col, marker, ms, lbl, (lx, ly), ha, va in landmarks_hm:
        ax_h.add_patch(
            Rectangle((px - 0.5, py - 0.5), 1.0, 1.0, facecolor="none", edgecolor=col, lw=1.45, zorder=8)
        )
        ax_h.scatter([px], [py], s=ms, facecolor=col, edgecolor="white", linewidths=1.05, marker=marker, zorder=9)
        ax_h.text(
            lx,
            ly,
            lbl,
            fontsize=5.8,
            fontweight="bold",
            color=col,
            ha=ha,
            va=va,
            linespacing=0.95,
            path_effects=[pe.withStroke(linewidth=2.3, foreground="white")],
            zorder=10,
        )

    # Axis ticks and spines for the 8x16 SOM grid
    ax_h.set_xticks([0.5, 3.5, 7.5, 11.5, 15.5])
    ax_h.set_xticklabels(["1", "4", "8", "12", "16"], fontsize=6.0)
    ax_h.set_yticks([0.5, 2.5, 4.5, 7.5])
    ax_h.set_yticklabels(["1", "3", "5", "8"], fontsize=6.0)
    ax_h.tick_params(length=2.4, width=0.65, color=style.INK, pad=2.0)
    ax_h.set_xlabel("SOM grid column u (128 prototypes reshaped to 8 x 16 plane)", fontsize=6.6, labelpad=3.0)
    ax_h.set_ylabel("SOM row v", fontsize=6.6, labelpad=3.5)
    for sp in ax_h.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.75)
        sp.set_color("#8B7B99")

    # Colorbar
    cax = fig.add_axes([0.884, hm_rect[1], 0.017, hm_rect[3]], zorder=5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=style.MIGRATION_DIV_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ticks=[-0.003, 0.000, 0.003, 0.006, 0.009])
    cb.ax.set_yticklabels(["-0.003", "0.000", "+0.003", "+0.006", "+0.009"], fontsize=5.8)
    cb.ax.tick_params(length=2.2, width=0.6, color=style.INK, pad=2.0)
    cb.outline.set_linewidth(0.65)
    cb.outline.set_edgecolor("#8B7B99")
    cb.set_label("SOM shift dPsi (A1B2 - Root)", fontsize=6.3, labelpad=4.0, color=style.INK)

    # Zero-crossing dotted leader hairlines connecting Molecule Cards to their SOM destination tiles
    leader_specs = [
        (card_centers_bottom[0], (p95[0], p95[1] - 0.50), "#493D54"),
        (card_centers_bottom[1], (p51[0], p51[1] - 0.50), "#BE557A"),
        (card_centers_bottom[2], (p78[0], p78[1] - 0.50), "#8F1F49"),
    ]
    for (fx, fy), (hx, hy), lcol in leader_specs:
        con = ConnectionPatch(
            xyA=(fx, fy),
            coordsA=fig.transFigure,
            xyB=(hx, hy),
            coordsB=ax_h.transData,
            linestyle=":",
            linewidth=0.95,
            color=lcol,
            alpha=0.78,
            zorder=4,
        )
        fig.add_artist(con)

    # Compact legend below bottom heatmap
    leg_handles = [
        mlines.Line2D([], [], color="#8F1F49", lw=2.0, label="Selected 2-step edit (A1 -> A1B2)"),
        mlines.Line2D([], [], color="#635473", lw=1.2, ls="--", label="Rejected path (A2B2 -> C37)"),
        mlines.Line2D([], [], color="none", marker="o", markersize=3.2, markerfacecolor="#564568", markeredgecolor="none", label="Occupied SOM cell (n=42)"),
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.465, 0.004),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=1.8,
        columnspacing=1.35,
    )

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_m0, ax_m1, ax_m2, ax_h],
        panel_ids=["m0", "m1", "m2", "h"],
        exclude_axes=[bg_ax, cax],
        row_groups=[{"panels": ["m0", "m1", "m2"], "check_widths": True}],
        require_panel_labels=False,
        strict=True,
    )

    svg_path = OUT_DIR / "Fig2b_llm_directed.svg"
    pdf_path = OUT_DIR / "Fig2b_llm_directed.pdf"
    png_path = OUT_DIR / "Fig2b_llm_directed.png"
    tiff_path = OUT_DIR / "Fig2b_llm_directed.tiff"

    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(tiff_path, dpi=600, facecolor="white")
    plt.close(fig)

    # Export reproducible source CSVs
    pd.DataFrame(data["som_table"]).to_csv(SRC_DIR / "Fig2b_llm_directed_som_grid.csv", index=False)
    chain_rows = [
        {
            "stage": "Root",
            "edit_id": "root",
            "cell": int(root["cell"]),
            "som_row": int(cell_to_rc[int(root["cell"])][0]),
            "som_col": int(cell_to_rc[int(root["cell"])][1]),
            "lean_smiles": root["lean_smiles"],
            "nH2": float(root["nH2"]),
            "h2_wt_percent": float(root["h2_wt_percent"]),
            "destination_response_delta": 0.0,
            "delta_psi_at_cell": float(data["delta_psi_total"][int(root["cell"])]),
        },
        {
            "stage": "Step1",
            "edit_id": s1["id"],
            "cell": int(s1["cell"]),
            "som_row": int(cell_to_rc[int(s1["cell"])][0]),
            "som_col": int(cell_to_rc[int(s1["cell"])][1]),
            "lean_smiles": s1["lean_smiles"],
            "nH2": float(s1["nH2"]),
            "h2_wt_percent": float(s1["h2_wt_percent"]),
            "destination_response_delta": float(s1["destination_response_delta"]),
            "delta_psi_at_cell": float(data["delta_psi_total"][int(s1["cell"])]),
        },
        {
            "stage": "Step2",
            "edit_id": s2["id"],
            "cell": int(s2["cell"]),
            "som_row": int(cell_to_rc[int(s2["cell"])][0]),
            "som_col": int(cell_to_rc[int(s2["cell"])][1]),
            "lean_smiles": s2["lean_smiles"],
            "nH2": float(s2["nH2"]),
            "h2_wt_percent": float(s2["h2_wt_percent"]),
            "destination_response_delta": float(s2["destination_response_delta"]),
            "delta_psi_at_cell": float(data["delta_psi_total"][int(s2["cell"])]),
        },
    ]
    pd.DataFrame(chain_rows).to_csv(SRC_DIR / "Fig2b_llm_directed_chain.csv", index=False)

    contract = {
        "figure": "Fig2b_llm_directed",
        "width_mm": 180.0,
        "height_mm": 120.0,
        "run_dir": str(run_dir),
        "generation": 6,
        "chosen_ids": data["chosen_ids"],
        "argmax_cells": data["argmax_cells"],
        "path_cells": [int(root["cell"]), int(s1["cell"]), int(s2["cell"])],
        "h2_wt_percent": [float(root["h2_wt_percent"]), float(s1["h2_wt_percent"]), float(s2["h2_wt_percent"])],
        "delta_psi_at_dst": float(data["delta_psi_total"][int(s2["cell"])]),
        "arrows": False,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2b_llm_directed.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = build_figure()
    print(json.dumps(summary, indent=2), flush=True)
