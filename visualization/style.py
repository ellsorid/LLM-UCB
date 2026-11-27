"""Unified Nature-style + D:\\figure lavender-to-burgundy visual engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.transforms import ScaledTranslation
from scipy.ndimage import gaussian_filter

# Core ink and neutral tones
INK = "#493D54"
DARK_INK = "#382E42"
LIGHT_INK = "#6F5F7D"
MUTED = "#887491"
UNDEF_GRID = "#B7AFBF"

# Cool purple / lavender family (structure, bootstrap, Whole, lean)
BASE_LAVENDER = "#EDE6F5"
COOL_PALE = "#E2D7EE"
COOL_SOFT = "#CBBCE2"
COOL_MID = "#9B85C2"
COOL_SLATE = "#8172B5"
COOL_DARK = "#766083"

# Warm rose / burgundy family (discovery, hero, Local, rich)
ROSE_PALE = "#F0D9E4"
ROSE_SOFT = "#DDB3C7"
ROSE_MID = "#B7788C"
WINE_MID = "#963F65"
WINE = "#872541"

# 3D shading and shadow anchors
SHADOW_HEX = "#675675"
GRID_HEX = "#725684"
GRID_MAJOR_HEX = "#634876"

# Smooth sequential colormap matching D:\figure\SOM aesthetics
SOM_RELIEF_CMAP = LinearSegmentedColormap.from_list(
    "som_relief",
    [
        (0.00, "#EEE8F6"),
        (0.26, "#D3C4E5"),
        (0.52, "#B79BC9"),
        (0.76, "#AE6C91"),
        (1.00, "#872541"),
    ],
)

# Cool lavender-to-deep-violet colormap for structure differential (||dC||)
STRUCT_DIFF_CMAP = LinearSegmentedColormap.from_list(
    "struct_diff",
    [
        (0.00, "#F2EDF7"),
        (0.32, "#D2C3E6"),
        (0.66, "#9C83C4"),
        (1.00, "#5B4182"),
    ],
)

# Rose-to-burgundy colormap for score differential (dmu)
SCORE_DIFF_CMAP = LinearSegmentedColormap.from_list(
    "score_diff",
    [
        (0.00, "#EFEAF6"),
        (0.28, "#DDBCD1"),
        (0.62, "#BA7593"),
        (1.00, "#872541"),
    ],
)

# Diverging lavender-white-burgundy colormap (saddle / signed delta)
SOM_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "som_diverging",
    [
        (0.00, "#7C67A8"),
        (0.28, "#BFB0DA"),
        (0.50, "#F2EDF6"),
        (0.72, "#C889A1"),
        (1.00, "#872541"),
    ],
)


def apply_nature_som_style(font_size: float = 6.8, axes_linewidth: float = 0.7) -> None:
    """Apply Nature-compliant typography and editable vector export settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "axes.titlesize": font_size + 0.7,
            "axes.labelsize": font_size,
            "xtick.labelsize": max(font_size - 0.8, 5.5),
            "ytick.labelsize": max(font_size - 0.8, 5.5),
            "legend.fontsize": max(font_size - 0.6, 5.8),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": axes_linewidth,
            "xtick.major.width": axes_linewidth * 0.75,
            "ytick.major.width": axes_linewidth * 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def shade_surface_white_composited(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    values: np.ndarray,
    cmap: LinearSegmentedColormap,
    norm: mpl.colors.Normalize,
    opacity_base: float = 0.82,
    light: tuple[float, float, float] = (-0.48, -0.48, 0.735),
    edge_taper: bool = True,
    smooth_sigma: float = 0.40,
    stain_density: np.ndarray | None = None,
    stain_hex: str | None = None,
    stain_strength: float = 0.45,
) -> np.ndarray:
    """Compute white-composited directional surface RGBA array with optional deep molecular pigment stain."""
    xyz = np.stack([X, Y, Z], axis=-1)
    normals = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    norms = np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-10)
    normals = normals / norms

    light_vec = np.asarray(light, dtype=float)
    light_vec = light_vec / np.maximum(np.linalg.norm(light_vec), 1e-9)
    diffuse = np.clip(normals @ light_vec, 0.0, 1.0)
    val_norm = np.clip(norm(values), 0.0, 1.0)
    shoulder = np.clip(1.0 - np.abs(normals[..., 2]), 0.0, 1.0) * np.exp(-((val_norm - 0.25) / 0.28) ** 2)

    brightness = 0.78 + 0.29 * diffuse - 0.075 * shoulder
    rgb = cmap(val_norm)[..., :3]
    if stain_density is not None and stain_hex is not None:
        deep_rgb = np.asarray(to_rgb(stain_hex), dtype=float)[None, None, :]
        w = np.clip(stain_density * stain_strength, 0.0, 0.72)[..., None]
        rgb = rgb * (1.0 - w) + deep_rgb * w
        brightness = brightness * (1.0 - 0.14 * np.clip(stain_density, 0.0, 1.0))

    rgb = np.clip(rgb * brightness[..., None], 0.0, 1.0)
    if smooth_sigma > 0:
        rgb = gaussian_filter(rgb, sigma=(smooth_sigma, smooth_sigma, 0.0))

    op = np.full_like(Z, opacity_base, dtype=float)
    if stain_density is not None:
        op = np.clip(op + 0.12 * np.clip(stain_density, 0.0, 1.0), 0.0, 0.96)
    if edge_taper:
        row = np.linspace(0.0, 1.0, Z.shape[0])[:, None]
        col = np.linspace(0.0, 1.0, Z.shape[1])[None, :]
        edge = np.minimum(np.minimum(row, 1.0 - row), np.minimum(col, 1.0 - col))
        taper = np.clip(edge / 0.08, 0.0, 1.0)
        taper = taper * taper * (3.0 - 2.0 * taper)
        op = op * (0.45 + 0.55 * taper)

    composited = 1.0 - op[..., None] + op[..., None] * rgb
    rgba = np.ones((*Z.shape, 4), dtype=float)
    rgba[..., :3] = np.clip(composited, 0.0, 1.0)
    return rgba


def make_organic_feathered_shadow(
    organic_mask: np.ndarray,
    shadow_hex: str = SHADOW_HEX,
    max_alpha: float = 0.072,
    sigma: float = 8.5,
) -> np.ndarray:
    """Create a white-composited feathered drop shadow matching an organic hull mask."""
    alpha = gaussian_filter(organic_mask.astype(float) * max_alpha, sigma=sigma)
    shadow = np.ones((*organic_mask.shape, 4), dtype=float)
    shadow[..., :3] = (
        1.0
        - alpha[..., None]
        + alpha[..., None] * np.asarray(to_rgb(shadow_hex), dtype=float)
    )
    return np.clip(shadow, 0.0, 1.0)


def make_feathered_shadow(
    Z_shape: tuple[int, int],
    shadow_hex: str = SHADOW_HEX,
    max_alpha: float = 0.068,
    sigma: float = 7.0,
) -> np.ndarray:
    """Create a white-composited feathered drop shadow for 3D surface floor planes."""
    h, w = Z_shape
    row = np.linspace(0.0, 1.0, h)[:, None]
    col = np.linspace(0.0, 1.0, w)[None, :]
    edge = np.minimum(np.minimum(row, 1.0 - row), np.minimum(col, 1.0 - col))
    fade = np.clip(edge / 0.14, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    shadow_alpha = gaussian_filter(max_alpha * fade, sigma=sigma)
    shadow = np.ones((h, w, 4), dtype=float)
    shadow[..., :3] = (
        1.0
        - shadow_alpha[..., None]
        + shadow_alpha[..., None] * np.asarray(to_rgb(shadow_hex), dtype=float)
    )
    return np.clip(shadow, 0.0, 1.0)


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    x: float = 0.0,
    y: float = 1.0,
    x_offset_pt: float = -4.0,
    y_offset_pt: float = 3.0,
    fontsize: float = 8.0,
    color: str = DARK_INK,
    fontweight: str = "bold",
    va: str = "bottom",
) -> mpl.text.Text:
    """Place a lowercase bold panel label using physical point offsets."""
    offset = ScaledTranslation(
        x_offset_pt / 72.0,
        y_offset_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    return text_fn(
        x,
        y,
        label,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left",
        va=va,
    )


def save_outputs(
    fig: mpl.figure.Figure,
    out_dir: Path | str,
    stem: str,
    dpi_png: int = 600,
    formats: tuple[str, ...] = ("svg", "pdf", "png"),
    contract: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save figure artifacts (SVG/PDF/PNG) and optional figure contract JSON cleanly."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for fmt in formats:
        target = out_path / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if fmt in ("png", "tif", "tiff"):
            kwargs["dpi"] = dpi_png
        fig.savefig(target, **kwargs)
        saved[fmt] = target
    if contract is not None:
        contract_path = out_path / f"{stem}.contract.json"
        contract_path.write_text(
            json.dumps(contract, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        saved["contract"] = contract_path
    plt.close(fig)
    return saved


SADDLE_SATIN_COLORS = ["#8774B1", "#B8A6D2", "#EEE8F4", "#C69BB7", "#9A476D"]
SADDLE_SATIN_CMAP = LinearSegmentedColormap.from_list(
    "signed_deformation",
    SADDLE_SATIN_COLORS,
    N=2048,
)


def shade_satin_diverging_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    values: np.ndarray,
    norm: mpl.colors.Normalize,
    alpha: float = 0.96,
    cmap: mpl.colors.Colormap | None = None,
) -> np.ndarray:
    """Compute diffuse + specular satin RGBA facecolors matching draw_saddle.py."""
    xyz = np.stack([X, Y, Z], axis=-1)
    normal = np.cross(np.gradient(xyz, axis=1), np.gradient(xyz, axis=0))
    if float(np.nanmean(normal[..., 2])) < 0.0:
        normal = -normal
    normal = normal / np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-10)

    light = np.array([-0.4, -0.5, 0.77], dtype=np.float64)
    light = light / np.linalg.norm(light)
    diffuse = np.maximum(normal @ light, 0.0)

    half = light + np.array([0.4, -0.6, 0.7], dtype=np.float64)
    half = half / np.linalg.norm(half)
    specular = np.maximum(normal @ half, 0.0) ** 32

    use_cmap = SADDLE_SATIN_CMAP if cmap is None else cmap
    rgba = use_cmap(norm(values)).copy()
    rgba[..., :3] *= (0.83 + 0.17 * diffuse)[..., None]
    rgba[..., :3] += (1.0 - rgba[..., :3]) * (0.10 * specular)[..., None]
    rgba[..., 3] = alpha
    return np.clip(rgba, 0.0, 1.0)


PLANAR_RED_COLORS = ["#FDF5F7", "#F2C7D4", "#DB87A3", "#BA4B73", "#8B1E47"]
PLANAR_RED_CMAP = LinearSegmentedColormap.from_list(
    "planar_red_move",
    PLANAR_RED_COLORS,
    N=2048,
)

SCORE_PURPLE_COLORS = ["#F8F5FC", "#DDD1F0", "#B39BD9", "#8267B5", "#51348A"]
SCORE_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "score_purple_move",
    SCORE_PURPLE_COLORS,
    N=2048,
)

MIGRATION_DIV_COLORS = ["#593D94", "#8971BA", "#C3B5DF", "#EEE8F4", "#DC9AB0", "#BE557A", "#8F1F49"]
MIGRATION_DIV_CMAP = LinearSegmentedColormap.from_list(
    "migration_purple_red",
    MIGRATION_DIV_COLORS,
    N=2048,
)
