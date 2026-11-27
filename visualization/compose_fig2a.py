"""Assemble Panel a (Fig2a_composed) directly from the three standalone SVG primitives."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import fitz
import numpy as np
from PIL import Image

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "output" / "fig2_saddle"
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from matplotlib.colors import LinearSegmentedColormap
from visualization.style import MIGRATION_DIV_CMAP, SADDLE_SATIN_CMAP

W_MM, H_MM = 180.0, 106.0
W_PT = W_MM / 25.4 * 72.0  # 510.23622 pt
H_PT = H_MM / 25.4 * 72.0  # 300.47244 pt


def _extract_primitive(svg_path: Path, prefix: str) -> tuple[str, list[ET.Element], ET.Element]:
    """Load source SVG, drop standalone text/callout/colorbar elements, and namespace IDs."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    src_vb = root.attrib.get("viewBox", "0 0 510.23622 340.15748")

    defs_out: list[ET.Element] = []
    for child in root:
        if child.tag == f"{{{SVG_NS}}}defs":
            defs_out.append(copy.deepcopy(child))

    fig_g = next(el for el in root.iter() if el.attrib.get("id") == "figure_1")
    axes_g = next(el for el in fig_g if el.attrib.get("id") == "axes_1")
    axes_copy = copy.deepcopy(axes_g)

    # Remove any text groups if present inside axes_1 so primitives remain clean graphic objects.
    def _strip_text(parent: ET.Element) -> None:
        to_remove = []
        for ch in parent:
            cid = ch.attrib.get("id", "")
            if ch.tag == f"{{{SVG_NS}}}text" or cid.startswith("text_"):
                to_remove.append(ch)
            else:
                _strip_text(ch)
        for ch in to_remove:
            parent.remove(ch)

    _strip_text(axes_copy)

    id_map: dict[str, str] = {}
    for container in defs_out + [axes_copy]:
        for el in container.iter():
            old_id = el.attrib.get("id")
            if old_id:
                new_id = f"{prefix}{old_id}"
                id_map[old_id] = new_id
                el.attrib["id"] = new_id

    url_pat = re.compile(r"url\(#([^)]+)\)")
    for container in defs_out + [axes_copy]:
        for el in container.iter():
            for attr in list(el.attrib.keys()):
                val = el.attrib[attr]
                if "url(#" in val:
                    val = url_pat.sub(lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})", val)
                    el.attrib[attr] = val
                elif attr in ("href", f"{{{XLINK_NS}}}href") and val.startswith("#"):
                    key = val[1:]
                    if key in id_map:
                        el.attrib[attr] = f"#{id_map[key]}"

    return src_vb, defs_out, axes_copy


def _measure_tight_bbox(src_vb: str, defs_list: list[ET.Element], axes_el: ET.Element) -> tuple[float, float, float, float]:
    """Render extracted primitive alone via PyMuPDF and compute tight non-white bbox in source pt."""
    vb_parts = [float(x) for x in src_vb.split()]
    sw, sh = vb_parts[2], vb_parts[3]

    tmp_svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{sw:.5f}pt",
            "height": f"{sh:.5f}pt",
            "viewBox": src_vb,
        },
    )
    for d in defs_list:
        tmp_svg.append(copy.deepcopy(d))
    tmp_svg.append(copy.deepcopy(axes_el))

    svg_bytes = ET.tostring(tmp_svg, encoding="utf-8", xml_declaration=True)
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "probe.svg"
        tmp_path.write_bytes(svg_bytes)
        with fitz.open(str(tmp_path)) as sdoc:
            pdf_bytes = sdoc.convert_to_pdf()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdoc:
        pix = pdoc[0].get_pixmap(dpi=144, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

    mask = np.any(arr < 252, axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0, sw, sh)

    px_to_pt_x = sw / pix.width
    px_to_pt_y = sh / pix.height
    pad = 2.5
    x0 = max(0.0, float(xs.min()) * px_to_pt_x - pad)
    y0 = max(0.0, float(ys.min()) * px_to_pt_y - pad)
    x1 = min(sw, float(xs.max() + 1) * px_to_pt_x + pad)
    y1 = min(sh, float(ys.max() + 1) * px_to_pt_y + pad)
    return (x0, y0, x1, y1)


def _add_vector_colorbar(
    parent: ET.Element,
    x0: float,
    y0: float,
    width: float,
    height: float,
    cmap: LinearSegmentedColormap,
    n_bins: int = 128,
) -> None:
    """Draw a PyMuPDF-safe horizontal colorbar from contiguous vector strips."""
    g = ET.SubElement(parent, f"{{{SVG_NS}}}g")
    step = width / n_bins
    # Slight +0.15px overlap prevents subpixel seams in vector PDF/SVG viewers.
    for i in range(n_bins):
        r, g_c, b, _ = cmap((i + 0.5) / n_bins)
        hex_col = f"#{int(round(r * 255)):02x}{int(round(g_c * 255)):02x}{int(round(b * 255)):02x}"
        w_strip = step + (0.15 if i < n_bins - 1 else 0.0)
        ET.SubElement(
            g,
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{x0 + i * step:.3f}",
                "y": f"{y0:.3f}",
                "width": f"{w_strip:.3f}",
                "height": f"{height:.3f}",
                "fill": hex_col,
                "stroke": "none",
            },
        )
    ET.SubElement(
        g,
        f"{{{SVG_NS}}}rect",
        {
            "x": f"{x0:.3f}",
            "y": f"{y0:.3f}",
            "width": f"{width:.3f}",
            "height": f"{height:.3f}",
            "fill": "none",
            "stroke": "#D4CCE0",
            "stroke-width": "0.35",
        },
    )


def _add_text(
    parent: ET.Element,
    x: float,
    y: float,
    text: str,
    size_pt: float,
    color: str,
    bold: bool = False,
    anchor: str = "start",
) -> None:
    """Append isolated native editable Arial SVG text element in exact pt user units."""
    holder = ET.SubElement(parent, f"{{{SVG_NS}}}g", {"transform": f"translate({x:.2f},{y:.2f})"})
    el = ET.SubElement(
        holder,
        f"{{{SVG_NS}}}text",
        {
            "x": "0",
            "y": "0",
            "fill": color,
            "font-family": "Arial, Helvetica, sans-serif",
            "font-size": f"{size_pt:.2f}",
            "font-weight": "700" if bold else "400",
            "text-anchor": anchor,
        },
    )
    el.text = text


def compose_panel_a() -> dict[str, object]:
    """Assemble a1 (structure saddle), a2 (score gradient), and a3 (SOM migration relief) into Fig2a."""
    primitives = [
        ("a1", "p1_", OUT_DIR / "Fig2_directed_saddle.svg", (10.0, 20.0, 218.0, 140.0)),
        ("a2", "p2_", OUT_DIR / "Fig2_score_gradient.svg", (10.0, 156.0, 218.0, 272.0)),
        ("a3", "p3_", OUT_DIR / "Fig2_som_migration_relief.svg", (234.0, 24.0, 500.0, 270.0)),
    ]

    comp_svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{W_PT:.5f}pt",
            "height": f"{H_PT:.5f}pt",
            "viewBox": f"0 0 {W_PT:.5f} {H_PT:.5f}",
            "version": "1.1",
        },
    )

    master_defs = ET.SubElement(comp_svg, f"{{{SVG_NS}}}defs")

    # White background canvas
    ET.SubElement(
        comp_svg,
        f"{{{SVG_NS}}}rect",
        {"x": "0", "y": "0", "width": f"{W_PT:.5f}", "height": f"{H_PT:.5f}", "fill": "#ffffff"},
    )

    placed_meta: dict[str, dict[str, object]] = {}
    for pid, prefix, path, (tx0, ty0, tx1, ty1) in primitives:
        src_vb, defs_list, axes_el = _extract_primitive(path, prefix)
        for d in defs_list:
            for ch in d:
                master_defs.append(ch)
        bx0, by0, bx1, by1 = _measure_tight_bbox(src_vb, defs_list, axes_el)
        bw = max(bx1 - bx0, 1e-6)
        bh = max(by1 - by0, 1e-6)
        tw = tx1 - tx0
        th = ty1 - ty0
        scale = min(tw / bw, th / bh)
        ox = tx0 + 0.5 * (tw - bw * scale) - bx0 * scale
        oy = ty0 + 0.5 * (th - bh * scale) - by0 * scale

        wrapper = ET.SubElement(
            comp_svg,
            f"{{{SVG_NS}}}g",
            {
                "id": f"panel_{pid}",
                "transform": f"translate({ox:.3f},{oy:.3f}) scale({scale:.6f})",
            },
        )
        wrapper.append(axes_el)
        placed_meta[pid] = {
            "source_svg": str(path),
            "source_tight_bbox_pt": [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)],
            "target_box_pt": [tx0, ty0, tx1, ty1],
            "scale": round(scale, 5),
        }

    # Subtle column separator hairline
    ET.SubElement(
        comp_svg,
        f"{{{SVG_NS}}}line",
        {
            "x1": "226.0",
            "y1": "20.0",
            "x2": "226.0",
            "y2": "276.0",
            "stroke": "#E2DCE8",
            "stroke-width": "0.65",
        },
    )

    # Minimal native Arial panel labels & concise subtitles (8.0 pt bold tag + 6.8 pt title)
    _add_text(comp_svg, 12.0, 16.0, "a1", 8.0, "#382E42", bold=True)
    _add_text(comp_svg, 25.0, 16.0, "Structure differential", 6.8, "#6D5D7C", bold=False)

    _add_text(comp_svg, 12.0, 152.0, "a2", 8.0, "#382E42", bold=True)
    _add_text(comp_svg, 25.0, 152.0, "Score gradient", 6.8, "#6D5D7C", bold=False)

    _add_text(comp_svg, 236.0, 16.0, "a3", 8.0, "#382E42", bold=True)
    _add_text(comp_svg, 249.0, 16.0, "SOM migration & upward score probability", 6.8, "#6D5D7C", bold=False)

    # Compact vector colorbar for left column (a1 + a2)
    _add_vector_colorbar(comp_svg, 44.0, 280.0, 140.0, 4.8, SADDLE_SATIN_CMAP)
    _add_text(comp_svg, 44.0, 294.0, "Downward", 5.8, "#493D54", anchor="start")
    _add_text(comp_svg, 114.0, 294.0, "Z = 0", 5.8, "#493D54", anchor="middle")
    _add_text(comp_svg, 184.0, 294.0, "Upward", 5.8, "#493D54", anchor="end")

    # Compact vector colorbar for right hero (a3)
    _add_vector_colorbar(comp_svg, 292.0, 280.0, 150.0, 4.8, MIGRATION_DIV_CMAP)
    _add_text(comp_svg, 292.0, 294.0, "Emigration", 5.8, "#493D54", anchor="start")
    _add_text(comp_svg, 367.0, 294.0, "Zero shift", 5.8, "#493D54", anchor="middle")
    _add_text(comp_svg, 442.0, 294.0, "Immigration", 5.8, "#493D54", anchor="end")

    svg_path = OUT_DIR / "Fig2a_composed.svg"
    pdf_path = OUT_DIR / "Fig2a_composed.pdf"
    png_path = OUT_DIR / "Fig2a_composed.png"
    tiff_path = OUT_DIR / "Fig2a_composed.tiff"

    svg_bytes = ET.tostring(comp_svg, encoding="utf-8", xml_declaration=True)
    svg_path.write_bytes(svg_bytes)

    with fitz.open(str(svg_path)) as sdoc:
        pdf_bytes = sdoc.convert_to_pdf()
    pdf_path.write_bytes(pdf_bytes)

    with fitz.open(str(pdf_path)) as pdoc:
        pix = pdoc[0].get_pixmap(dpi=600, alpha=False)
        pix.save(str(png_path))
    with Image.open(png_path) as img:
        img.save(tiff_path, dpi=(600, 600))

    contract = {
        "figure": "Fig2a_composed",
        "width_mm": W_MM,
        "height_mm": H_MM,
        "width_pt": round(W_PT, 3),
        "height_pt": round(H_PT, 3),
        "composed_directly_from_svg_primitives": True,
        "panels": placed_meta,
        "outputs": [str(svg_path), str(pdf_path), str(png_path), str(tiff_path)],
    }
    (OUT_DIR / "Fig2a_composed.contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract


if __name__ == "__main__":
    summary = compose_panel_a()
    print(json.dumps(summary["panels"], indent=2), flush=True)
