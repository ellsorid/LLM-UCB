"""Stage a minimal native runtime with one 40–70 enthalpy definition."""

from pathlib import Path
import shutil

WINDOW_MODULE = """import numpy as np

DH_LOW, DH_TARGET, DH_HIGH = 40.0, 54.0, 70.0
DH_EDGES = (DH_LOW, 48.0, DH_TARGET, 62.0, np.nextafter(DH_HIGH, np.inf))


def dh_quality(value, low=DH_LOW, target=DH_TARGET, high=DH_HIGH):
    left = max(target - low, 1e-9)
    right = max(high - target, 1e-9)
    return np.clip(1.0 - np.abs(value - target) / np.where(value <= target, left, right), 0.0, 1.0)
"""


def replace(text, old, new, expected=1):
    if text.count(old) != expected:
        raise RuntimeError(f"Native source changed around {old!r}")
    return text.replace(old, new)


def stage(source: Path, destination: Path):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "enthalpy_window.py").write_text(WINDOW_MODULE, encoding="utf-8")
    shared_import = "from enthalpy_window import DH_LOW, DH_TARGET, DH_HIGH, DH_EDGES, dh_quality\n"
    utils_path = destination / "utils.py"
    text = utils_path.read_text(encoding="utf-8")
    text = replace(text, "from adaptive_ga import (", shared_import + "from adaptive_ga import (")
    text = replace(text, "[48.0, 51.0, 54.0, 58.0, 62.000001]", "DH_EDGES")
    text = replace(text, "dh < 48.0 or dh > 62.0", "dh < DH_LOW or dh > DH_HIGH")
    text = replace(text, "np.clip(1.0 - abs(dh - 54.0) / 8.0, 0.0, 1.0)", "dh_quality(dh)")
    text = replace(text, "pdh.between(48.0, 62.0)", "pdh.between(DH_LOW, DH_HIGH)")
    text = replace(text, """        if 'target_template_score' in df.columns:
            furan_like = pd.to_numeric(df['target_template_score'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64) >= 2.0
        else:
            furan_like = np.zeros(df.shape[0], dtype=bool)
        x_low = np.where(furan_like, 48.0, 55.0)
        x_high = np.where(furan_like, 62.0, 68.0)
        pass_enthalpy = pd.Series(np.isfinite(x) & (x >= x_low) & (x <= x_high), index=df.index)""",
                   """        pass_enthalpy = pd.Series(np.isfinite(x) & (x >= DH_LOW) & (x <= DH_HIGH), index=df.index)""")
    text = replace(text, """    dh_half_width = max(dh_target - dh_low, dh_high - dh_target, 1e-9)
    dh_score = _pareto_clip01(1.0 - np.abs(dh - dh_target) / dh_half_width)""",
                   """    dh_score = dh_quality(dh, dh_low, dh_target, dh_high)""")
    text = replace(text, "x_low, x_high, x_ref = 48.0, 62.0, 55.0", "x_low, x_high, x_ref = DH_LOW, DH_HIGH, 55.0")
    text = replace(text, "x_low, x_high, x_ref = 54.0, 68.0, 61.0", "x_low, x_high, x_ref = DH_LOW, DH_HIGH, 61.0")
    for field, value, constant in (
        ("dh_window_low", "48.0", "DH_LOW"), ("dh_window_high", "62.0", "DH_HIGH"),
        ("dh_target", "54.0", "DH_TARGET"),
    ):
        text = replace(text, f"'{field}': {value}", f"'{field}': {constant}")
        text = replace(text, f"cfg.get('{field}', {value})", f"cfg.get('{field}', {constant})",
                       1 if field == "dh_target" else 2)
    text = replace(text, "cfg.get('lohc_dh_low', 48.0)", "cfg.get('lohc_dh_low', DH_LOW)")
    text = replace(text, "cfg.get('lohc_dh_high', 62.0)", "cfg.get('lohc_dh_high', DH_HIGH)")
    utils_path.write_text(text, encoding="utf-8")

    path = destination / "adaptive_ga.py"
    text = path.read_text(encoding="utf-8")
    text = replace(text, "import pandas as pd\n", "import pandas as pd\n" + shared_import)
    text = replace(text, 'cfg.get("dh_window_low", 48.0)', 'cfg.get("dh_window_low", DH_LOW)', 2)
    text = replace(text, 'cfg.get("dh_window_high", 62.0)', 'cfg.get("dh_window_high", DH_HIGH)', 2)
    text = replace(text, 'cfg.get("dh_target", 54.0)', 'cfg.get("dh_target", DH_TARGET)')
    text = replace(text, "48.0 <= float(dh) <= 62.0", "DH_LOW <= float(dh) <= DH_HIGH")
    text = replace(text, "[48.0, 51.0, 54.0, 58.0, 62.000001]", "DH_EDGES")
    text = replace(text, "np.clip(1.0 - abs(dh - 54.0) / 8.0, 0.0, 1.0)", "dh_quality(dh)")
    text = replace(text, 'np.clip(1.0 - abs(float(row["delta_H_per_H2"]) - 54.0) / 8.0, 0.0, 1.0)',
                   'dh_quality(float(row["delta_H_per_H2"]))')
    path.write_text(text, encoding="utf-8")

    path = destination / "main.py"
    text = path.read_text(encoding="utf-8")
    text = replace(text, "from pathlib import Path\n", "from pathlib import Path\n" + shared_import)
    for field, value, constant in (
        ("dh_window_low", "48.0", "DH_LOW"), ("dh_window_high", "62.0", "DH_HIGH"),
        ("dh_target", "54.0", "DH_TARGET"), ("lohc_dh_low", "48.0", "DH_LOW"), ("lohc_dh_high", "62.0", "DH_HIGH"),
    ):
        text = replace(text, f"'{field}': {value}", f"'{field}': {constant}")
    path.write_text(text, encoding="utf-8")

    path = destination / "predict.py"
    text = path.read_text(encoding="utf-8")
    text = replace(text, """        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
        models.append(model)""", """        # A fixed point prediction keeps the same pair comparable across arms.
        model.eval()
        models.append(model)""")
    path.write_text(text, encoding="utf-8")
