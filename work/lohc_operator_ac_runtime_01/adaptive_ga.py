from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from enthalpy_window import DH_LOW, DH_TARGET, DH_HIGH, DH_EDGES, dh_quality


MUTATION_OPERATORS = (
    "group_addition",
    "relocation",
    "atom_insertion",
    "atom_deletion",
    "bond_mutation",
    "bond_addition",
    "bond_deletion",
    "atom_substitution",
)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def classify_objective_state(row, config=None) -> tuple[str, str, str]:
    cfg = config or {}
    h2_min = float(cfg.get("h2wt_min", 4.2))
    dh_low = float(cfg.get("dh_window_low", DH_LOW))
    dh_high = float(cfg.get("dh_window_high", DH_HIGH))
    sa_max = float(cfg.get("sa_target_max", 4.0))

    h2 = row.get("h2_wt_percent", np.nan)
    dh = row.get("delta_H_per_H2", np.nan)
    sa = row.get("SA score", np.nan)

    h2_state = "unknown" if not _finite(h2) else ("sufficient" if float(h2) >= h2_min else "deficient")
    if not _finite(dh):
        dh_state = "unknown"
    elif float(dh) < dh_low:
        dh_state = "low"
    elif float(dh) > dh_high:
        dh_state = "high"
    else:
        dh_state = "in_window"
    sa_state = "gate_only"
    return h2_state, dh_state, sa_state


def context_key(platform_id, h2_state, dh_state, sa_state) -> str:
    return "|".join(
        str(v) if str(v).strip() else "unknown"
        for v in (platform_id, h2_state, dh_state, sa_state)
    )


@dataclass
class OperatorExperiencePolicy:
    mode: str = "ucb"
    beta: float = 0.8
    random_fraction: float = 0.10
    warmup_generations: int = 5
    temperature: float = 0.35
    seed: int = 42
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in {"random", "mean", "ucb"}:
            raise ValueError(f"Unsupported adaptive mode: {self.mode}")
        self.rng = np.random.default_rng(int(self.seed))

    def _entry(self, key: str, operator: str) -> dict:
        bucket = self.stats.setdefault(key, {})
        return bucket.setdefault(
            operator,
            {
                "count": 0,
                "reward_sum": 0.0,
                "reward_sq_sum": 0.0,
                "valid_count": 0,
                "pareto_entry_count": 0,
                "target_hit_count": 0,
                "delta_h2_sum": 0.0,
                "delta_dh_gain_sum": 0.0,
                "delta_sa_gain_sum": 0.0,
            },
        )

    def _scores(self, key: str) -> np.ndarray:
        counts = np.array([self._entry(key, op)["count"] for op in MUTATION_OPERATORS], dtype=float)
        rewards = np.array([self._entry(key, op)["reward_sum"] for op in MUTATION_OPERATORS], dtype=float)
        means = np.divide(rewards, counts, out=np.zeros_like(rewards), where=counts > 0)
        means = np.nan_to_num(means, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.mode == "random":
            return np.zeros(len(MUTATION_OPERATORS), dtype=float)
        if self.mode == "mean":
            return means + (counts == 0).astype(float) * 0.25
        total = max(1.0, float(counts.sum()))
        uncertainty = float(self.beta) * np.sqrt(np.log(total + 2.0) / (counts + 1.0))
        return means + uncertainty

    def choose_order(self, key: str, generation: int, parent_smi=None) -> tuple[list[int], dict[str, float]]:
        n_ops = len(MUTATION_OPERATORS)
        if (
            self.mode == "random"
            or generation < int(self.warmup_generations)
            or self.rng.random() < float(self.random_fraction)
        ):
            order = self.rng.permutation(n_ops).tolist()
            probs = np.full(n_ops, 1.0 / n_ops, dtype=float)
        else:
            scores = self._scores(key)
            shifted = (scores - np.max(scores)) / max(float(self.temperature), 1e-6)
            probs = np.exp(np.clip(shifted, -30.0, 30.0))
            prob_sum = float(np.sum(probs))
            if not np.all(np.isfinite(probs)) or not np.isfinite(prob_sum) or prob_sum <= 0.0:
                probs = np.full(n_ops, 1.0 / n_ops, dtype=float)
            else:
                probs = probs / prob_sum
            order = self.rng.choice(n_ops, size=n_ops, replace=False, p=probs).tolist()
        return order, {op: float(probs[i]) for i, op in enumerate(MUTATION_OPERATORS)}

    def update(self, transitions: pd.DataFrame) -> None:
        if transitions is None or transitions.empty:
            return
        for row in transitions.to_dict("records"):
            operator = str(row.get("mutation_operator", ""))
            if operator not in MUTATION_OPERATORS:
                continue
            key = str(row.get("experience_context", "unknown|unknown|unknown|unknown"))
            entry = self._entry(key, operator)
            reward = float(row.get("operator_reward", -1.0))
            if not np.isfinite(reward):
                reward = 0.0 if int(row.get("child_valid", 0)) else -1.0
            entry["count"] += 1
            entry["reward_sum"] += reward
            entry["reward_sq_sum"] += reward * reward
            entry["valid_count"] += int(row.get("child_valid", 0))
            entry["pareto_entry_count"] += int(row.get("entered_pareto_selection", 0))
            entry["target_hit_count"] += int(row.get("child_target_hit", 0))
            for source, target in (
                ("delta_h2", "delta_h2_sum"),
                ("delta_dh_gain", "delta_dh_gain_sum"),
                ("delta_sa_gain", "delta_sa_gain_sum"),
            ):
                value = row.get(source, np.nan)
                if _finite(value):
                    entry[target] += float(value)

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for key, operators in self.stats.items():
            platform, h2_state, dh_state, sa_state = (key.split("|") + ["unknown"] * 4)[:4]
            for operator in MUTATION_OPERATORS:
                entry = self._entry(key, operator)
                count = int(entry["count"])
                denom = max(1, count)
                mean_reward = entry["reward_sum"] / denom
                variance = max(0.0, entry["reward_sq_sum"] / denom - mean_reward**2)
                rows.append(
                    {
                        "experience_context": key,
                        "platform_id": platform,
                        "h2_state": h2_state,
                        "dh_state": dh_state,
                        "sa_state": sa_state,
                        "mutation_operator": operator,
                        "count": count,
                        "mean_reward": mean_reward if count else np.nan,
                        "reward_std": math.sqrt(variance) if count else np.nan,
                        "valid_rate": entry["valid_count"] / denom if count else np.nan,
                        "pareto_entry_rate": entry["pareto_entry_count"] / denom if count else np.nan,
                        "target_hit_rate": entry["target_hit_count"] / denom if count else np.nan,
                        "mean_delta_h2": entry["delta_h2_sum"] / denom if count else np.nan,
                        "mean_delta_dh_gain": entry["delta_dh_gain_sum"] / denom if count else np.nan,
                        "mean_delta_sa_gain": entry["delta_sa_gain_sum"] / denom if count else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, output_dir) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_frame().to_csv(output_dir / "operator_experience.csv", index=False)
        with (output_dir / "operator_experience_state.json").open("w", encoding="utf-8") as handle:
            json.dump(self.stats, handle, indent=2, sort_keys=True)


def build_transition_records(df_ranked, generation: int, config=None) -> pd.DataFrame:
    if df_ranked is None or df_ranked.empty or "origin_type" not in df_ranked.columns:
        return pd.DataFrame()
    cfg = config or {}
    rows = df_ranked[df_ranked["origin_type"].astype(str) == "guided_mutation"].copy()
    if "operation_generation" in rows.columns:
        op_generation = pd.to_numeric(rows["operation_generation"], errors="coerce")
        rows = rows[op_generation == int(generation)].copy()
    if rows.empty:
        return pd.DataFrame()

    target_dh = float(cfg.get("dh_target", DH_TARGET))
    h2_min = float(cfg.get("h2wt_min", 4.2))
    dh_low = float(cfg.get("dh_window_low", DH_LOW))
    dh_high = float(cfg.get("dh_window_high", DH_HIGH))
    sa_max = float(cfg.get("sa_target_max", 4.0))
    records = []
    for row in rows.to_dict("records"):
        child_h2 = row.get("h2_wt_percent", np.nan)
        child_dh = row.get("delta_H_per_H2", np.nan)
        child_sa = row.get("SA score", np.nan)
        parent_h2 = row.get("parent_h2_wt_percent", np.nan)
        parent_dh = row.get("parent_delta_H_per_H2", np.nan)
        parent_sa = row.get("parent_SA_score", np.nan)
        child_valid = int(all(_finite(v) for v in (child_h2, child_dh, child_sa)))
        delta_h2 = float(child_h2) - float(parent_h2) if _finite(child_h2) and _finite(parent_h2) else np.nan
        delta_dh_gain = (
            float(dh_quality(child_dh) - dh_quality(parent_dh))
            if _finite(child_dh) and _finite(parent_dh)
            else np.nan
        )
        delta_sa_gain = float(parent_sa) - float(child_sa) if _finite(child_sa) and _finite(parent_sa) else np.nan
        child_target_hit = int(
            child_valid
            and float(child_h2) >= h2_min
            and dh_low <= float(child_dh) <= dh_high
            and float(child_sa) <= sa_max
        )
        entered = int(row.get("selected_pareto", 0) == 1 and int(row.get("parent_selected_pareto", 0)) == 0)
        if not child_valid or float(child_sa) > sa_max:
            reward = -1.0
        else:
            h2_component = (
                float(np.clip(delta_h2 / 1.5, -1.0, 1.0))
                if _finite(delta_h2)
                else 0.0
            )
            dh_component = (
                float(np.clip(delta_dh_gain, -1.0, 1.0))
                if _finite(delta_dh_gain)
                else 0.0
            )
            sa_component = (
                float(np.clip(delta_sa_gain / 2.0, -1.0, 1.0))
                if _finite(delta_sa_gain)
                else 0.0
            )
            reward = (
                0.35 * h2_component
                + 0.30 * dh_component
                + 0.10 * child_target_hit
                + 0.05 * entered
            )
        records.append(
            {
                "generation": int(generation),
                "parent_smi": row.get("parent_smi", ""),
                "child_smi": row.get("smi", ""),
                "platform_id": row.get("parent_platform_id", row.get("platform_id", "unassigned")),
                "experience_context": row.get("experience_context", "unknown|unknown|unknown|unknown"),
                "mutation_operator": row.get("mutation_operator", ""),
                "operator_probability": row.get("operator_probability", np.nan),
                "parent_h2_wt_percent": parent_h2,
                "child_h2_wt_percent": child_h2,
                "parent_delta_H_per_H2": parent_dh,
                "child_delta_H_per_H2": child_dh,
                "parent_SA_score": parent_sa,
                "child_SA_score": child_sa,
                "delta_h2": delta_h2,
                "delta_dh_gain": delta_dh_gain,
                "delta_sa_gain": delta_sa_gain,
                "child_valid": child_valid,
                "child_target_hit": child_target_hit,
                "entered_pareto_selection": entered,
                "operator_reward": reward if reward == -1.0 else reward / 0.8,
            }
        )
    return pd.DataFrame(records)


def append_csv(frame: pd.DataFrame, path) -> None:
    if frame is None or frame.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


class ExplorationTracker:
    def __init__(self, config=None):
        self.config = config or {}
        self.seen_smis = set()
        self.seen_scaffolds = set()
        self.target_cells = {}
        self.metrics = []
        self.total_target_hits = 0
        self.total_evaluations = 0
        self.hv_grid_size = int(self.config.get("hypervolume_grid_size", 18))
        self.hv_occupied = np.zeros((self.hv_grid_size,) * 2, dtype=bool)

    @staticmethod
    def _bin(value, edges):
        if not _finite(value):
            return -1
        return int(np.digitize([float(value)], edges, right=False)[0] - 1)

    def _target_cell(self, row):
        platform_id = str(row.get("platform_id", "unassigned"))
        if platform_id not in BIO_PLATFORM_IDS:
            return None
        h2 = row.get("h2_wt_percent", np.nan)
        dh = row.get("delta_H_per_H2", np.nan)
        sa = row.get("SA score", np.nan)
        if not all(_finite(v) for v in (h2, dh, sa)):
            return None
        if float(h2) < 4.2 or not DH_LOW <= float(dh) <= DH_HIGH or float(sa) > 4.0:
            return None
        h2_bin = self._bin(h2, [4.2, 4.8, 5.4, 6.0, np.inf])
        return (platform_id, h2_bin)

    @staticmethod
    def _quality(row):
        h2 = float(row["h2_wt_percent"])
        dh = float(row["delta_H_per_H2"])
        sa = float(row["SA score"])
        h2_q = np.clip((h2 - 4.2) / (7.0 - 4.2), 0.0, 1.0)
        dh_q = dh_quality(dh)
        return float((h2_q + dh_q) / 2.0)

    @staticmethod
    def _update_hypervolume_grid(grid, points):
        size = int(grid.shape[0])
        for point in points:
            values = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
            if not np.all(np.isfinite(values)):
                continue
            indices = np.floor(values * (size - 1)).astype(int)
            grid[: indices[0] + 1, : indices[1] + 1] = True
        return float(grid.mean())

    def update(self, frame: pd.DataFrame, generation: int) -> dict:
        if frame is None or frame.empty:
            return {}
        new_rows = frame[~frame["smi"].astype(str).isin(self.seen_smis)].copy()
        self.seen_smis.update(new_rows["smi"].astype(str).tolist())
        self.total_evaluations += int(new_rows.shape[0])
        new_scaffolds = set(str(v) for v in new_rows.get("scaffold_smi", pd.Series(dtype=str)).dropna() if str(v))
        unseen_scaffolds = new_scaffolds - self.seen_scaffolds
        self.seen_scaffolds.update(new_scaffolds)

        previous_cell_count = len(self.target_cells)
        previous_qd_sum = float(sum(self.target_cells.values()))
        generation_hits = 0
        hv_points = []
        for row in new_rows.to_dict("records"):
            cell = self._target_cell(row)
            if cell is None:
                continue
            generation_hits += 1
            quality = self._quality(row)
            if quality > self.target_cells.get(cell, -np.inf):
                self.target_cells[cell] = quality
            h2 = np.clip((float(row["h2_wt_percent"]) - 4.2) / 2.8, 0.0, 1.0)
            dh = dh_quality(float(row["delta_H_per_H2"]))
            hv_points.append((h2, dh))
        generation_grid = np.zeros_like(self.hv_occupied)
        generation_hv = self._update_hypervolume_grid(generation_grid, hv_points)
        cumulative_hv = self._update_hypervolume_grid(self.hv_occupied, hv_points)
        self.total_target_hits += generation_hits
        total_cells = max(1, len(BIO_PLATFORM_IDS) * 4)
        current_qd_sum = float(sum(self.target_cells.values()))
        generation_new_cells = int(len(self.target_cells) - previous_cell_count)
        generation_qd_gain = float((current_qd_sum - previous_qd_sum) / total_cells)
        metrics = {
            "generation": int(generation),
            "gnn_evaluations": int(self.total_evaluations),
            "cumulative_unique_molecules": int(len(self.seen_smis)),
            "cumulative_unique_scaffolds": int(len(self.seen_scaffolds)),
            "new_unique_molecules": int(new_rows.shape[0]),
            "new_unique_scaffolds": int(len(unseen_scaffolds)),
            "new_scaffold_rate": float(len(unseen_scaffolds) / max(1, new_rows.shape[0])),
            "generation_target_hits": int(generation_hits),
            "generation_new_target_cells": generation_new_cells,
            "generation_qd_gain": generation_qd_gain,
            "cumulative_target_hits": int(self.total_target_hits),
            "target_hit_rate": float(generation_hits / max(1, new_rows.shape[0])),
            "occupied_target_cells": int(len(self.target_cells)),
            "target_coverage": float(len(self.target_cells) / total_cells),
            "qd_score": float(sum(self.target_cells.values()) / total_cells),
            "generation_hypervolume": generation_hv,
            "cumulative_hypervolume": cumulative_hv,
        }
        self.metrics.append(metrics)
        return metrics

    def is_saturated(self) -> bool:
        """Return True when the target property space has converged."""
        patience = int(
            self.config.get(
                "property_saturation_patience",
                self.config.get("saturation_patience", 50),
            )
        )
        minimum = int(
            self.config.get(
                "property_saturation_min_generations",
                self.config.get("saturation_min_generations", 100),
            )
        )
        if len(self.metrics) < max(patience + 1, minimum):
            return False

        old = self.metrics[-patience - 1]
        new = self.metrics[-1]

        def relative_growth(name):
            old_value = float(old.get(name, 0.0))
            new_value = float(new.get(name, 0.0))
            return (new_value - old_value) / max(abs(old_value), 1e-9)

        new_cells = int(new.get("occupied_target_cells", 0)) - int(
            old.get("occupied_target_cells", 0)
        )
        property_saturated = (
            new_cells <= int(self.config.get("max_new_target_cells", 0))
            and relative_growth("qd_score")
            < float(self.config.get("qd_growth_threshold", 0.001))
            and relative_growth("cumulative_hypervolume")
            < float(self.config.get("hypervolume_growth_threshold", 0.001))
        )
        if not property_saturated:
            return False

        # Structural exhaustion is optional because new scaffolds can continue
        # appearing long after target coverage, QD and hypervolume have converged.
        if bool(self.config.get("require_scaffold_saturation", False)):
            recent = self.metrics[-patience:]
            mean_scaffold_rate = float(
                np.mean([m.get("new_scaffold_rate", 0.0) for m in recent])
            )
            return mean_scaffold_rate < float(
                self.config.get("scaffold_rate_threshold", 0.01)
            )
        return True


BIO_PLATFORM_IDS = (
    "glycerol_like",
    "fatty_acid_like",
    "levulinic_like",
    "furan",
    "furfural_like",
    "furfuryl_alcohol_like",
    "phenol",
    "guaiacol_like",
    "syringol_like",
    "benzofuran",
)
