from __future__ import annotations

import argparse
import os
from multiprocessing import freeze_support
from pathlib import Path
from enthalpy_window import DH_LOW, DH_TARGET, DH_HIGH, DH_EDGES, dh_quality

import pandas as pd


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import utils  # noqa: E402

CONFIG = {
    'pop_size': 1000,
    'generations': 1000,
    'mutation_rate': 0.22,
    'min_mutation_rate': 0.12,
    'max_mutation_rate': 0.45,
    'stagnation_patience': 8,
    'stagnation_mutation_boost': 0.12,
    'elite_fraction': 0.005,
    'num_parents': 1400,
    'max_mol_weight': 260.0,
    'sa_max_generation': 4.0,
    'immigrant_fraction': 0.55,
    'refresh_fraction': 0.35,
    'num_fittest': 100000,
    'target_archive_size': 0,
    'switch_archive_size': 10000,
    'min_candidate_pool_size': 20000,
    'max_candidate_pool_size': 100000,
    'archive_oversample_factor': 2.0,
    'target_immigrant_fraction': 0.15,
    'child_oversample_factor': 1.5,
    'target_child_oversample_factor': 1.5,
    'immigrant_oversample_factor': 2.0,
    'target_immigrant_oversample_factor': 2.0,
    'crossover_attempt_factor': 12,
    'max_crossover_attempts': 200000,
    'postprocess_every': 100,
    'stats_every': 100,
    'pareto_export_rows': 100000,
    'runtime_postprocess': True,
    'runtime_stats_export': True,
    'platform_guided': True,
    'preserve_platform_in_mutation': True,
    'same_platform_crossover': True,
    'retain_parent_scaffold_in_crossover': True,
    'platform_balanced_immigrants': True,
    'survivor_screening': {
        'selection_ratio': 0.10,
        'selection_by_ratio': True,
        'dh_target': DH_TARGET,
        'dh_window_low': DH_LOW,
        'dh_window_high': DH_HIGH,
        'n_clusters': 12,
        'representative_mode': 'best',
    },
    'keep_core_ratio': 0.60,
    'keep_cluster_ratio': 0.25,
    'keep_platform_ratio': 0.10,
    'guided_mutant_fraction': 0.25,
    'guided_mutant_oversample_factor': 1.3,
    'adaptive_mode': 'ucb',
    'operator_ucb_beta': 0.8,
    'operator_random_fraction': 0.10,
    'operator_warmup_generations': 5,
    'operator_temperature': 0.35,
    'full_model_mode': False,
    'keep_qd_ratio': 0.30,
    'frontier_parent_weight': 0.75,
    'chemical_qd_mode': False,
    'keep_chemical_qd_ratio': 0.20,
    'chemical_parent_weight': 0.60,
    'chemical_behavior_cells': 96,
    'chemical_local_archive_k': 3,
    'chemical_novelty_k': 10,
    'chemical_novelty_sample_limit': 600,
    'allowed_heavy_atoms': 'C,O,N',
    'allowed_derivatization_atoms': 'N',
    'max_pending_gnn_per_generation': 3000,
    'adaptive_output_dir': 'results/adaptive_ga/ucb_seed42',
    'random_seed': 42,
    'stop_at_archive_size': False,
    'saturation_enabled': True,
    'saturation_min_generations': 100,
    'saturation_patience': 50,
    'coverage_growth_threshold': 0.005,
    'qd_growth_threshold': 0.001,
    'scaffold_rate_threshold': 0.01,
    'hypervolume_growth_threshold': 0.001,
    'max_new_target_cells': 0,
    'require_scaffold_saturation': False,
    'run_selection_redraw': False,
    'compute_full_hydrogenation': False,
    'uncertainty_penalty_enabled': False,
    'uncertainty_penalty_lambda': 0.05,
    'uncertainty_threshold': 20.0,
    'uncertainty_soft_width': 5.0,
    'uncertainty_hard_penalty': 0.0,
    'lohc_probability_scoring': False,
    'lohc_probability_weight': 8.0,
    'lohc_h2_min': 5.5,
    'lohc_dh_low': DH_LOW,
    'lohc_dh_high': DH_HIGH,
    'lohc_dh_edge_width': 4.0,
    'lohc_sa_max': 4.0,
    'lohc_mw_max': 260.0,
}



def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(CONFIG)
    cfg["survivor_screening"] = dict(CONFIG["survivor_screening"])
    cfg["pop_size"] = int(args.population)
    cfg["generations"] = int(args.generations)
    cfg["num_parents"] = int(args.parents or args.population)
    cfg["target_archive_size"] = int(args.target_archive_size)
    cfg["runtime_postprocess"] = bool(args.runtime_postprocess)
    cfg["adaptive_mode"] = str(args.adaptive_mode)
    cfg["random_seed"] = int(args.seed)
    cfg["stop_at_archive_size"] = bool(args.stop_at_archive_size)
    cfg["full_model_mode"] = bool(args.full_model_mode)
    cfg["keep_qd_ratio"] = float(args.keep_qd_ratio)
    cfg["frontier_parent_weight"] = float(args.frontier_parent_weight)
    cfg["chemical_qd_mode"] = bool(args.chemical_qd_mode)
    cfg["keep_chemical_qd_ratio"] = float(args.keep_chemical_qd_ratio)
    cfg["chemical_parent_weight"] = float(args.chemical_parent_weight)
    cfg["chemical_behavior_cells"] = int(args.chemical_behavior_cells)
    cfg["chemical_local_archive_k"] = int(args.chemical_local_archive_k)
    cfg["chemical_novelty_k"] = int(args.chemical_novelty_k)
    cfg["chemical_novelty_sample_limit"] = int(args.chemical_novelty_sample_limit)
    cfg["max_pending_gnn_per_generation"] = int(args.max_pending_gnn_per_generation)
    cfg["uncertainty_penalty_enabled"] = bool(args.uncertainty_penalty)
    cfg["uncertainty_penalty_lambda"] = float(args.uncertainty_penalty_lambda)
    cfg["uncertainty_threshold"] = float(args.uncertainty_threshold)
    cfg["uncertainty_soft_width"] = float(args.uncertainty_soft_width)
    cfg["uncertainty_hard_penalty"] = float(args.uncertainty_hard_penalty)
    cfg["lohc_probability_scoring"] = bool(args.lohc_probability_scoring)
    cfg["lohc_probability_weight"] = float(args.lohc_probability_weight)
    cfg["lohc_h2_min"] = float(args.lohc_h2_min)
    cfg["lohc_dh_low"] = float(args.lohc_dh_low)
    cfg["lohc_dh_high"] = float(args.lohc_dh_high)
    cfg["lohc_dh_edge_width"] = float(args.lohc_dh_edge_width)
    cfg["lohc_sa_max"] = float(args.lohc_sa_max)
    cfg["lohc_mw_max"] = float(args.lohc_mw_max)
    cfg["adaptive_output_dir"] = str(Path("results") / "adaptive_ga" / args.run_label)
    return cfg


def save_hyperparams(cfg: dict, out_dir: Path = ROOT / "results" / "frames") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {key: value for key, value in cfg.items() if key != "survivor_screening"}
    rows["screening_config"] = str(cfg["survivor_screening"])
    pd.DataFrame.from_dict(rows, orient="index", columns=["value"]).to_csv(
        out_dir / "hyperparams.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GA + GNN molecular generation demo."
    )
    parser.add_argument("--population", type=int, default=CONFIG["pop_size"])
    parser.add_argument("--generations", type=int, default=CONFIG["generations"])
    parser.add_argument("--parents", type=int, default=0)
    parser.add_argument(
        "--target-archive-size",
        type=int,
        default=CONFIG["target_archive_size"],
    )
    parser.add_argument(
        "--runtime-postprocess",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["runtime_postprocess"],
        help="Also export intermediate selection results during the run.",
    )
    parser.add_argument(
        "--adaptive-mode",
        choices=["random", "mean", "ucb"],
        default=CONFIG["adaptive_mode"],
        help="Mutation-operator policy used for ablation experiments.",
    )
    parser.add_argument("--run-label", default="ucb_seed42")
    parser.add_argument("--seed", type=int, default=CONFIG["random_seed"])
    parser.add_argument(
        "--full-model-mode",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["full_model_mode"],
        help="Enable frontier-guided MAP-Elites/QD survivor selection and GNN prefiltering.",
    )
    parser.add_argument("--keep-qd-ratio", type=float, default=CONFIG["keep_qd_ratio"])
    parser.add_argument("--frontier-parent-weight", type=float, default=CONFIG["frontier_parent_weight"])
    parser.add_argument(
        "--chemical-qd-mode",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["chemical_qd_mode"],
        help="Enable chemical behavior-aware QD archive with fingerprint novelty and local Pareto competition.",
    )
    parser.add_argument("--keep-chemical-qd-ratio", type=float, default=CONFIG["keep_chemical_qd_ratio"])
    parser.add_argument("--chemical-parent-weight", type=float, default=CONFIG["chemical_parent_weight"])
    parser.add_argument("--chemical-behavior-cells", type=int, default=CONFIG["chemical_behavior_cells"])
    parser.add_argument("--chemical-local-archive-k", type=int, default=CONFIG["chemical_local_archive_k"])
    parser.add_argument("--chemical-novelty-k", type=int, default=CONFIG["chemical_novelty_k"])
    parser.add_argument("--chemical-novelty-sample-limit", type=int, default=CONFIG["chemical_novelty_sample_limit"])
    parser.add_argument(
        "--max-pending-gnn-per-generation",
        type=int,
        default=CONFIG["max_pending_gnn_per_generation"],
        help="Maximum unevaluated candidates sent to GNN each generation in full-model mode; 0 disables the budget.",
    )
    parser.add_argument(
        "--stop-at-archive-size",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["stop_at_archive_size"],
        help="Treat target archive size as a hard stop instead of a reporting target.",
    )
    parser.add_argument(
        "--uncertainty-penalty",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["uncertainty_penalty_enabled"],
        help="Penalize GA fitness by the 5-fold GNN ensemble disagreement for delta H.",
    )
    parser.add_argument("--uncertainty-penalty-lambda", type=float, default=CONFIG["uncertainty_penalty_lambda"])
    parser.add_argument("--uncertainty-threshold", type=float, default=CONFIG["uncertainty_threshold"])
    parser.add_argument("--uncertainty-soft-width", type=float, default=CONFIG["uncertainty_soft_width"])
    parser.add_argument("--uncertainty-hard-penalty", type=float, default=CONFIG["uncertainty_hard_penalty"])
    parser.add_argument(
        "--lohc-probability-scoring",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["lohc_probability_scoring"],
        help="Add multiplicative cheap-descriptor LOHC probability score to GA fitness.",
    )
    parser.add_argument("--lohc-probability-weight", type=float, default=CONFIG["lohc_probability_weight"])
    parser.add_argument("--lohc-h2-min", type=float, default=CONFIG["lohc_h2_min"])
    parser.add_argument("--lohc-dh-low", type=float, default=CONFIG["lohc_dh_low"])
    parser.add_argument("--lohc-dh-high", type=float, default=CONFIG["lohc_dh_high"])
    parser.add_argument("--lohc-dh-edge-width", type=float, default=CONFIG["lohc_dh_edge_width"])
    parser.add_argument("--lohc-sa-max", type=float, default=CONFIG["lohc_sa_max"])
    parser.add_argument("--lohc-mw-max", type=float, default=CONFIG["lohc_mw_max"])
    return parser.parse_args()


def main() -> float:
    args = parse_args()
    cfg = build_config(args)
    save_hyperparams(cfg, ROOT / cfg["adaptive_output_dir"])
    minutes = utils.run_simplified_workflow(cfg)
    print(f"finished in {minutes:.2f} min")
    return minutes


if __name__ == "__main__":
    freeze_support()
    main()
