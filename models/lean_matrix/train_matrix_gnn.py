
# ============================================================
# Main Training Script
#
# 12-Channel Dense Matrix GNN
#
# Input:
#   M: [B,N,N,12]
#
# Output:
#   delta_H
#
# Cross-validation:
#   5-fold GroupKFold
#
# GPU:
#   CUDA + AMP
# ============================================================

import os
import json
import math
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from config import (
    DEFAULTS,
    HIDDEN_DIM,
    NUM_GNN_LAYERS,
    SCHEMA_VERSION
)

from matrix_data import (
    load_matrix_archive,
    build_matrix_groups,
    get_matrix,
    MatrixDataset,
    collate_matrix_batch
)

from model import MatrixGIN

from engine import (
    train_one_epoch,
    evaluate,
    create_ema_model,
    cpu_state_dict
)


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Matrix-group audit
# ============================================================

def audit_groups(
    archive,
    groups,
    targets,
    n_samples,
    outdir,
    tolerance=1e-5
):

    group_map = {}

    for i, group in enumerate(groups):

        group_map.setdefault(
            group,
            []
        ).append(i)

    conflicts = []

    audit_rows = []

    for group, indices in group_map.items():

        values = targets[indices]

        target_range = (
            float(np.max(values) - np.min(values))
        )

        if target_range > tolerance:

            conflicts.append({

                "group": group,

                "count": len(indices),

                "target_min": float(np.min(values)),

                "target_max": float(np.max(values)),

                "target_range": target_range

            })

        for i in indices:

            audit_rows.append({

                "matrix_id": archive["ids"][i],

                "source_row": int(
                    archive["source_rows"][i]
                ),

                "group": group,

                "delta_H": float(targets[i]),

                "group_size": len(indices),

                "group_target_range": target_range

            })

    pd.DataFrame(audit_rows).to_csv(

        os.path.join(
            outdir,
            "matrix_group_audit.csv"
        ),

        index=False,

        encoding="utf-8-sig"

    )

    pd.DataFrame(conflicts).to_csv(

        os.path.join(
            outdir,
            "group_label_conflicts.csv"
        ),

        index=False,

        encoding="utf-8-sig"

    )

    print("\n========== GROUP AUDIT ==========")

    print("Total samples:", n_samples)

    print("Unique groups:", len(group_map))

    print("Duplicate-group samples:", sum(
        len(v) for v in group_map.values()
        if len(v) > 1
    ))

    print("Conflicting target groups:", len(conflicts))

    if conflicts:

        print(
            "WARNING: Identical or WL-equivalent "
            "matrix groups have different delta_H labels."
        )

        print(
            "Inspect group_label_conflicts.csv"
        )

    print("=================================")

    return conflicts


# ============================================================
# 3. DataLoader
# ============================================================

def make_loader(

    archive,

    targets,

    indices,

    batch_size,

    shuffle,

    device,

    num_workers

):

    dataset = MatrixDataset(

        archive,

        targets,

        indices

    )

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        collate_fn=collate_matrix_batch,

        pin_memory=(device.type == "cuda"),

        drop_last=False

    )


# ============================================================
# 4. Smoke test
# ============================================================

def smoke_test(

    archive,

    targets,

    device,

    batch_size,

    hidden_dim,

    dropout,

    use_amp

):

    print("\n========== MATRIX SMOKE TEST ==========")

    n = min(
        batch_size,
        len(targets)
    )

    loader = make_loader(

        archive,

        targets,

        np.arange(n),

        batch_size=n,

        shuffle=False,

        device=device,

        num_workers=0

    )

    model = MatrixGIN(

        hidden_dim=hidden_dim,

        dropout=dropout,

        num_layers=NUM_GNN_LAYERS

    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    M, mask, y = next(
        iter(loader)
    )

    M = M.to(device)
    mask = mask.to(device)
    y = y.to(device)

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.autocast(

        device_type=device.type,

        dtype=torch.float16,

        enabled=use_amp

    ):

        prediction = model(
            M,
            mask
        )

        loss = nn.SmoothL1Loss()(
            prediction,
            y
        )

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    print("Input matrix:", tuple(M.shape))

    print("Atom mask:", tuple(mask.shape))

    print("Prediction:", tuple(prediction.shape))

    print("Loss:", float(loss.detach().cpu()))

    print("Forward: PASS")

    print("Backward: PASS")

    print("Device:", device)

    print("AMP:", use_amp)

    print("========================================")

    del model
    del optimizer

    if device.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# 5. Five-fold cross-validation
# ============================================================

def train_cv(

    archive,

    targets,

    groups,

    args,

    device

):

    unique_groups = np.unique(groups)

    if len(unique_groups) < args.n_splits:

        raise ValueError(
            "Insufficient unique groups for GroupKFold"
        )

    splitter = GroupKFold(
        n_splits=args.n_splits
    )

    checkpoint_dir = os.path.join(
        args.outdir,
        "checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    use_amp = (
        args.amp and
        device.type == "cuda"
    )

    oof_predictions = np.full(
        len(targets),
        np.nan
    )

    fold_results = []

    print("\n========== TRAINING CONFIGURATION ==========")

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory / 1024**3,
                2
            ),
            "GB"
        )

    print("Input: 12-channel molecular matrix")

    print("Architecture: Dense Matrix GIN")

    print("GIN layers:", NUM_GNN_LAYERS)

    print("Hidden dimension:", args.hidden_dim)

    print("Batch size:", args.batch_size)

    print("Epochs:", args.epochs)

    print("CV folds:", args.n_splits)

    print("AMP:", use_amp)

    print("=============================================")

    for fold, (train_idx, val_idx) in enumerate(

        splitter.split(

            np.zeros(len(targets)),

            targets,

            groups

        ),

        start=1

    ):

        print(
            f"\n========== FOLD {fold}/{args.n_splits} =========="
        )

        # ----------------------------------------------------
        # Target normalization
        # Training fold only.
        # ----------------------------------------------------

        target_mean = float(
            np.mean(targets[train_idx])
        )

        target_std = float(
            np.std(targets[train_idx])
        )

        if target_std < 1e-12:
            target_std = 1.0

        normalized_targets = (

            targets - target_mean

        ) / target_std

        # ----------------------------------------------------
        # Loaders
        # ----------------------------------------------------

        train_loader = make_loader(

            archive,

            normalized_targets,

            train_idx,

            args.batch_size,

            True,

            device,

            args.num_workers

        )

        val_loader = make_loader(

            archive,

            normalized_targets,

            val_idx,

            args.batch_size,

            False,

            device,

            args.num_workers

        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = MatrixGIN(

            hidden_dim=args.hidden_dim,

            dropout=args.dropout,

            num_layers=NUM_GNN_LAYERS

        ).to(device)

        ema_model = create_ema_model(
            model
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(

            model.parameters(),

            lr=args.lr,

            weight_decay=args.weight_decay

        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            patience=4,

            factor=0.5,

            min_lr=1e-6

        )

        criterion = nn.SmoothL1Loss()

        scaler = torch.amp.GradScaler(

            "cuda",

            enabled=use_amp

        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        best_r2 = -float("inf")

        best_state = None

        best_epoch = 0

        patience_counter = 0

        # ----------------------------------------------------
        # Epoch loop
        # ----------------------------------------------------

        for epoch in range(

            1,

            args.epochs + 1

        ):

            # Linear warmup
            if epoch <= args.warmup_epochs:

                lr = (

                    args.lr *

                    epoch /

                    max(args.warmup_epochs, 1)

                )

                for group in optimizer.param_groups:

                    group["lr"] = lr

            # Training
            train_loss = train_one_epoch(

                model=model,

                ema_model=ema_model,

                loader=train_loader,

                optimizer=optimizer,

                criterion=criterion,

                scaler=scaler,

                device=device,

                use_amp=use_amp,

                ema_decay=args.ema_decay,

                gradient_clip=args.gradient_clip

            )

            # Validation
            val_loss, val_pred, val_true = evaluate(

                ema_model,

                val_loader,

                device,

                target_mean,

                target_std,

                use_amp

            )

            val_r2 = r2_score(
                val_true,
                val_pred
            )

            val_rmse = math.sqrt(
                mean_squared_error(
                    val_true,
                    val_pred
                )
            )

            val_mae = mean_absolute_error(
                val_true,
                val_pred
            )

            if epoch > args.warmup_epochs:

                scheduler.step(val_r2)

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(

                f"Epoch {epoch:03d} | "

                f"LR {current_lr:.2e} | "

                f"Train Loss {train_loss:.5f} | "

                f"Val Loss {val_loss:.5f} | "

                f"R2 {val_r2:.4f} | "

                f"RMSE {val_rmse:.4f} | "

                f"MAE {val_mae:.4f}"

            )

            # Best model
            if (

                np.isfinite(val_r2)

                and val_r2 > best_r2

            ):

                best_r2 = val_r2

                best_epoch = epoch

                best_state = cpu_state_dict(
                    ema_model
                )

                patience_counter = 0

            else:

                patience_counter += 1

            if (
                patience_counter >=
                args.patience
            ):

                print(
                    "Early stopping triggered."
                )

                break

        if best_state is None:

            raise RuntimeError(
                f"No valid checkpoint in fold {fold}"
            )

        # ----------------------------------------------------
        # Restore best model
        # ----------------------------------------------------

        ema_model.load_state_dict(
            best_state
        )

        final_loss, final_pred, final_true = evaluate(

            ema_model,

            val_loader,

            device,

            target_mean,

            target_std,

            use_amp

        )

        final_r2 = r2_score(
            final_true,
            final_pred
        )

        final_rmse = math.sqrt(
            mean_squared_error(
                final_true,
                final_pred
            )
        )

        final_mae = mean_absolute_error(
            final_true,
            final_pred
        )

        oof_predictions[val_idx] = final_pred

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(

            checkpoint_dir,

            f"fold_{fold}_best.pth"

        )

        torch.save({

            "model_state_dict": best_state,

            "fold": fold,

            "best_epoch": best_epoch,

            "target_mean": target_mean,

            "target_std": target_std,

            "schema_version": SCHEMA_VERSION,

            "channel_names": archive["channel_names"].tolist(),

            "hidden_dim": args.hidden_dim,

            "num_gnn_layers": NUM_GNN_LAYERS,

            "config": vars(args)

        }, checkpoint_path)

        fold_results.append({

            "fold": fold,

            "train_size": len(train_idx),

            "validation_size": len(val_idx),

            "best_epoch": best_epoch,

            "best_val_r2": best_r2,

            "final_val_r2": final_r2,

            "final_val_rmse": final_rmse,

            "final_val_mae": final_mae,

            "checkpoint": checkpoint_path

        })

        print("\nFOLD RESULT")

        print("Best epoch:", best_epoch)

        print("R2:", final_r2)

        print("RMSE:", final_rmse)

        print("MAE:", final_mae)

        del model
        del ema_model
        del optimizer

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # OOF metrics
    # --------------------------------------------------------

    if np.isnan(oof_predictions).any():

        raise RuntimeError(
            "Incomplete OOF predictions"
        )

    overall_r2 = r2_score(
        targets,
        oof_predictions
    )

    overall_rmse = math.sqrt(
        mean_squared_error(
            targets,
            oof_predictions
        )
    )

    overall_mae = mean_absolute_error(
        targets,
        oof_predictions
    )

    # --------------------------------------------------------
    # Save OOF
    # --------------------------------------------------------

    oof_df = pd.DataFrame({

        "matrix_id": archive["ids"][:len(targets)],

        "source_row": archive["source_rows"][:len(targets)],

        "true_delta_H": targets,

        "predicted_delta_H": oof_predictions,

        "absolute_error": np.abs(
            targets - oof_predictions
        ),

        "matrix_group": groups

    })

    oof_df.to_csv(

        os.path.join(
            args.outdir,
            "oof_predictions.csv"
        ),

        index=False,

        encoding="utf-8-sig"

    )

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_df.to_csv(

        os.path.join(
            args.outdir,
            "fold_metrics.csv"
        ),

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "experiment": "12-Channel Dense Matrix GNN",

        "schema_version": SCHEMA_VERSION,

        "input": "M_data [N,N,12]",

        "total_samples": len(targets),

        "unique_groups": len(
            np.unique(groups)
        ),

        "device": str(device),

        "gpu": (

            torch.cuda.get_device_name(0)

            if device.type == "cuda"

            else None

        ),

        "hidden_dim": args.hidden_dim,

        "gin_layers": NUM_GNN_LAYERS,

        "batch_size": args.batch_size,

        "n_splits": args.n_splits,

        "overall_oof_r2": float(overall_r2),

        "overall_oof_rmse": float(overall_rmse),

        "overall_oof_mae": float(overall_mae),

        "mean_fold_r2": float(
            fold_df["final_val_r2"].mean()
        ),

        "std_fold_r2": float(
            fold_df["final_val_r2"].std()
        ),

        "mean_fold_rmse": float(
            fold_df["final_val_rmse"].mean()
        ),

        "matrix_input_only": True,

        "smiles_model_input": False,

        "graph_conversion": False,

        "dual_molecule_encoding": False

    }

    with open(

        os.path.join(
            args.outdir,
            "training_summary.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== TRAINING COMPLETE ==========")

    print("OOF R2:", overall_r2)

    print("OOF RMSE:", overall_rmse)

    print("OOF MAE:", overall_mae)

    print("========================================")

    return summary


# ============================================================
# 6. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matrix",
        required=True
    )

    parser.add_argument(
        "--target-col",
        default="delta_H"
    )

    parser.add_argument(
        "--outdir",
        default="matrix_gnn_12ch_results"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULTS["batch_size"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULTS["epochs"]
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULTS["learning_rate"]
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULTS["weight_decay"]
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULTS["dropout"]
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=DEFAULTS["ema_decay"]
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULTS["warmup_epochs"]
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULTS["early_stop_patience"]
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULTS["n_splits"]
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=DEFAULTS["gradient_clip"]
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULTS["num_workers"]
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    parser.add_argument(
        "--amp",
        action="store_true"
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true"
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    parser.add_argument(
        "--strict-group-labels",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"]
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        device = torch.device("cuda")

    elif args.device == "cpu":

        device = torch.device("cpu")

    else:

        device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"

        )

    print("Device:", device)

    # --------------------------------------------------------
    # Load matrix archive
    # --------------------------------------------------------

    archive = load_matrix_archive(

        args.matrix,

        target_col=args.target_col

    )

    n_total = len(
        archive["ids"]
    )

    if args.max_samples is not None:

        n_samples = min(
            args.max_samples,
            n_total
        )

    else:

        n_samples = n_total

    if n_samples < 2:

        raise ValueError(
            "At least two samples are required"
        )

    targets = archive["targets"][:n_samples]

    # --------------------------------------------------------
    # Matrix-native grouping
    # --------------------------------------------------------

    groups = build_matrix_groups(
        archive,
        n_samples
    )

    conflicts = audit_groups(

        archive,

        groups,

        targets,

        n_samples,

        args.outdir

    )

    if conflicts and args.strict_group_labels:

        raise ValueError(
            "Conflicting targets detected within matrix groups. "
            "Inspect group_label_conflicts.csv."
        )

    print("\n========== DATA SUMMARY ==========")

    print("Samples:", n_samples)

    print("Matrix shape: [N,N,12]")

    print("Channels:", 12)

    print("Unique groups:", len(np.unique(groups)))

    print("Dual encoding: OFF")

    print("SMILES input: OFF")

    print("Graph conversion: OFF")

    print("==================================")

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    if args.preflight_only:

        print(
            "\nPreflight successful. Training not started."
        )

        return

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = (
        args.amp and
        device.type == "cuda"
    )

    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------

    if args.smoke_test:

        smoke_test(

            archive,

            targets,

            device,

            args.batch_size,

            args.hidden_dim,

            args.dropout,

            use_amp

        )

        return

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    train_cv(

        archive=archive,

        targets=targets,

        groups=groups,

        args=args,

        device=device

    )


if __name__ == "__main__":

    main()
