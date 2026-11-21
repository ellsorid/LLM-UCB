
# ============================================================
# Training Engine
# ============================================================

import copy
import numpy as np

import torch
import torch.nn as nn


# ============================================================
# EMA
# ============================================================

def update_ema(
    ema_model,
    model,
    decay=0.995
):

    with torch.no_grad():

        for ema_p, model_p in zip(
            ema_model.parameters(),
            model.parameters()
        ):

            ema_p.mul_(decay).add_(
                model_p,
                alpha=1.0 - decay
            )

        for ema_b, model_b in zip(
            ema_model.buffers(),
            model.buffers()
        ):

            ema_b.copy_(model_b)


def create_ema_model(model):

    ema_model = copy.deepcopy(
        model
    )

    ema_model.eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    return ema_model


def cpu_state_dict(model):

    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


# ============================================================
# Device transfer
# ============================================================

def move_batch(batch, device):

    M, mask, y = batch

    return (

        M.to(
            device,
            non_blocking=True
        ),

        mask.to(
            device,
            non_blocking=True
        ),

        y.to(
            device,
            non_blocking=True
        )

    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(

    model,

    ema_model,

    loader,

    optimizer,

    criterion,

    scaler,

    device,

    use_amp,

    ema_decay,

    gradient_clip

):

    model.train()

    total_loss = 0.0

    for batch in loader:

        M, mask, y = move_batch(
            batch,
            device
        )

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

            loss = criterion(
                prediction,
                y
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            gradient_clip

        )

        scaler.step(optimizer)

        scaler.update()

        update_ema(
            ema_model,
            model,
            ema_decay
        )

        total_loss += (
            loss.detach().item() * y.size(0)
        )

    return total_loss / len(loader.dataset)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(

    model,

    loader,

    device,

    target_mean,

    target_std,

    use_amp

):

    model.eval()

    predictions = []
    targets = []

    criterion = nn.SmoothL1Loss()

    total_loss = 0.0

    for batch in loader:

        M, mask, y = move_batch(
            batch,
            device
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

            loss = criterion(
                prediction,
                y
            )

        total_loss += (
            loss.item() * y.size(0)
        )

        predictions.extend(
            prediction.float().cpu().numpy()
        )

        targets.extend(
            y.float().cpu().numpy()
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    # Recover physical target units.
    predictions = (
        predictions * target_std + target_mean
    )

    targets = (
        targets * target_std + target_mean
    )

    return (

        total_loss / len(loader.dataset),

        predictions,

        targets

    )
