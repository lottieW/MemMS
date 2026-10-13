import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from merlion.evaluate.anomaly import ScoreType
from sklearn.metrics import average_precision_score, roc_auc_score

from models import ad_predict
from .early_stopping import EarlyStopping


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _format_timing(timing):
    parts = []
    for key, value in timing.items():
        if "batches" in key:
            parts.append(f"{key}={int(value)}")
        else:
            parts.append(f"{key}={value:.3f}s")
    return " | ".join(parts)


def _raw_scores(output):
    """Window reconstruction error and nearest-normal-memory novelty."""
    return torch.stack([output["recon_error"], output["novelty"]], dim=-1)


def _score(output, config, model):
    raw = _raw_scores(output)
    normalized = (raw - model.score_mean) / model.score_std.clamp_min(1e-6)
    return normalized[:, 0] + config.score_memory_weight * normalized[:, 1]


def _with_reconstruction_error(output, x):
    output = dict(output)
    output["recon_error"] = F.mse_loss(
        output["x_recon"], x, reduction="none"
    ).mean(dim=(1, 2))
    return output


def _losses(output, config):
    recon = output["recon_error"].mean()
    prior = output["prior"]
    current = output["current"]
    kl = (prior * (
        prior.clamp_min(1e-8).log() - current.clamp_min(1e-8).log()
    )).sum(-1).mean(1)
    route = (kl * output["confidence"]).mean()
    usage = current.mean((0, 1))
    balance = config.num_experts * usage.square().sum() - 1
    total = recon + config.lambda_aux * (route + 0.1 * balance)
    return total, recon, route, balance


@torch.no_grad()
def refresh_memory(model, data_loader, config, device):
    """Rebuild memory using only training windows labeled normal (target == 0)."""
    model.eval()
    model.memory.reset()
    count = 0
    for x, target in data_loader:
        normal_mask = target.reshape(-1) == 0
        if not normal_mask.any():
            continue
        x = x[normal_mask]

        x = x.float().to(device).permute(0, 2, 1)
        output = _with_reconstruction_error(model(x), x)
        model.memory.update(
            output["z_global"], output["current"], output["recon_error"],
        )
        count += len(x)
    if not count:
        raise ValueError("Cannot refresh memory: no training windows with target == 0")


@torch.no_grad()
def _calibrate_scores(model, data_loader, device):
    """Fit score statistics on held-out validation windows with frozen memory."""
    model.eval()
    count = 0
    total = torch.zeros_like(model.score_mean, dtype=torch.float64)
    squared = torch.zeros_like(total)
    for x, _ in data_loader:
        x = x.float().to(device).permute(0, 2, 1)
        raw = _raw_scores(_with_reconstruction_error(model(x), x)).double()
        total += raw.sum(0)
        squared += raw.square().sum(0)
        count += len(raw)
    if not count:
        raise ValueError("Cannot calibrate scores with an empty validation loader")
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    model.score_mean.copy_(mean)
    model.score_std.copy_(std)


def _continuous_metrics(target, score, config, logger, split):
    target = np.asarray(target).reshape(-1)
    score = np.asarray(score).reshape(-1)
    metrics = {}
    if len(np.unique(target)) < 2:
        logger.warning("%s continuous metrics skipped: target contains only one class", split)
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan
    else:
        try:
            metrics["ROC_AUC"] = roc_auc_score(target, score)
            metrics["PR_AUC"] = average_precision_score(target, score)
        except ValueError as exc:
            logger.warning("%s ROC-AUC and PR-AUC unavailable: %s", split, exc)
            metrics["ROC_AUC"] = np.nan
            metrics["PR_AUC"] = np.nan

    try:
        from vus.metrics import get_metrics

        vus_results = get_metrics(
            score, target, metric="vus", slidingWindow=config.window_size
        )
        metrics["VUS_ROC"] = vus_results["VUS_ROC"]
        metrics["VUS_PR"] = vus_results["VUS_PR"]
    except Exception as exc:
        logger.warning(
            "%s VUS metrics unavailable; falling back to np.nan. This is usually "
            "caused by a vus package/dependency compatibility issue: %s",
            split, exc
        )
        metrics["VUS_ROC"] = np.nan
        metrics["VUS_PR"] = np.nan

    logger.info(
        "%s continuous metrics | ROC-AUC: %.5f | PR-AUC: %.5f | VUS-ROC: %.5f | VUS-PR: %.5f",
        split, metrics["ROC_AUC"], metrics["PR_AUC"], metrics["VUS_ROC"], metrics["VUS_PR"]
    )
    print(
        f"{split} continuous metrics | "
        f"ROC-AUC: {metrics['ROC_AUC']:.5f} | "
        f"PR-AUC: {metrics['PR_AUC']:.5f} | "
        f"VUS-ROC: {metrics['VUS_ROC']:.5f} | "
        f"VUS-PR: {metrics['VUS_PR']:.5f}"
    )
    return metrics


def _run_epoch(model, data_loader, config, device, optimizer=None, collect_details=False):
    is_training = optimizer is not None
    profile_timing = getattr(config, "profile_timing", False)
    model.train(is_training)

    totals, targets, scores = [], [], []
    batch_sizes = []
    timing = defaultdict(float)
    batch_count = 0
    details = {
        key: [] for key in ["score", "recon_error", "novelty", "target"]
    }

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for x_s, target in data_loader:
            batch_count += 1
            step_start = time.perf_counter()
            x_s = x_s.float().to(device).permute(0, 2, 1)
            target = target.long().to(device).reshape(-1)
            if profile_timing:
                _sync_if_cuda(device)
                timing["to_device"] += time.perf_counter() - step_start

            if is_training:
                optimizer.zero_grad()
            forward_start = time.perf_counter()
            output = _with_reconstruction_error(model(x_s), x_s)
            score = _score(output, config, model)
            losses = _losses(output, config)

            if not torch.isfinite(losses[0]):
                raise RuntimeError("Nonfinite loss")
            if profile_timing:
                _sync_if_cuda(device)
                timing["forward_loss"] += time.perf_counter() - forward_start

            if is_training:
                backward_start = time.perf_counter()
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if profile_timing:
                    _sync_if_cuda(device)
                    timing["backward_step"] += time.perf_counter() - backward_start

            collect_start = time.perf_counter()
            totals.append([item.detach().item() for item in losses])
            batch_sizes.append(len(x_s))
            targets.append(target.detach())
            scores.append(score.detach())

            if collect_details:
                details["score"].append(score.detach().cpu())
                for key in ["recon_error", "novelty"]:
                    details[key].append(output[key].detach().cpu())
                details["target"].append(target.detach().cpu())

            if profile_timing:
                timing["collect"] += time.perf_counter() - collect_start

    if not totals:
        raise ValueError("Cannot run an epoch with an empty data loader")

    mean_losses = np.average(np.asarray(totals), axis=0, weights=batch_sizes)
    joined_details = {
        key: torch.cat(value).numpy() for key, value in details.items() if value
    }
    timing["batches"] = float(batch_count)

    return (
        torch.cat(targets).cpu().numpy(),
        torch.cat(scores).cpu().numpy(),
        mean_losses,
        joined_details,
        dict(timing),
    )


def _save_diagnostics(config, idx, seed, trial_id, details, val_metrics, test_metrics):
    if not getattr(config, "save_diagnostics", True):
        return
    study_name = getattr(config, "study_name", "manual")
    path = os.path.join(
        "scores",
        "MemMS",
        config.dataset,
        study_name,
        f"trial_{trial_id}",
        f"seed_{seed}",
    )
    os.makedirs(path, exist_ok=True)
    payload = dict(details)
    payload.update({f"validation_{key}": value for key, value in val_metrics.items()})
    payload.update({f"test_{key}": value for key, value in test_metrics.items()})
    np.savez(os.path.join(path, f"diagnostics_{idx}_{seed}.npz"), **payload)


def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx, seed):
    profile_timing = getattr(config, "profile_timing", False)
    trial_id = str(getattr(config, "trial_id", os.environ.get("CURRENT_TRIAL_ID", "manual")))
    study_name = getattr(config, "study_name", "manual")
    logger = logging.getLogger(f"trial_{trial_id}")
    save_path = os.path.join(
        "best_network",
        "MemMS",
        config.dataset,
        study_name,
        f"trial_{trial_id}",
        f"seed_{seed}",
    )
    os.makedirs(save_path, exist_ok=True)

    early_stopping = EarlyStopping(save_path, idx, patience=config.early_stop)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    timing = defaultdict(float)

    ordered_train_dl = DataLoader(
        train_dl.dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=train_dl.collate_fn,
    )

    for epoch in range(1, config.num_epoch + 1):
        epoch_start = time.perf_counter()

        _, _, train_losses, _, train_timing = _run_epoch(model, train_dl, config, device, model_optimizer)

        refresh_start = time.perf_counter()
        refresh_memory(model, ordered_train_dl, config, device)
        if profile_timing:
            timing["refresh_memory"] += time.perf_counter() - refresh_start

        val_target, val_scores, val_losses, _, val_timing = _run_epoch(model, val_dl, config, device)

        timing["train_epoch_total"] += sum(value for key, value in train_timing.items() if key != "batches")
        timing["val_epoch_total"] += sum(value for key, value in val_timing.items() if key != "batches")
        timing["train_batches"] += train_timing.get("batches", 0.0)
        timing["val_batches"] += val_timing.get("batches", 0.0)
        timing["train_to_device"] += train_timing.get("to_device", 0.0)
        timing["train_forward_loss"] += train_timing.get("forward_loss", 0.0)
        timing["train_backward_step"] += train_timing.get("backward_step", 0.0)
        timing["train_collect"] += train_timing.get("collect", 0.0)
        timing["val_to_device"] += val_timing.get("to_device", 0.0)
        timing["val_forward_loss"] += val_timing.get("forward_loss", 0.0)
        timing["val_collect"] += val_timing.get("collect", 0.0)

        scheduler_start = time.perf_counter()
        scheduler.step(val_losses[1])
        early_stopping(0, 0, 0, -val_losses[1], val_scores, None, model)
        if profile_timing:
            timing["scheduler_early_stopping"] += time.perf_counter() - scheduler_start
            timing["epoch_wall"] += time.perf_counter() - epoch_start

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Epoch: %d | total: %.4f | recon: %.4f | route: %.4f | balance: %.4f",
                epoch, *train_losses
            )

        if early_stopping.early_stop:
            break

    checkpoint = os.path.join(save_path, str(idx).zfill(2) + "_best_network.pth")
    checkpoint_start = time.perf_counter()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if profile_timing:
        _sync_if_cuda(device)
        timing["load_checkpoint"] += time.perf_counter() - checkpoint_start

    _calibrate_scores(model, val_dl, device)
    if getattr(config, "keep_checkpoints", False):
        torch.save(model.state_dict(), checkpoint)

    final_val_start = time.perf_counter()
    val_target, val_scores, _, _, final_val_timing = _run_epoch(model, val_dl, config, device)
    if profile_timing:
        timing["final_val_wall"] += time.perf_counter() - final_val_start
        timing["final_val_batches"] += final_val_timing.get("batches", 0.0)

    final_test_start = time.perf_counter()
    test_target, test_scores, _, details, final_test_timing = _run_epoch(model, test_dl, config, device, collect_details=True)
    if profile_timing:
        timing["final_test_wall"] += time.perf_counter() - final_test_start
        timing["final_test_batches"] += final_test_timing.get("batches", 0.0)

    print(f"Test target shape: {test_target.shape}, \n"
          f"Test scores shape: {test_scores.shape}")

    metrics_start = time.perf_counter()
    test_metrics = _continuous_metrics(test_target, test_scores, config, logger, "Test")
    if profile_timing:
        timing["continuous_metrics"] += time.perf_counter() - metrics_start

    diagnostics_start = time.perf_counter()
    _save_diagnostics(config, idx, seed, trial_id, details, {}, test_metrics)
    if profile_timing:
        timing["save_diagnostics"] += time.perf_counter() - diagnostics_start

    if not getattr(config, "keep_checkpoints", False) and os.path.exists(checkpoint):
        remove_start = time.perf_counter()
        os.remove(checkpoint)
        if profile_timing:
            timing["remove_checkpoint"] += time.perf_counter() - remove_start

    predict_start = time.perf_counter()
    _, val_score, _, _, _ = ad_predict(val_target, val_scores, config.threshold_determine, config.detect_nu)
    test_aff, test_rpa, test_pa, test_pw, predict = ad_predict(
        test_target, test_scores, config.threshold_determine, config.detect_nu
    )
    if profile_timing:
        timing["ad_predict"] += time.perf_counter() - predict_start

    logger.info("Best val RPA F1: %.4f", val_score.f1(ScoreType.RevisedPointAdjusted))
    if profile_timing:
        print(f"[MemMS timing][Trainer entity {idx}] {_format_timing(timing)}")
        model_timing = getattr(model, "timing", None)
        if model_timing:
            print(f"[MemMS timing][Model entity {idx}] {_format_timing(model_timing)}")

    return (
        test_scores,
        test_aff,
        test_rpa,
        test_pa,
        test_pw,
        predict,
        test_metrics["ROC_AUC"],
        test_metrics["PR_AUC"],
        test_metrics["VUS_ROC"],
        test_metrics["VUS_PR"],
    )
