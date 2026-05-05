import torch
from pathlib import Path


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, strict=True, init_only=False):
    """init_only=True: load only model weights (filter shape mismatches), ignore optimizer/scheduler/epoch."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    if not strict or init_only:
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[ckpt] skipped {len(skipped)} mismatched keys (e.g. {skipped[:3]})")
        if missing:
            print(f"[ckpt] missing {len(missing)} keys (will be reinitialized)")
    else:
        model.load_state_dict(sd)
    if init_only:
        return 0, -1.0
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[ckpt] optimizer state mismatch — fresh optimizer: {e}")
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, KeyError) as e:
            print(f"[ckpt] scheduler state mismatch — fresh scheduler: {e}")
    return ckpt.get("epoch", 0), ckpt.get("best_metric", -1.0)
