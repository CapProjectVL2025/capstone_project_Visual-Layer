import wandb

def init_wandb(cfg):
    if not cfg.get("wandb", {}).get("use", False):
        return None
    import wandb
    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        config=cfg
    )
    return run
