from torch.utils.tensorboard.writer import SummaryWriter

class TrainLogger:
    def __init__(self, cfg: dict):
        log_dir = "runs"
        run_name = cfg.get("run_name", None)
        self.tb = SummaryWriter(log_dir=log_dir)
        self.use_wandb = cfg.get("use_wandb", False)
        self.wandb_on = False
        self.wandb = None
        if self.use_wandb:
            try:
                self.wandb = __import__('wandb')
                self.wandb.init(project=cfg.get("wandb_project", "nucdiff"), name=run_name, config=cfg)
                self.wandb_on = True
            except ImportError:
                self.wandb = None
    def log_scalar(self, tag, value, step):
        self.tb.add_scalar(tag, value, step)
        if self.wandb_on and self.wandb is not None:
            self.wandb.log({tag: value}, step=step)
    def close(self):
        self.tb.close()
        if self.wandb_on and self.wandb is not None:
            self.wandb.finish() 