import wandb
from omegaconf import OmegaConf

class Logger:
    """
        Class to initialize logger object for writing down experiment resulst to wandb.
    """
    def __init__(self, config):

        self.verbose = config["logger"].get("verbose", False)
        self.run = wandb.init(
            project=config["logger"]["project"],
            entity=config["logger"]["entity"],
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=config["logger"].get("tags", None),
            notes=config["logger"].get("notes", None),
            group=config["logger"].get("group", None),
            mode=config["logger"].get("mode", None),
            reinit=True,
            )
        self.define_metrics()

    def log(self, data, step=None, commit=False):
        wandb.log(data, step=step, commit=commit)

    def log_item(self, tag, val, step=None, commit=False, **kwargs):
        self.log({tag: val, **kwargs}, step=step, commit=commit)
        if self.verbose:
            print(f"{tag}: {val}")

    def commit(self):
        self.log({}, commit=True)

    def log_xp_matrix(self, tag, mat, step=None, columns=None, rows=None, commit=False, **kwargs):
        if rows is None:
            rows = [str(i) for i in range(mat.shape[0])]
        if columns is None:
            columns = [str(i) for i in range(mat.shape[1])]
        tab = wandb.Table(
                columns=columns,
                data=mat,
                rows=rows
                )
        wandb.log({tag: tab, **kwargs}, step=step, commit=commit)

    def define_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("checkpoint")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="checkpoint")