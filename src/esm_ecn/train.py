import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from esm_ecn.constants import DATA_FOLDER
from esm_ecn.model import LitModel

def setup_experiment(experiment_name, project, resume, wandb_debug, accelerator, epochs):
    # Initialize Wandb logger
    if experiment_name is None:
        wandb_logger = WandbLogger(
            project=project, 
            resume='allow' if resume else 'never',
            mode='offline' if wandb_debug else 'online'
        )

        wandb_experiment = wandb_logger.experiment
        experiment_name = wandb_experiment.id
    else:
        wandb_logger = WandbLogger(
            project=project, 
            id=experiment_name,
            resume='must' if resume else 'never',
            mode='offline' if wandb_debug else 'online'
        )

    # Initialize model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=DATA_FOLDER / 'checkpoints' / experiment_name,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
        save_weights_only=False
    )

    # Initialize a trainer
    if accelerator == "cpu":
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            accelerator="cpu",
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            devices=1
        )
    
    return trainer, experiment_name

def load_best_checkpoint(experiment_name, mlp, focal_loss, lr):
    checkpoint_path = DATA_FOLDER / 'checkpoints' / experiment_name / "best-checkpoint.ckpt"
    model = LitModel.load_from_checkpoint(checkpoint_path, model=mlp)
    model.focal_loss = focal_loss
    checkpoint = torch.load(checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    model.optimizer = optimizer
    return model
