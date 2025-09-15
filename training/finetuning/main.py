import argparse
import logging
import os
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.tuner.tuning import Tuner
import datetime
import dataset
import models
from train_model import RareTrainModel
from pathlib import Path

base_cfg = {
    "training": {
        "seed": 1337,
        "num_epochs": 100,
        "lr_finder": True,
        "min_lr": 1e-7,
        "max_lr": 1e-3,
        "learning_rate": 1e-4,
        "weight_decay": 0,
        "early_stopping": 30,
        "weighted_sampler": True,
        "weighted_loss": True,
        "precision": "bf16-mixed",
        "batch_size": 8,
    },
    "model": {
        "model_name": "dinov3_vitl16",
        "weight_path": "your_weights_here",
    },
    "paths": {
        "output_dir": "outputs",
        "resource_path": "data"
    },
    "dataset": {
        "num_workers": 4,
        "transforms": {
            "mode": "top1",
            "height": 224,
            "width": 224
        },
        "split": "fold_0",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with configurable paths and dataset options')

    # Path configurations
    parser.add_argument('--output-dir', type=str, help='Output directory for saving results')
    parser.add_argument('--resource-path', type=str, help='Path to resources')

    # Dataset configurations
    parser.add_argument('--dataset-split', type=str, choices=['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'],
                        help='Dataset split to use')
    parser.add_argument('--transforms-mode', type=str, choices=['top1', 'top2', 'top3', 'top4'],
                        help='Transformation mode for the dataset')

    parser.add_argument('--weight-path', type=str, help='Path to dino weights')
    parser.add_argument('--epochs', type=int, help='(Maximum) number of epochs to train for. Default is 100.')

    return parser.parse_args()


def main(cfg):
    # Set seeds for reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    import random
    random.seed(cfg.training.seed)

    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.use_deterministic_algorithms(True)

    exp_id = f"{cfg.model.model_name}_{cfg.dataset.split}"
    train_dataloader, val_dataloader, test_dataloader = dataset.load_dataloaders(cfg, dataset.RAREDiseaseRecognition)
    data_module = dataset.PLDataModule(train_dataloader, val_dataloader, test_dataloader, None)

    class_weights = None
    if cfg.training.weighted_loss:
        class_counts = train_dataloader.dataset.get_class_counts()
        class_weights = torch.FloatTensor([1.0/class_counts[0], 1.0/class_counts[1]])

    if "dinov3" in cfg.model.model_name:
        base_model = models.DinoV3(config=cfg, lora=True)

    model = RareTrainModel(cfg, "classification", base_model, 2, val_dataloader.dataset.dataset.data_df, class_weights=class_weights)

    output_dir = Path(cfg.paths.output_dir) / f"RARE_{cfg.model.model_name}" / cfg.dataset.transforms.mode
    output_dir_conf = output_dir / "configs"
    output_dir_model = output_dir / "models"

    os.makedirs(output_dir_conf, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    callbacks = []
    if cfg.training.early_stopping > 0:
        # Create an EarlyStopping callback
        early_stop_callback = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            min_delta=0.00,      # Minimum change in the monitored metric to qualify as an improvement
            patience=cfg.training.early_stopping,         # Number of epochs with no improvement after which training will be stopped
            verbose=True,       # Enable verbose mode
            mode='min'           # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
        )
        callbacks.append(early_stop_callback)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir_model,
        filename=exp_id,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        accelerator="gpu",  # Automatically detect GPU if available
        devices=1,
        callbacks=callbacks,
        precision=cfg.training.precision,
        deterministic=True,
        enable_checkpointing=True,
        log_every_n_steps=10  # Logs nach jedem Batch
    )
    if cfg.training.lr_finder:
        trainer_tune = pl.Trainer(
            max_epochs=cfg.training.num_epochs,
            accelerator="auto",
            devices=1,
            precision=cfg.training.precision,
            deterministic=True,
            enable_checkpointing=False,
            log_every_n_steps=10
        )

        tuner = Tuner(trainer_tune)
        try:
            lr_finder = tuner.lr_find(
                model=model,
                train_dataloaders=data_module.train_dataloader(),
                val_dataloaders=data_module.val_dataloader(),
                min_lr=cfg.training.min_lr,
                max_lr=cfg.training.max_lr,
                num_training=100,
            )
        except _TunerExitException:
            logging.info("LR Finder failed successfully")

    model.weight_decay = model.learning_rate / 10

    trainer.fit(model, data_module)
    if len(test_dataloader.dataset) > 0:
        trainer.test(model, dataloaders=test_dataloader, verbose=True)

    # Save the final model
    torch.save(model.state_dict(), output_dir_model / f"final_{exp_id}.pth")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Create base configuration
    cfg = OmegaConf.create(base_cfg)

    # Update configuration with command line arguments
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.resource_path:
        cfg.paths.resource_path = args.resource_path
    if args.dataset_split:
        cfg.dataset.split = args.dataset_split
    if args.transforms_mode:
        cfg.dataset.transforms.mode = args.transforms_mode
    if args.weight_path:
        cfg.model.weight_path = args.weight_path
    if args.epochs:
        cfg.training.num_epochs = args.epochs

    main(cfg)
