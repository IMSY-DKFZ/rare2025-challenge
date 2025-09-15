import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as pl
from metrics import compute_challenge_metrics

class RareTrainModel(pl.LightningModule):
    def __init__(
            self,
            args,
            task_type,
            adapted_model,
            num_classes,
            val_dataframe = pd.DataFrame(),
            class_weights=None,
    ):
        super().__init__()
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S-%f')
        self.save_hyperparameters(ignore=['adapted_model'])
        self.args = args
        self.task = task_type
        self.adapted_model = adapted_model
        self.feature_dim = adapted_model.feat_dim
        self.num_classes = num_classes
        self.predictions = []
        self.val_df = val_dataframe

        self.criterion = self._init_criterion(class_weights=class_weights)
        self.learning_rate = args.training.learning_rate
        self.weight_decay = args.training.weight_decay

        # For triplet_classification fold tracking
        split = args.dataset.split if "split" in args.dataset else "custom"
        self.fold = split

    def _init_criterion(self, class_weights):
        return nn.CrossEntropyLoss(weight=class_weights).cuda()

    def forward(self, x):
        return self.adapted_model(x)

    def _step_common(self, batch, batch_idx, step_loss_key):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        if step_loss_key in ["val_loss", "test_loss"]:
            self.predictions.append(logits.detach().cpu())
        self.log(step_loss_key, loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step_common(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        return self._step_common(batch, batch_idx, "test_loss")


    def on_train_epoch_end(self):
        self.predictions = []

    def on_validation_epoch_end(self):
        if self.predictions:
            all_pred = torch.cat(self.predictions, dim=0)
            y_true = np.array(self.val_df["target"].to_list(), dtype=np.uint8)[:len(all_pred)]
            if len(all_pred) != len(y_true):
                print(f"Warning: Prediction length ({len(all_pred)}) doesn't match target length ({len(y_true)})")
                all_pred = all_pred[:len(y_true)]  # Truncate to match target length
            y_pred = nn.functional.softmax(all_pred, dim=1)[:, 1].float().cpu().numpy()
            sample_metrics = compute_challenge_metrics(y_true, y_pred)
            print("\nSingle computation (no bootstrap):")
            for metric, value in sample_metrics.items():
                print(f"  {metric}: {value:.4f}")
                self.log(f"val_{metric}", value)
        self.predictions = []

    def on_test_epoch_end(self):
        if self.predictions:
            all_pred = torch.cat(self.predictions)
            y_true = np.array(self.val_df["target"].to_list(), dtype=np.uint8)
            y_pred = nn.functional.softmax(all_pred, dim=1)[:, 1].float().cpu().numpy()
            sample_metrics = compute_challenge_metrics(y_true, y_pred)
            print("\nSingle computation (no bootstrap):")
            for metric, value in sample_metrics.items():
                print(f"  {metric}: {value:.4f}")
                self.log(f"test_{metric}", value)
            self._save_logits("test")
        self.predictions = []

    def _save_logits(self, stage):
        all_pred = torch.cat(self.predictions).float().numpy()
        df = pd.DataFrame(all_pred, columns=[f"logits_{i}" for i in range(all_pred.shape[1])])
        self.output_df = pd.concat([self.val_df, df], axis=1)
        self.output_path = Path(self.args.paths.output_dir) / f"RARE_{self.args.model.model_name}" / self.args.dataset.transforms.mode
        os.makedirs(self.output_path, exist_ok=True)
        self.output_df.to_csv(f"{self.output_path}/{stage}_predictions_{self.fold}.csv", index=False)
        self.predictions = []

    def configure_optimizers(self):
        params_to_optimize = []
        if self.adapted_model.head is not None:
            params_to_optimize.extend(self.adapted_model.head.parameters())
        for name, param in self.adapted_model.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer