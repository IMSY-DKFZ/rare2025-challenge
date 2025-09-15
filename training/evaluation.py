"""Validation and evaluation functions."""

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import logging

from validation.metrics import prevalence_corrected_ppv_at_90_recall_gpu

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, device):
        self.device = device

    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch and return predictions."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        predictions_list = []
        labels_list = []
        logits_list = []
        all_paths = []

        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                old_labels = labels

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                if outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = (outputs.squeeze() > 0).long()
                
                total += labels.size(0)
                correct += (predicted == old_labels).sum().item()

                predictions_list.append(predicted)
                labels_list.append(old_labels)
                logits_list.append(outputs)
                all_paths.extend(paths)

        # Move to CPU once
        all_predictions = torch.cat(predictions_list)
        all_labels = torch.cat(labels_list)
        all_logits = torch.cat(logits_list)

        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        # Calculate additional metrics
        try:
            if torch.tensor(all_logits).shape[1] == 2:
                probs = torch.nn.functional.softmax(all_logits, dim=1)[:, 1]
            else:
                probs = all_logits.squeeze()

            ppv = prevalence_corrected_ppv_at_90_recall_gpu(
                probs,
                torch.tensor(all_labels).to(self.device),
                prevalence=1/101
)

                
        except Exception as e:
            logger.warning(f"PPV calculation failed: {e}")
            ppv = 0.0
        
        return epoch_loss, epoch_acc, ppv, all_predictions.cpu().numpy(), all_labels.cpu().numpy(), all_logits.cpu().numpy(), all_paths
    
    def create_predictions_dataframe(self, val_paths_list, val_labels_true, val_logits, fold, split_type='val'):
        """Create predictions DataFrame for a fold."""
        return pd.DataFrame({
            'image_path': val_paths_list,
            'sample_id': [Path(p).name for p in val_paths_list],
            'center': [
                Path(p).parts[-3].split('_')[1] 
                if len(Path(p).parts) >= 3 and '_' in Path(p).parts[-3] 
                else 'unknown' 
                for p in val_paths_list
            ],
            'target': val_labels_true,
            'logits_0': [logits[0] if len(logits)==2 else 0 for logits in val_logits],
            'logits_1': [logits[1] if len(logits)==2 else logits for logits in val_logits],
            'fold': fold,
            'split_type': split_type  # 'val' or 'test'
        })
    
    def print_classification_report(self, y_true, y_pred, fold=None):
        """Print classification report."""
        fold_str = f" for Fold {fold}" if fold is not None else ""
        logger.info(f'\nClassification Report{fold_str}:')
        logger.info('\n' + classification_report(y_true, y_pred, target_names=['NBDE', 'NEO']))
    