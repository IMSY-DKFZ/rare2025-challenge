"""Main training script with support for both 5-fold CV and holdout validation."""

import os
import pandas as pd
import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from .config import TrainingConfig, get_arg_parser
from .training import Trainer
from .evaluation import Evaluator
from .utils import set_random_seeds, create_checkpoint_dir, setup_logging

import cv2
cv2.setNumThreads(1)

# Setup logging
logger = setup_logging()


def run_cross_validation(trainer, evaluator, loader, config, checkpoint_dir):
    """Run cross-validation (either 5-fold or holdout)."""
    all_oof_predictions = []
    all_test_predictions = []  # For holdout CV test predictions
    fold_ppvs = []
    
    if config.single_fold:
        folds = [config.fold_number]
        logger.info(f"Running on a single fold with number {config.fold_number}")
    else:
        folds = range(config.n_folds)
    # Always run 5 folds
    for fold in folds:
        logger.info(f"\nStarting Fold {fold}")
        
        if config.cv_type == 'holdout_cv':
            # Holdout CV: train on train, validate on val, and also predict on test
            train_paths, train_labels = loader.get_fold_data('holdout_cv', fold, 'train')
            val_paths, val_labels = loader.get_fold_data('holdout_cv', fold, 'val')
            test_paths, test_labels = loader.get_fold_data('holdout_cv', fold, 'test')
        else:
            # 5-fold CV: train on train, validate on test (no separate test set)
            train_paths, train_labels = loader.get_fold_data('5fold_cv', fold, 'train')
            val_paths, val_labels = loader.get_fold_data('5fold_cv', fold, 'test')
            test_paths, test_labels = None, None
        
        logger.info(f"Fold {fold} - Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
        if test_paths is not None:
            logger.info(f"Fold {fold} - Test samples: {len(test_paths)}")
        logger.info(f"Train class distribution: {np.bincount(train_labels)}")
        logger.info(f"Val class distribution: {np.bincount(val_labels)}")
        if test_labels is not None:
            logger.info(f"Test class distribution: {np.bincount(test_labels)}")
        
        # Hyperparameter search if enabled
        if config.hp_search:
            best_hp, best_checkpoint_path = trainer.hyperparameter_search(fold, train_paths, train_labels, val_paths, val_labels, checkpoint_dir)
            # Update config with best hyperparameters for this fold
            # Load the best model from Optuna and evaluate it
            val_preds, val_labels_true, val_logits, val_paths_list, fold_ppv = trainer.load_and_evaluate_best_model(
                fold, val_paths, val_labels, best_checkpoint_path, checkpoint_dir
            )
        else:
            # Original training flow for no HP search
            val_preds, val_labels_true, val_logits, val_paths_list, fold_ppv = trainer.train_fold(
                fold, train_paths, train_labels, val_paths, val_labels, None, checkpoint_dir
            )
        
        fold_ppvs.append(fold_ppv)
        
        # Create validation predictions DataFrame
        val_fold_df = evaluator.create_predictions_dataframe(
            val_paths_list, val_labels_true, val_logits, fold, split_type='val'
        )
        all_oof_predictions.append(val_fold_df)
        
        # For holdout CV, also predict on test set using the trained model
        if config.cv_type == 'holdout_cv' and test_paths is not None:
            test_preds, test_labels_true, test_logits, test_paths_list = trainer.predict_on_test_set(
                fold, test_paths, test_labels, checkpoint_dir
            )
            
            # Create test predictions DataFrame
            test_fold_df = evaluator.create_predictions_dataframe(
                test_paths_list, test_labels_true, test_logits, fold, split_type='test'
            )
            all_test_predictions.append(test_fold_df)
    
    # Combine predictions
    final_val_predictions = pd.concat(all_oof_predictions, ignore_index=True)
    
    if config.cv_type == 'holdout_cv':
        final_test_predictions = pd.concat(all_test_predictions, ignore_index=True)
        return final_val_predictions, final_test_predictions, fold_ppvs
    else:
        return final_val_predictions, None, fold_ppvs

def run_train_test(trainer, evaluator, loader, config, checkpoint_dir):
    """Run train test split on the different centers"""
    all_oof_predictions = []
    all_test_predictions = []  # For holdout CV test predictions
    fold_ppvs = []
    
    logger.info(f"\nStarting training")
    
    if config.cv_type == 'center1':
        train_paths, train_labels, test_paths, test_labels = loader.get_train_test_split('center1_train_center2_test')
    else:
        train_paths, train_labels, test_paths, test_labels = loader.get_train_test_split('center2_train_center1_test')
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
                                                                        train_paths,
                                                                        train_labels,
                                                                        test_size=0.2,
                                                                        stratify=train_labels,
                                                                        random_state=42
                                                                        )   
    
    logger.info(f"{config.cv_type} - Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    if test_paths is not None:
        logger.info(f"Test samples: {len(test_paths)}")
    logger.info(f"Train class distribution: {np.bincount(train_labels)}")
    logger.info(f"Val class distribution: {np.bincount(val_labels)}")
    if test_labels is not None:
        logger.info(f"Test class distribution: {np.bincount(test_labels)}")
    
    # Hyperparameter search if enabled
    if config.hp_search:
        best_hp, best_checkpoint_path = trainer.hyperparameter_search(0, train_paths, train_labels, val_paths, val_labels, checkpoint_dir)
        # Update config with best hyperparameters for this fold
        # Load the best model from Optuna and evaluate it
        val_preds, val_labels_true, val_logits, val_paths_list, fold_ppv = trainer.load_and_evaluate_best_model(
            0, val_paths, val_labels, best_checkpoint_path, checkpoint_dir
        )
    else:
        # Original training flow for no HP search
        val_preds, val_labels_true, val_logits, val_paths_list, fold_ppv = trainer.train_fold(
            0, train_paths, train_labels, val_paths, val_labels, None, checkpoint_dir
        )
    
    fold_ppvs.append(fold_ppv)
    
    # Create validation predictions DataFrame
    val_fold_df = evaluator.create_predictions_dataframe(
        val_paths_list, val_labels_true, val_logits, 0, split_type='val'
    )
    all_oof_predictions.append(val_fold_df)
    
    # For holdout CV, also predict on test set using the trained model
    test_preds, test_labels_true, test_logits, test_paths_list = trainer.predict_on_test_set(
        0, test_paths, test_labels, checkpoint_dir
    )
    
    # Create test predictions DataFrame
    test_fold_df = evaluator.create_predictions_dataframe(
        test_paths_list, test_labels_true, test_logits, 0, split_type='test'
    )
    all_test_predictions.append(test_fold_df)
    
    # Combine predictions
    final_val_predictions = pd.concat(all_oof_predictions, ignore_index=True)
    
    final_test_predictions = pd.concat(all_test_predictions, ignore_index=True)
    return final_val_predictions, final_test_predictions, fold_ppvs


def main():
    """Main training function."""
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Create config from arguments
    config = TrainingConfig.from_args(args)
    
    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(config)
    logger.info(f'Checkpoints will be saved to: {checkpoint_dir}')
    
    # Load split data
    from data_splitting.split_loader import SplitLoader
    loader = SplitLoader(config.splits_dir, config.data_dir)
    
    # Initialize trainer and evaluator
    trainer = Trainer(config, device)
    evaluator = Evaluator(device)
    if config.cv_type in ["center1", "center2"]:
        # Run train test split
        val_predictions, test_predictions, fold_ppvs = run_train_test(
            trainer, evaluator, loader, config, checkpoint_dir
        )
    else:
        # Run cross-validation
        val_predictions, test_predictions, fold_ppvs = run_cross_validation(
            trainer, evaluator, loader, config, checkpoint_dir
        )
    
    # Save validation results
    val_output_path = os.path.join(checkpoint_dir, 'oof_val_predictions.csv')
    val_predictions.to_csv(val_output_path, index=False)
    logger.info(f"\nValidation predictions saved to: {val_output_path}")
    
    # Save test results if holdout CV
    if (config.cv_type in ['holdout_cv', 'center1', 'center2']) and test_predictions is not None:
        test_output_path = os.path.join(checkpoint_dir, 'oof_test_predictions.csv')
        test_predictions.to_csv(test_output_path, index=False)
        logger.info(f"Test predictions saved to: {test_output_path}")
    
    logger.info(f"Final validation DataFrame shape: {val_predictions.shape}")
    if test_predictions is not None:
        logger.info(f"Final test DataFrame shape: {test_predictions.shape}")
    logger.info(f"Mean validation PPV across folds: {np.mean(fold_ppvs):.4f} ± {np.std(fold_ppvs):.4f}")


if __name__ == "__main__":
    main()