#!/bin/bash

EPOCHS=50
TRIALS=50
BASE_OUTPUT="results/resnet"
OPTUNA_STORAGE="sqlite:///final_runs.db"
CV_TYPE="5fold_cv"
LOSS="surrogate"

CONDA_ENV="rare"

echo "🧪 Running final training runs"

augmentations=("top1" "top2" "top3" "top4")
folds=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

for augmentation in "${augmentations[@]}"; do
    
    python -m training.train --epochs $EPOCHS \
            --cv_type $CV_TYPE \
            --output_dir "${BASE_OUTPUT}/${augmentation}" \
            --hp_search \
            --optuna_study_name "${augmentation}_${CV_TYPE}" \
            --optuna_storage $OPTUNA_STORAGE \
            --hp_trials $TRIALS \
            --loss_type $LOSS

    for fold in "${folds[@]}"; do
        conda run --no-capture-output -n $CONDA_ENV python training/finetuning/main.py \
              --output-dir results/vitl/ \
              --resource-path data/ \
              --dataset-split "${fold}" \
              --transforms-mode "${augmentation}" \
              --weight-path "resources/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
              --epochs 100
    done

    echo "✅ Training with augmentation $augmentation completed"
    echo "⏸️ Pausing 10 seconds..."
    sleep 10

done
