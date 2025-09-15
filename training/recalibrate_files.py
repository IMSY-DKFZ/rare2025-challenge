import pandas as pd
import torch
from psrcal.calibration import calibrate, AffineCalLogLoss
from pathlib import Path
from glob import glob


def save_params(t, b, out_path):
    """Save calibration parameters as a torch checkpoint."""
    torch.save({"t": t, "b": b}, out_path)


def load_params(in_path, device="cpu"):
    """Load calibration parameters from a torch checkpoint."""
    checkpoint = torch.load(in_path, map_location=device)
    return checkpoint["t"].to(device), checkpoint["b"].to(device)


def compute_recalibration(csv_path, out="calibration_params.pt"):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract logits and targets
    if "logits_0" in df.keys():
        val_preds = df[["logits_0", "logits_1"]].values
        val_labels = df["target"].values
    else:
        val_preds = df[["0", "1"]].values
        val_labels = df["target"].values

    # Run calibration
    preds, (t, b) = calibrate(
        trnscores=torch.tensor(val_preds),
        trnlabels=torch.tensor(val_labels),
        tstscores=torch.tensor(val_preds),
        calclass=AffineCalLogLoss,
        bias=True,
        priors=[100 / 101, 1 / 101],
        quiet=True,
    )
    print(t.item())
    print(b[0].item())
    output_file = Path(csv_path).parent / out
    save_params(t, b, output_file)
    print(f"Calibration parameters saved to {output_file}")



if __name__ == "__main__":
    print("Recalibrating CSV files...")
    #ResNet recalibration
    for csv_path in glob("results/resnet/*/*/oof_val_predictions.csv"):
        print(csv_path)
        for fold in [0, 1, 2, 3, 4]:
            df = pd.read_csv(csv_path)
            df_fold = df[df['fold']==fold]
            save_path = Path(csv_path).parent / f"fold{fold}_oofs.csv"
            df_fold.to_csv(save_path, index=False)
            out = f"fold_{fold}_calibration_params.pt"
            compute_recalibration(csv_path=save_path,
                                out=out)

    #DINOv3 recalibration
    for fold in [0, 1, 2, 3, 4]:
        for csv_path in glob(f"results/vitl/*/*/test_predictions_fold_{fold}.csv"):
            print(csv_path)
            out = f"fold_{fold}_calibration_params.pt"
            compute_recalibration(csv_path=csv_path,
                                  out=out)