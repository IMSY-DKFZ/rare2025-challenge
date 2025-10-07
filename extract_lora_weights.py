import os
from pathlib import Path

import torch

def extract_lora_weights(checkpoint_path, output_path:Path):
    """
    Extract weights containing 'lora' in their name from a PyTorch checkpoint
    and save them as a new checkpoint.

    Args:
        checkpoint_path (str): Path to the original checkpoint file
        output_path (str): Path where to save the extracted LoRA weights.
                          If None, will use original filename with '_lora_only' suffix
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load the original checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Extract state_dict if checkpoint is in a specific format
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            # Assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        print("Checkpoint format not recognized")
        return

    # Extract weights containing 'lora' in their name
    lora_state_dict = {k: v for k, v in state_dict.items() if ('lora' in k.lower() or "head" in k.lower())}
    dino_state_dict = {k: v for k, v in state_dict.items() if k not in lora_state_dict.keys()}

    if not lora_state_dict:
        print("No LoRA weights found in the checkpoint.")
        return

    print(f"Found {len(lora_state_dict)} LoRA weights.")

    # Create a new checkpoint with the same structure but only LoRA weights
    new_checkpoint = checkpoint.copy() if isinstance(checkpoint, dict) else {}

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            new_checkpoint["state_dict"] = lora_state_dict
        elif "model_state_dict" in checkpoint:
            new_checkpoint["model_state_dict"] = lora_state_dict
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            new_checkpoint["model"] = lora_state_dict
        else:
            new_checkpoint = lora_state_dict

    # Save the new checkpoint
    try:
        torch.save(new_checkpoint, output_path)
        print(f"LoRA weights saved to {output_path}")
    except Exception as e:
        print(f"Error saving LoRA weights: {e}")

    dino_file_exists = any(file.name.endswith("_dino.pth") for file in output_path.parent.iterdir() if file.is_file())
    if not dino_file_exists:
        dino_file_path = output_path.parent / f"{output_path.stem}_dino.pth"
        try:
            torch.save(dino_state_dict, dino_file_path)
            print(f"DINO weights saved to {dino_file_path}")
        except Exception as e:
            print(f"Error saving DINO weights: {e}")
    else:
        print(f"A DINO weights file already exists in the directory, skipping extraction")

def extract_lora_weights_new_structure(base_path: Path, output_base_path: Path = None):
    """
    Extract LoRA weights from checkpoints in the new directory structure.

    New structure: BASE_PATH/top1/DinoV3_LoRAModel_top1_center1/best_val_ppv_model.ckpt

    Args:
        base_path (Path): Base directory containing top-level folders
        output_base_path (Path, optional): Base directory for output. If None, will create
                                          folders next to the original ones with '_lora_extract' suffix
    """
    if output_base_path is None:
        output_base_path = base_path

    # Get all top-level directories like 'top1', 'top2', etc.
    top_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('top')]

    for top_dir in top_dirs:
        print(f"Processing directory: {top_dir}")

        # Find all model directories inside this top directory
        model_dir = Path(top_dir) / "models"
        # Find checkpoint file
        for fold in [0, 1, 2, 3, 4]:
            checkpoint_file = model_dir / f'final_dinov3_vitl16_fold_{fold}.pth'
            print(checkpoint_file)
            if checkpoint_file.exists():
                # Create output directory structure
                output_dir = checkpoint_file.parent
                # Output path for extracted weights
                output_path = output_dir / f"{checkpoint_file.stem}_lora.pth"

                print(f"Extracting LoRA weights from {checkpoint_file} to {output_path}")
                extract_lora_weights(checkpoint_file, output_path)
            else:
                print(f"Checkpoint file not found in {model_dir}")

if __name__ == "__main__":
    BASE_PATH = Path(f"results/vitl/RARE_dinov3_vitl16/")
    # Use the new function for the new directory structure
    extract_lora_weights_new_structure(BASE_PATH)