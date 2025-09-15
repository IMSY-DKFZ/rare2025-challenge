"""
Memory-efficient vectorized inference - processes TTA transforms on-the-fly to avoid disk space issues.
"""
# Set environment variables for debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.timm_model import TimmClassificationModel, DINOv3ViT_L
from typing import List, Dict
import gc
from time import time

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
REPO_PATH = Path("/opt/app/dino_repo/")

def load_params(in_path, device="cpu"):
    """Load calibration parameters from a torch checkpoint."""
    checkpoint = torch.load(in_path, map_location=device)
    return checkpoint["t"].to(device), checkpoint["b"].to(device)

def stack_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack multiple state dicts into a single state dict with batched parameters."""
    if not state_dicts:
        return {}
    
    print(f"Stacking {len(state_dicts)} state dictionaries...")
    stacked_dict = {}
    
    keys = list(state_dicts[0].keys())
    for i, key in enumerate(keys):
        tensors = [sd[key] for sd in state_dicts]
        stacked_dict[key] = torch.stack(tensors, dim=0)
        
        if i % 50 == 0:
            print(f"Stacked parameter {i+1}/{len(keys)}: {key}")
    
    return stacked_dict


class VectorizedResNet(nn.Module):
    """Vectorized ResNet that processes multiple COMPLETE models in parallel."""
    
    def __init__(self, state_dicts: List[Dict[str, torch.Tensor]], device):
        super().__init__()
        self.device = device
        self.num_models = len(state_dicts)
        
        print(f"Creating vectorized ResNet for {self.num_models} models...")
        
        # Create template model
        template_weights = RESOURCE_PATH / "resnet/top2/fold_0_best_checkpoint.pth"
        template_model = TimmClassificationModel(
            num_classes=2, 
            weights=template_weights,
            device=device
        )
        
        self.template = template_model.model
        
        # Stack all state dicts
        stacked_params = stack_state_dicts(state_dicts)
        
        # Clean parameter names
        self.stacked_params = {}
        for name, stacked_param in stacked_params.items():
            self.stacked_params[name] = stacked_param.to(device)
        
        del template_model
        torch.cuda.empty_cache()
    
    def forward(self, x):
        def single_model_forward(params, x_single):
            return torch.func.functional_call(self.template, params, (x_single,))
        
        vmapped_forward = torch.vmap(single_model_forward, in_dims=(0, None))
        return vmapped_forward(self.stacked_params, x)


class VectorizedDINOv3Group(nn.Module):
    """Vectorized DINOv3 for a small group of models (e.g., one augmentation type)."""
    
    def __init__(self, state_dicts: List[Dict[str, torch.Tensor]], device):
        super().__init__()
        self.device = device
        self.num_models = len(state_dicts)

        print(f"Creating vectorized DINOv3 group for {self.num_models} models...")
        
        # Create template model
        dinov3_weights = RESOURCE_PATH / "vitl/RARE_dinov3_vitl16/top1/models/final_dinov3_vitl16_fold0_lora_dino.pth"
        template_weights = RESOURCE_PATH / "vitl/RARE_dinov3_vitl16/top1/models/final_dinov3_vitl16_fold0_lora.pth"
        self.template = DINOv3ViT_L(lora=True, weights_path=dinov3_weights, lora_weights_path=template_weights)
        
        # Clean and stack state dicts
        cleaned_state_dicts = []
        for sd in state_dicts:
            cleaned = sd.copy()
            if "criterion.weight" in cleaned:
                cleaned.pop("criterion.weight")
            cleaned = {k.replace("adapted_model.", "").replace("classifier.", ""): v 
                      for k, v in cleaned.items()}
            cleaned_state_dicts.append(cleaned)
        
        stacked_params = stack_state_dicts(cleaned_state_dicts)
        self.stacked_params = {name: param.to(device) for name, param in stacked_params.items()}
        
        print(f"DINOv3: Loaded {len(self.stacked_params)} parameter groups")
        torch.cuda.empty_cache()
    
    def forward(self, x):
        def single_model_forward(params, x_single):
            return torch.func.functional_call(self.template, params, (x_single,))
        
        vmapped_forward = torch.vmap(single_model_forward, in_dims=(0, None))
        return vmapped_forward(self.stacked_params, x)


class TTADataset(Dataset):
    """Dataset that applies TTA transforms on-the-fly"""
    
    def __init__(self, original_images: np.ndarray, tta_transforms: List):
        self.original_images = original_images
        self.tta_transforms = tta_transforms
        self.num_images = len(original_images)
        self.num_transforms = len(tta_transforms)
    
    def __len__(self):
        return self.num_images * self.num_transforms
    
    def __getitem__(self, idx):
        # Calculate which image and which transform
        img_idx = idx // self.num_transforms
        transform_idx = idx % self.num_transforms
        
        # Apply transform
        image = self.original_images[img_idx]
        transform = self.tta_transforms[transform_idx]
        transformed = transform(image=image)['image']
        
        return transformed, img_idx, transform_idx


class PreTransformedDataset(Dataset):
    def __init__(self, transformed_images: torch.Tensor):
        self.images = transformed_images.cpu()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


def apply_single_transform_cpu(images: np.ndarray, transform) -> torch.Tensor:
    """Apply a single transform to all images on CPU"""
    transformed_batch = []
    for img in images:
        transformed = transform(image=img)['image']
        transformed_batch.append(transformed)
    return torch.stack(transformed_batch)

def create_vectorized_resnet(device: torch.device, temperatures, biases):
    """Create vectorized ResNet ensemble - all 20 models at once"""
    print("Loading all ResNet50 state dicts...")
    
    state_dicts = []
    for augment in ["top1","top2","top3","top4"]:
        for fold in [0,1,2,3,4]:
            weights_path = RESOURCE_PATH / f"resnet/{augment}/fold_{fold}_best_checkpoint.pth"
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)['model_state_dict']
            state_dicts.append(state_dict)
            t, b = load_params(RESOURCE_PATH / f"resnet/{augment}/fold_{fold}_calibration_params.pt")
            temperatures.append(t)
            biases.append(b)
    
    print(f"Loaded {len(state_dicts)} ResNet50 state dicts")
    return VectorizedResNet(state_dicts, device)

def create_dinov3_group(device: torch.device, temperatures, biases):
    """Create vectorized DINOv3 ensemble for each augmentation type and fold (4 x 5 models)"""
    state_dicts = []
    dinov3_augments = ['top1', 'top2', 'top3', 'top4']
    
    for augment in dinov3_augments:
        print(f"Loading DINOv3 '{augment}' group...")
        for fold in [0, 1, 2, 3, 4]:
            weights_path = RESOURCE_PATH / f"vitl/RARE_dinov3_vitl16/{augment}/models/final_dinov3_vitl16_fold_{fold}_lora.pth"
            state_dict = torch.load(weights_path, map_location=device)
            state_dicts.append(state_dict)
            t, b = load_params(RESOURCE_PATH / f"vitl/RARE_dinov3_vitl16/{augment}/fold_{fold}_calibration_params.pt")
            temperatures.append(t)
            biases.append(b)
    
    print(f"Loaded {len(state_dicts)} state dicts for DINOv3-base")
    return VectorizedDINOv3Group(state_dicts, device)

def run_inference_on_group(group_ensemble, dataloader, device, group_name):
    print(f"Running inference on {group_name} group...")
    
    group_ensemble.eval()
    predictions = []
    
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            outputs = group_ensemble(batch)  # [num_models_in_group, batch_size, num_classes]
            predictions.append(outputs.cpu())
    
    result = torch.cat(predictions, dim=1)  # [num_models_in_group, total_samples, num_classes]
    print(f"Completed {group_name}: {result.shape[0]} models")
    
    return result


def run():
    print("First printout in the run-method")
    interface_key = get_interface_key()
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]
    return handler()

def interface_0_handler():
    """Memory-efficient vectorized inference with grouped DINOv3 processing"""
    print("Starting memory-efficient vectorized inference...")
    
    input_images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    print(f"Loaded {len(input_images)} input images")
    
    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pre-compute transforms
    from model.timm_model import TimmClassificationModel
    default_transform = TimmClassificationModel.default_transforms()
    
    print("Pre-computing default transform...")
    transformed_images = []
    for img in input_images:
        transformed = default_transform(image=img)['image']
        transformed_images.append(transformed)
    
    transformed_tensor = torch.stack(transformed_images)
    dataset = PreTransformedDataset(transformed_tensor)
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False
    )
    
    all_predictions = []
    temperatures = []
    biases = []
    
    # Process vectorized ResNet ensemble (all 20 at once)
    print("Processing vectorized ResNet50 ensemble...")
    resnet_ensemble = create_vectorized_resnet(device, temperatures, biases)
    
    resnet_results = run_inference_on_group(resnet_ensemble, dataloader, device, "ResNet50")
    for i in range(resnet_results.shape[0]):
        all_predictions.append(resnet_results[i])
    
    del resnet_ensemble
    torch.cuda.empty_cache()
    gc.collect()
    
    # Process DINOv3 ensembles one group at a time
    print("Processing DINOv3 ensembles...")
    
    # Create group ensemble
    dinov3_group = create_dinov3_group(device, temperatures, biases)
    # Run inference
    group_results = run_inference_on_group(dinov3_group, dataloader, device, f"DINOv3")
    # Add to predictions
    for i in range(group_results.shape[0]):
        all_predictions.append(group_results[i])
    
    # Clean up
    del dinov3_group
    torch.cuda.empty_cache()
    gc.collect()
    
    # Final ensemble
    print(f"Total models processed: {len(all_predictions)} (should be 40: 20 ResNet + 20 DINOv3)")
    stacked_preds = torch.stack(all_predictions)
    print(f"Shape of stacked results: {stacked_preds.shape}")

    t=time()
    recal_preds = recal_all_models(stacked_preds, temperatures, biases, torch.tensor([100/101, 1/101]))
    print(f"Recalibration time: {time()-t}")
    print(f"Shape of recalibrated results: {recal_preds.shape}")

    ensemble_results = noisy_or_pool(recal_preds).tolist()
    
    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=ensemble_results,
    )
    print("Results written successfully")
    return 0

def noisy_or_pool(logits: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    logits: (N, M, 2)
    weights: (M,) or None
    Returns: (N,) pooled positive-class probability
    """
    p = torch.nn.functional.softmax(logits, dim=2)[:,:,1]
    
    if weights is None:
        weights = torch.ones(p.size(1), device=logits.device, dtype=logits.dtype)
    
    weights = torch.clamp(weights, 0, 1)
    return 1 - torch.prod(1 - p * weights, dim=0)


# Utility functions remain the same
def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))

def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())

def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)

def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def expand_stack_by_repeat(arr, target_slices=384):
    # arr: (Z, H, W, C)
    z, h, w, c = arr.shape
    if target_slices <= z:
        return arr[:target_slices]
    reps = (target_slices + z - 1) // z  # ceiling division
    expanded = np.tile(arr, (reps, 1, 1, 1))[:target_slices]
    return expanded  # (target_slices, H, W, C)

def recal_all_models(logits, t_list, b_list, prior):
    """
    logits: [n_models, n_images, 2]
    t_list: list or tensor of shape [n_models] (each element a scalar tensor)
    b_list: list or tensor of shape [n_models, 2]
    prior: [2]
    """
    recalibrated = []
    for i in range(logits.shape[0]):
        preds = logits[i]              # [n_images, 2]
        t = t_list[i]                  # scalar tensor
        b = b_list[i]                  # [2] tensor
        recalibrated.append(recal_with_trained(preds, t, b, prior))
    return torch.stack(recalibrated) 

def recal_with_trained(preds, t, b, prior):
    recal_preds = (t*preds + b) + torch.log(prior)
    return recal_preds-torch.logsumexp(recal_preds, axis=-1, keepdim=True)

if __name__ == "__main__":
    print("Called inference.py. Run about to start!")
    raise SystemExit(run())