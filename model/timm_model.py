import os

import timm
import numpy as np
from pathlib import Path
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import math

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

class TimmClassificationModel:
    def __init__(self, weights: None, num_classes: int = 1, device: torch.device = None, transform = None):
        """
        Wrapper for creating and managing a classification model using timm.

        :param device: PyTorch device to move the model to. Defaults to 'cuda' if available.
        :param num_classes: Number of output classes. Default is 1.
        :param pretrained: Whether to load pretrained weights. Default is True.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet50GastroNet(num_classes=num_classes)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=False)['model_state_dict'], strict=True)
        self.model.to(self.device).eval()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()


    def predict(self, images: list[np.ndarray]):
        """
        Accepts a list of numpy images (HWC, uint8 or float),
        converts them to PIL Images, applies transforms, and runs inference.
        """
        # pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        probs = []
        for img in images:
            img = self.transform(image=img)['image'].unsqueeze(0).to(self.device)  # Add batch dimension
            with torch.inference_mode():
                logits = self.model(img)
                prob = logits.cpu()

            probs.append(prob)

        return torch.cat(probs, dim=0)
    
    def predict_batched(self, images, batch_size=8):
        outs = []
        with torch.inference_mode():
            for s in range(0, len(images), batch_size):
                chunk = images[s:s+batch_size]
                ts = [self.transform(image=im)["image"] for im in chunk]
                batch = torch.stack(ts).to(self.device)
                logits = self.model(batch).cpu()
                outs.append(logits)
        return torch.cat(outs, dim=0)

    @staticmethod
    def default_transforms():
        return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ])

class ResNet50GastroNet(torch.nn.Module):
    """ResNet50 model with GastroNet weights adaptation."""
    
    def __init__(self, num_classes: int = 2):
        super(ResNet50GastroNet, self).__init__()
        
        # Load model with GastroNet weights
        # First create a standard ResNet50
        self.backbone = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
            
        # If the model doesn't have the right number of classes, modify the classifier
        if hasattr(self.backbone, 'fc'):
            if self.backbone.fc.out_features != num_classes:
                self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            if self.backbone.classifier.out_features != num_classes:
                self.backbone.classifier = torch.nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class DINOv3ViT_L(torch.nn.Module):
    def __init__(self, transform=None, lora=False, weights_path="dino3l.pth", lora_weights_path=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        repo_root_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        REPO_PATH = os.path.join(repo_root_dir, "app/dino_repo")
        self.model = torch.hub.load(REPO_PATH, 'dinov3_vitl16', source='local', pretrained=False)
        if lora:
            self._add_lora_layers()
        self.head = torch.nn.Linear(1024, 2)
        checkpoint = torch.load(weights_path, map_location=self.device)
        if not lora:
            checkpoint.pop("criterion.ema_threshold")
            checkpoint.pop("criterion.initialized")
        else:
            checkpoint.pop("criterion.weight")
            lora_weights = torch.load(lora_weights_path, map_location=self.device)
            checkpoint.update(lora_weights)
        checkpoint = {k.replace("adapted_model.", "").replace("classifier.", ""): v for k, v in checkpoint.items()}
        self.load_state_dict(checkpoint, strict=True)
        self.to(self.device).eval()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

    def forward(self, x):
        features = self.model(x)
        return self.head(features)

    def predict(self, images: list[np.ndarray]):
        probs = []
        for img in images:
            img = self.transform(image=img)['image'].unsqueeze(0).to(self.device)  # Add batch dimension
            with torch.inference_mode():
                logits = self.forward(img)
                prob = logits.cpu()
            probs.append(prob)
        return torch.cat(probs, dim=0)

    def _add_lora_layers(self, rank=32, alpha=64):
        def _find_and_replace_linear_layers(module, path=""):
            for name, child in module.named_children():
                new_path = f"{path}.{name}" if path else name
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    _find_and_replace_linear_layers(child, new_path)

        _find_and_replace_linear_layers(self.model)

    @staticmethod
    def default_transforms():
        return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def predict_batched(model, images, batch_size=8):
    outs = []
    with torch.inference_mode():
        for s in range(0, len(images), batch_size):
            chunk = images[s:s+batch_size]
            ts = [model.transform(image=im)["image"] for im in chunk]
            batch = torch.stack(ts).to(model.device)
            logits = model.forward(batch).cpu()
            outs.append(logits)
    return torch.cat(outs, dim=0)

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.down = torch.nn.Linear(in_features, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        # add dropout
        self.dropout = torch.nn.Dropout(0.1)

        # Initialize weights
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return self.up(self.dropout(self.down(x))) * self.scale

class LoRALinear(torch.nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.lora = LoRALayer(
            self.in_features, linear_layer.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def default_transforms(im_size=224):
    return A.Compose([
    A.Resize(im_size, im_size),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
    ])
