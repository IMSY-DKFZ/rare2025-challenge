"""Model definitions and loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import timm
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

def _process_state_dict(checkpoint):
    """Normalize checkpoint format so it can always be loaded into timm ResNet."""
    # Case 1: full checkpoint with "state_dict"
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip "student_backbone." prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("student_backbone."):
            new_key = k.replace("student_backbone.", "", 1)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    return new_state_dict

class ResNet50GastroNet(nn.Module):
    """ResNet50 model with GastroNet (https://doi.org/10.1016/j.media.2024.103298) weights adaptation."""

    def __init__(
        self,
        num_classes: int = 2,
        repo_id: str = "tgwboers/GastroNet-5M_Pretrained_Weights",
        filename: str = "RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1.pth",
        our_weights: bool = False
    ):
        super(ResNet50GastroNet, self).__init__()

        # Load model with GastroNet weights
        try:
            # First create a standard ResNet50
            self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=num_classes)

            if our_weights:
                model_path = filename
                checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            else:
                # Then download and load the specific GastroNet weights
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)

                # Load the weights
                checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = _process_state_dict(checkpoint)
            self.backbone.load_state_dict(state_dict, strict=False)

            logger.info(f"Successfully loaded GastroNet weights from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load GastroNet weights: {e}")
            # Fallback to ImageNet pretrained
            self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)

        # Ensure correct number of classes
        if hasattr(self.backbone, "fc"):
            if self.backbone.fc.out_features != num_classes:
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif hasattr(self.backbone, "classifier"):
            if self.backbone.classifier.out_features != num_classes:
                self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class CE_PPVAtRecallLoss(nn.Module):
    """
    CE + ranking for PPV at fixed recall level (binary model with 2 logits).
    """

    def __init__(self, recall_level=0.90, lambda_=1.0, margin=0.5, beta=0.99, class_weights=None):
        super(CE_PPVAtRecallLoss, self).__init__()
        if not (0 < recall_level < 1):
            raise ValueError("recall_level must be between 0 and 1.")

        self.recall_level = recall_level
        self.quantile = 1.0 - recall_level
        self.lambda_ = lambda_
        self.margin = margin
        self.beta = beta
        self.class_weights = class_weights

        self.register_buffer("ema_threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, inputs, targets):
        # Ensure long targets
        targets = targets.long()

        base_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)

        # logits for positive class
        pos_logits = inputs[:, 1]
        pos_sample_logits = pos_logits[targets == 1]
        neg_sample_logits = pos_logits[targets == 0]

        if pos_sample_logits.numel() == 0 or neg_sample_logits.numel() == 0:
            return base_loss

        with torch.no_grad():
            batch_threshold = torch.quantile(pos_sample_logits, self.quantile)
            if not self.initialized:
                self.ema_threshold.copy_(batch_threshold)
                self.initialized = torch.tensor(True, device=self.ema_threshold.device)
            else:
                self.ema_threshold.mul_(self.beta).add_(batch_threshold, alpha=1 - self.beta)

        violations = neg_sample_logits - (self.ema_threshold.detach() - self.margin)
        ranking_loss = torch.mean(torch.relu(violations))
        return base_loss + self.lambda_ * ranking_loss


class DifferentiableSurrogateLoss(nn.Module):
    """
    Differentiable surrogate loss optimizing Precision (PPV) with recall constraint for binary case.
    """

    def __init__(self, recall_threshold=0.9, lambda_penalty=1.0, epsilon=1e-8):
        super(DifferentiableSurrogateLoss, self).__init__()
        self.recall_threshold = recall_threshold
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if logits.dim() > 1:
            logits = logits.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()

        y = targets.float()
        probs = torch.sigmoid(logits)

        tp_soft = y * probs
        fp_soft = (1 - y) * probs
        fn_soft = y * (1 - probs)

        tp_sum = torch.sum(tp_soft)
        fp_sum = torch.sum(fp_soft)
        fn_sum = torch.sum(fn_soft)

        precision = tp_sum / (tp_sum + fp_sum + self.epsilon)
        recall = tp_sum / (tp_sum + fn_sum + self.epsilon)

        recall_penalty = F.relu(self.recall_threshold - recall)
        loss = -precision + self.lambda_penalty * recall_penalty
        return loss


class MultiClassSurrogateLoss(nn.Module):
    """
    Multi-class version using softmax probabilities.
    """

    def __init__(self, num_classes, recall_threshold=0.9, lambda_penalty=1.0, epsilon=1e-8, reduction="mean"):
        super(MultiClassSurrogateLoss, self).__init__()
        self.num_classes = num_classes
        self.recall_threshold = recall_threshold
        self.lambda_penalty = lambda_penalty
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        total_loss = 0.0
        for c in range(self.num_classes):
            y_c = targets_onehot[:, c]
            probs_c = probs[:, c]

            tp_soft = y_c * probs_c
            fp_soft = (1 - y_c) * probs_c
            fn_soft = y_c * (1 - probs_c)

            tp_sum = torch.sum(tp_soft)
            fp_sum = torch.sum(fp_soft)
            fn_sum = torch.sum(fn_soft)

            precision_c = tp_sum / (tp_sum + fp_sum + self.epsilon)
            recall_c = tp_sum / (tp_sum + fn_sum + self.epsilon)

            recall_penalty_c = F.relu(self.recall_threshold - recall_c)
            loss_c = -precision_c + self.lambda_penalty * recall_penalty_c

            if self.reduction == "mean":
                total_loss += loss_c / self.num_classes
            else:
                total_loss += loss_c

        return total_loss

def create_loss_function(config, class_weights=None):
    """Create loss function based on configuration."""
    weights = class_weights if config.use_class_weights else None
    if weights is not None and isinstance(weights, torch.Tensor):
        # ensure device set later in forward
        pass

    lt = config.loss_type

    if lt == "ppv":
        return CE_PPVAtRecallLoss(
            recall_level=0.9,
            lambda_=config.ppv_lambda,
            margin=config.ppv_margin,
            beta=config.ppv_beta,
            class_weights=weights,
        )
    if lt == "surrogate":
        if config.num_classes == 1:
            return DifferentiableSurrogateLoss(lambda_penalty=config.surrogate_lambda)
        else:
            return MultiClassSurrogateLoss(num_classes=config.num_classes, lambda_penalty=config.surrogate_lambda)

    # default
    return nn.CrossEntropyLoss(weight=weights)


def create_model(config, phase="train"):
    """Create model based on configuration."""
    num_classes = config.num_classes

    if config.model_type == "resnet50":
        if config.model_filename is not None:
            return ResNet50GastroNet(num_classes=num_classes, filename=config.model_filename, our_weights=config.our_weights)
        else: 
            return ResNet50GastroNet(num_classes=num_classes, our_weights=config.our_weights)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")