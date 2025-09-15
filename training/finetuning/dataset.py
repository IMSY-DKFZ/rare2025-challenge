import random
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import lightning as pl

class PLDataModule(pl.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, predict_dataloader):
        super().__init__()
        self.train_dataloader_obj = train_dataloader
        self.val_dataloader_obj = val_dataloader
        self.test_dataloader_obj = test_dataloader
        self.predict_dataloader_obj = predict_dataloader

    def train_dataloader(self):
        return self.train_dataloader_obj

    def val_dataloader(self):
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self.test_dataloader_obj

    def predict_dataloader(self):
        if self.predict_dataloader_obj:
            return self.predict_dataloader_obj
        else:
            return self.test_dataloader_obj

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_dataloaders(args, DatasetClass):
    train_dataset = DatasetClass(hparams=args, mode="train")
    val_dataset = DatasetClass(hparams=args, mode="val")
    test_dataset = DatasetClass(hparams=args, mode="test")

    train_dataloader = None
    sampler = None
    if args.training.weighted_sampler and hasattr(DatasetClass, 'get_weighted_sampler'):
        sampler = train_dataset.get_weighted_sampler()
    if len(train_dataset) > 0:
        if sampler is None:
            train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.dataset.num_workers, worker_init_fn=seed_worker)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.dataset.num_workers, worker_init_fn=seed_worker, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.dataset.num_workers, worker_init_fn=seed_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.dataset.num_workers, worker_init_fn=seed_worker)
    return train_dataloader, val_dataloader, test_dataloader

class RAREDiseaseRecognition(Dataset):
    def __init__(self, hparams, mode):
        self.mode = mode
        self.dataset = RARE2025(hparams, mode)
        self.task_type = "classification"
        self.num_classes = 2
        self.transform = self.build_transform(hparams.dataset.transforms, mode)

    def __len__(self):
        length = len(self.dataset.data_df)
        return length

    def __getitem__(self, idx):
        item = self.dataset.data_df.loc[idx]
        label = item["target"]
        img = read_image(item["image_path"]).permute(1, 2, 0).numpy()
        img_dict = self.transform(image=img)
        img = img_dict["image"]
        return img, label

    def get_class_counts(self):
        labels = self.dataset.data_df["target"].tolist()
        return np.bincount(labels)

    def get_weighted_sampler(self):
        labels = self.dataset.data_df["target"].tolist()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def build_transform(self, transforms, mode):
        height = transforms.get('height', 224)
        width = transforms.get('width', 224)
        mean = transforms.get('mean', [0.485, 0.456, 0.406])
        std = transforms.get('std', [0.229, 0.224, 0.225])
        transform_buffer = [
            # Base
            A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8)
        ]
        if mode == "train":
            if transforms.mode == "top1":
                transform_buffer += [
                    # jigsaw | blur_variations | color_variations | optical_distortion
                    A.RandomGridShuffle(grid=(3, 3), p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.3),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.21),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.5)
                ]
            elif transforms.mode == "top2":
                transform_buffer += [
                    # affine | texture_variations | elastic
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        cval=0,
                        cval_mask=0,
                        mode=cv2.BORDER_REFLECT_101,
                        fit_output=False,
                        p=0.5,
                    ),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.21),
                    A.ElasticTransform(
                        alpha=120.0,
                        sigma=6.0,
                        alpha_affine=0,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    )
                ]
            elif transforms.mode == "top3":
                transform_buffer += [
                    # affine | blur_variations | texture_variations | optical_distortion
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        cval=0,
                        cval_mask=0,
                        mode=cv2.BORDER_REFLECT_101,
                        fit_output=False,
                        p=0.5,
                    ),
                    A.MotionBlur(blur_limit=7, p=0.3),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.21),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.21),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.5)
                ]
            elif transforms.mode == "top4":
                transform_buffer += [
                    # affine | color_variations | optical_distortion
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        cval=0,
                        cval_mask=0,
                        mode=cv2.BORDER_REFLECT_101,
                        fit_output=False,
                        p=0.5,
                    ),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.5)
                ]
            else:
                transform_buffer += [
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=[-0.2, 0.4], p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2)
                ]
        else:
            transform_buffer = [A.Resize(height=height, width=width)]

        transform_buffer += [
            # Normalize + tensor
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
        return A.Compose(transform_buffer)

class RARE2025(Dataset):
    def __init__(self, hparams, mode):
        #self.network_drive_path = Path(hparams.paths.data_path, "rare2025")
        self.data_df = self.prepare_rare(hparams, mode)

    def prepare_rare(self, hparams, mode):
        base_path = Path(hparams.paths.resource_path)
        val_fold = "test"
        if hparams.dataset.split == "center1":
            all_df = pd.read_csv(base_path / "splits" /"center1_train_center2_test.csv")
        elif hparams.dataset.split == "center2":
            all_df = pd.read_csv(base_path / "splits" / "center2_train_center1_test.csv")
        else:
            all_df = pd.read_csv(base_path / "splits" / "5fold_cv.csv")
            val_fold = hparams.dataset.split

        if mode == "train":
            df = all_df[all_df["split"] != val_fold].reset_index(drop=True)
            print(f"Training on {len(df)}")
        else:
            df = all_df[all_df["split"] == val_fold].reset_index(drop=True)
            print(f"Validating/Testing on {len(df)}")
        df['image_path'] = str(base_path) + "/train/" + df['image_path']
        return df