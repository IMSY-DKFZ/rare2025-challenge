"""
Split loader utility for loading data splits created by create_splits.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json


class SplitLoader:
    """
    Utility class for loading and working with data splits.
    """
    
    def __init__(self, splits_dir: str = 'splits', data_root: str = 'data/train'):
        """
        Initialize the split loader.
        
        Args:
            splits_dir: Directory containing the split CSV files
            data_root: Root directory of the actual data files
        """
        self.splits_dir = Path(splits_dir)
        self.data_root = Path(data_root)
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load metadata if available."""
        metadata_path = self.splits_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def available_splits(self) -> List[str]:
        """Get list of available split files."""
        csv_files = list(self.splits_dir.glob('*.csv'))
        return [f.stem for f in csv_files]
    
    def load_split(self, split_name: str) -> pd.DataFrame:
        """
        Load a specific split file.
        
        Args:
            split_name: Name of the split (without .csv extension)
            
        Returns:
            DataFrame with split information
        """
        split_path = self.splits_dir / f'{split_name}.csv'
        if not split_path.exists():
            raise FileNotFoundError(f"Split file {split_path} not found")
        
        df = pd.read_csv(split_path)
        
        # Add absolute paths if data_root is provided
        if self.data_root.exists():
            df['full_path'] = df['image_path'].apply(lambda x: str(self.data_root / x))
        
        return df
    
    def get_fold_data(self, split_name: str, fold: Union[int, str], 
                      split_type: str = 'train') -> Tuple[List[str], List[int]]:
        """
        Get file paths and labels for a specific fold.
        
        Args:
            split_name: Name of the split
            fold: Fold number/identifier
            split_type: 'train' or 'test' (for CV) or specific split name
            
        Returns:
            Tuple of (file_paths, labels)
        """
        df = self.load_split(split_name)
        
        if split_name == '5fold_cv':
            if split_type == 'train':
                mask = df['split'] != f'fold_{fold}'
            else:  # test
                mask = df['split'] == f'fold_{fold}'
        elif split_name == 'holdout_cv':
            if split_type == 'test':
                mask = df['split'] == 'test'
            elif split_type == 'train':
                mask = (df['split'] != f'fold_{fold}') & (df['split'] != f'test')
            else:  # validation
                mask = df['split'] == f'fold_{fold}'
        else:
            mask = df['split'] == split_type
        
        subset = df[mask]
        
        if 'full_path' in subset.columns:
            file_paths = subset['full_path'].tolist()
        else:
            file_paths = subset['image_path'].tolist()
            
        labels = subset['target'].tolist()
        
        return file_paths, labels
    
    def get_train_test_split(self, split_name: str) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Get train and test splits for center-based splits.
        
        Returns:
            Tuple of (train_paths, train_labels, test_paths, test_labels)
        """
        df = self.load_split(split_name)
        
        train_data = df[df['split'] == 'train']
        test_data = df[df['split'] == 'test']
        
        train_paths = train_data['full_path'].tolist() if 'full_path' in train_data.columns else train_data['image_path'].tolist()
        train_labels = train_data['target'].tolist()
        
        test_paths = test_data['full_path'].tolist() if 'full_path' in test_data.columns else test_data['image_path'].tolist()
        test_labels = test_data['target'].tolist()
        
        return train_paths, train_labels, test_paths, test_labels
    
    def get_split_info(self, split_name: str) -> Dict:
        """Get information about a specific split."""
        df = self.load_split(split_name)
        
        info = {
            'total_files': len(df),
            'splits': df['split'].unique().tolist(),
            'class_distribution': df.groupby(['split', 'target']).size().to_dict(),
            'center_distribution': df.groupby(['split', 'center']).size().to_dict()
        }
        
        return info
    
    def create_pytorch_datasets(self, split_name: str, transform=None, fold: Optional[int] = None):
        """
        Create PyTorch-compatible dataset splits.
        
        Note: This is a helper method. You'll need to implement your actual
        PyTorch Dataset class based on your specific needs.
        """
        try:
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        class ImageDataset(Dataset):
            def __init__(self, file_paths, labels, transform=None):
                self.file_paths = file_paths
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.file_paths)
            
            def __getitem__(self, idx):
                # This is a placeholder - implement based on your image loading needs
                from PIL import Image
                image = Image.open(self.file_paths[idx])
                label = self.labels[idx]
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        if split_name in ['center1_train_center2_test', 'center2_train_center1_test']:
            train_paths, train_labels, test_paths, test_labels = self.get_train_test_split(split_name)
            train_dataset = ImageDataset(train_paths, train_labels, transform)
            test_dataset = ImageDataset(test_paths, test_labels, transform)
            return train_dataset, test_dataset
        
        elif split_name == '5fold_cv' and fold is not None:
            train_paths, train_labels = self.get_fold_data(split_name, fold, 'train')
            val_paths, val_labels = self.get_fold_data(split_name, fold, 'test')
            train_dataset = ImageDataset(train_paths, train_labels, transform)
            val_dataset = ImageDataset(val_paths, val_labels, transform)
            return train_dataset, val_dataset
        
        else:
            raise ValueError(f"Unsupported split configuration: {split_name}")


# Example usage functions
def example_usage():
    """Example of how to use the SplitLoader."""
    
    # Initialize loader
    loader = SplitLoader('data/splits', 'data/train')
    
    # Check available splits
    print("Available splits:", loader.available_splits())
    
    # Load a specific split
    df = loader.load_split('5fold_cv')
    print(f"5-fold CV split shape: {df.shape}")
    
    # Get data for fold 0 of 5-fold CV
    train_paths, train_labels = loader.get_fold_data('5fold_cv', 0, 'train')
    val_paths, val_labels = loader.get_fold_data('5fold_cv', 0, 'test')
    
    print(f"Fold 0 - Train: {len(train_paths)} samples")
    print(f"Fold 0 - Val: {len(val_paths)} samples")
    
    # Get center-based split
    train_paths, train_labels, test_paths, test_labels = loader.get_train_test_split('center1_train_center2_test')
    print(f"Center split - Train: {len(train_paths)}, Test: {len(test_paths)}")
    
    # Get split information
    info = loader.get_split_info('5fold_cv')
    print("Split info:", info)


if __name__ == "__main__":
    example_usage()