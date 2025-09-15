"""
Loader for EVC_Barretts_FullSet dataset for testing trained models.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re
import logging

logger = logging.getLogger(__name__)


class EVCBarrettsLoader:
    """
    Loader for EVC_Barretts_FullSet dataset.
    
    This dataset contains endoscopic images with filenames structured as:
    patXX_imY_ZZZZ.png where:
    - XX: patient number
    - Y: image number for that patient  
    - ZZZZ: pathology (ACHD=cancer/class 1, NDBT=no cancer/class 0)
    """
    
    def __init__(self, data_root: str = 'data/EVC_Barretts_FullSet'):
        """
        Initialize the EVC Barretts loader.
        
        Args:
            data_root: Root directory of the EVC_Barretts_FullSet dataset
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / 'images'
        self.annotations_bmp_dir = self.data_root / 'annotations_bmp'
        self.annotations_mat_dir = self.data_root / 'annotations_mat'
        
        # Check if directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        logger.info(f"Initialized EVC Barretts loader with data root: {self.data_root}")
        
        # Load and parse image data
        self.image_data = self._load_image_data()
        
    def _load_image_data(self) -> pd.DataFrame:
        """Load and parse all image files from the dataset."""
        image_files = list(self.images_dir.glob('*.png'))
        
        if not image_files:
            raise ValueError(f"No PNG images found in {self.images_dir}")
        
        data = []
        filename_pattern = re.compile(r'pat(\d+)_im(\d+)_([A-Z]+)\.png')
        
        for image_path in image_files:
            match = filename_pattern.match(image_path.name)
            if match:
                patient_id = int(match.group(1))
                image_id = int(match.group(2))
                pathology = match.group(3)
                
                # Convert pathology to class label
                if pathology == 'NDBT':
                    target = 0  # No cancer
                elif pathology == 'ACHD':
                    target = 1  # Cancer
                else:
                    logger.warning(f"Unknown pathology '{pathology}' in {image_path.name}")
                    continue
                
                data.append({
                    'image_path': str(image_path),
                    'relative_path': str(image_path.relative_to(self.data_root)),
                    'filename': image_path.name,
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'pathology': pathology,
                    'target': target
                })
            else:
                logger.warning(f"Filename doesn't match expected pattern: {image_path.name}")
        
        if not data:
            raise ValueError("No valid images found matching the expected filename pattern")
        
        df = pd.DataFrame(data)
        
        # Log dataset statistics
        logger.info(f"Loaded {len(df)} images from EVC Barretts dataset")
        logger.info(f"Patients: {df['patient_id'].nunique()}")
        logger.info(f"Class distribution: {df['target'].value_counts().to_dict()}")
        logger.info(f"Pathology distribution: {df['pathology'].value_counts().to_dict()}")
        
        return df
    
    def get_all_data(self) -> Tuple[List[str], List[int]]:
        """
        Get all image paths and labels for inference.
        
        Returns:
            Tuple of (file_paths, labels)
        """
        file_paths = self.image_data['image_path'].tolist()
        labels = self.image_data['target'].tolist()
        
        return file_paths, labels
    
    def get_patient_data(self, patient_id: int) -> Tuple[List[str], List[int]]:
        """
        Get data for a specific patient.
        
        Args:
            patient_id: Patient ID to filter by
            
        Returns:
            Tuple of (file_paths, labels)
        """
        patient_data = self.image_data[self.image_data['patient_id'] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"No data found for patient {patient_id}")
        
        file_paths = patient_data['image_path'].tolist()
        labels = patient_data['target'].tolist()
        
        return file_paths, labels
    
    def get_by_pathology(self, pathology: str) -> Tuple[List[str], List[int]]:
        """
        Get data filtered by pathology type.
        
        Args:
            pathology: Either 'NDBT' or 'ACHD'
            
        Returns:
            Tuple of (file_paths, labels)
        """
        if pathology not in ['NDBT', 'ACHD']:
            raise ValueError(f"Pathology must be 'NDBT' or 'ACHD', got: {pathology}")
        
        pathology_data = self.image_data[self.image_data['pathology'] == pathology]
        
        file_paths = pathology_data['image_path'].tolist()
        labels = pathology_data['target'].tolist()
        
        return file_paths, labels
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive information about the dataset."""
        info = {
            'total_images': len(self.image_data),
            'total_patients': self.image_data['patient_id'].nunique(),
            'class_distribution': self.image_data['target'].value_counts().to_dict(),
            'pathology_distribution': self.image_data['pathology'].value_counts().to_dict(),
            'patients_per_class': self.image_data.groupby('target')['patient_id'].nunique().to_dict(),
            'images_per_patient': self.image_data.groupby('patient_id').size().describe().to_dict(),
            'patient_ids': sorted(self.image_data['patient_id'].unique().tolist())
        }
        
        return info
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame compatible with the evaluation pipeline.
        
        Returns:
            DataFrame with columns similar to the split loader format
        """
        df = self.image_data.copy()
        
        # Add columns to match expected format
        df['split'] = 'test'  # Mark all as test data
        df['center'] = 'evc_barretts'  # Dataset identifier
        df['full_path'] = df['image_path']
        
        return df
    
    def get_annotation_paths(self, image_path: str) -> List[str]:
        """
        Get annotation file paths for a given image (for future segmentation work).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of annotation file paths
        """
        image_name = Path(image_path).stem  # Remove .png extension
        
        # Look for bitmap annotations
        annotation_files = []
        for expert_num in range(1, 6):  # 5 experts
            annotation_path = self.annotations_bmp_dir / f"{image_name}_exp{expert_num}.bmp"
            if annotation_path.exists():
                annotation_files.append(str(annotation_path))
        
        return annotation_files


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = EVCBarrettsLoader('data/EVC_Barretts_FullSet')
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get all data
    all_paths, all_labels = loader.get_all_data()
    print(f"\nTotal samples: {len(all_paths)}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
    # Get data for specific patient
    try:
        patient_paths, patient_labels = loader.get_patient_data(1)
        print(f"\nPatient 1 samples: {len(patient_paths)}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Get data by pathology
    ndbt_paths, ndbt_labels = loader.get_by_pathology('NDBT')
    achd_paths, achd_labels = loader.get_by_pathology('ACHD')
    print(f"\nNDBT samples: {len(ndbt_paths)}")
    print(f"ACHD samples: {len(achd_paths)}")