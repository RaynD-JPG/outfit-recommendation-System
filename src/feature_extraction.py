"""
Feature Extraction Module
Personalized Outfit Recommendation System

This module provides functionality to extract visual features from fashion images
using pre-trained CNN models (ResNet50, VGG16, EfficientNet).
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extract visual features from images using pre-trained CNN models.
    
    Supports:
    - ResNet50 (default): 2048-dimensional features
    - VGG16: 4096-dimensional features  
    - EfficientNet-B0: 1280-dimensional features
    """
    
    def __init__(self, model_name: str = 'resnet50', device: str = 'auto'):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Name of the model ('resnet50', 'vgg16', 'efficientnet')
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name.lower()
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Loading {self.model_name}...")
        
        # Load model
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ {self.model_name} loaded successfully!")
        print(f"✓ Feature dimension: {self.get_feature_dim()}D")
    
    def _load_model(self):
        """Load and configure the specified pre-trained model."""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
            # Remove final FC layer (keep avgpool output)
            model = torch.nn.Sequential(*list(model.children())[:-1])

        elif self.model_name == 'vgg16':
            model = models.vgg16(weights='DEFAULT')
            # Remove classifier, keep features + avgpool
            model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

        elif self.model_name == 'efficientnet':
            model = models.efficientnet_b0(weights='DEFAULT')
            # Remove final classifier
            model.classifier = torch.nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of feature vectors."""
        if self.model_name == 'resnet50':
            return 2048
        elif self.model_name == 'vgg16':
            return 4096
        elif self.model_name == 'efficientnet':
            return 1280
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def extract_features(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Preprocess
            img_tensor = self.transform(img).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Convert to numpy and L2-normalize
            features = features.squeeze().cpu().numpy()
            features = features / (np.linalg.norm(features) + 1e-8)

            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_batch(self, image_paths: List[Union[str, Path]], 
                     batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once
            
        Returns:
            Array of feature vectors (N x feature_dim)
        """
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    continue
            
            if len(batch_tensors) == 0:
                continue
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch)
            
            # Add to results — reshape guards against single-image batch collapsing dims
            features = features.squeeze().cpu().numpy()
            if features.ndim == 1:
                features = features.reshape(1, -1)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            all_features.append(features)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_tensors)}/{len(image_paths)} images")
        
        return np.vstack(all_features)
    
    def save_features(self, features: np.ndarray, save_path: Union[str, Path]):
        """Save extracted features to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        print(f"✓ Features saved to {save_path}")
    
    def load_features(self, load_path: Union[str, Path]) -> np.ndarray:
        """Load previously extracted features."""
        return np.load(load_path)


def test_feature_extraction():
    """Test the feature extractor on sample images."""
    print("=" * 60)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 60)
    
    # Initialize extractor
    extractor = FeatureExtractor(model_name='resnet50')
    
    # Find sample images
    images_dir = Path('data/polyvore/images')
    if not images_dir.exists():
        print("ERROR: Images directory not found!")
        print("Please download the Polyvore dataset first.")
        return
    
    sample_images = list(images_dir.glob('*.jpg'))[:5]
    
    if len(sample_images) == 0:
        print("ERROR: No images found!")
        return
    
    print(f"\nTesting on {len(sample_images)} sample images...")
    
    # Extract features
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n{i}. Processing: {img_path.name}")
        features = extractor.extract_features(img_path)
        
        if features is not None:
            print(f"   ✓ Feature shape: {features.shape}")
            print(f"   ✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"   ✓ Feature mean: {features.mean():.3f}")
        else:
            print(f"   ✗ Failed to extract features")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    # Run test
    test_feature_extraction()
