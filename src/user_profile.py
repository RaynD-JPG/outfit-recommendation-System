"""
User Profile Module
Personalized Outfit Recommendation System

Takes 5-10 images uploaded by the user, extracts ResNet50 features
from each, and averages them into a single 2048D style profile vector.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction import FeatureExtractor


class UserProfileBuilder:
    """
    Builds a user style profile from uploaded images.

    Usage:
        builder = UserProfileBuilder()
        profile = builder.build_profile(['img1.jpg', 'img2.jpg', ...])
        builder.save_profile(profile, 'data/processed/user_profiles/user1.npy')
    """

    def __init__(self, model_name='resnet50'):
        self.extractor = FeatureExtractor(model_name=model_name)

    def build_profile(self, image_paths: list) -> np.ndarray:
        """
        Build a style profile vector from a list of images.

        Args:
            image_paths: List of paths to user-uploaded images (5-10 recommended)

        Returns:
            Normalized 2048D profile vector, or None if no images could be processed
        """
        if len(image_paths) == 0:
            print("ERROR: No images provided.")
            return None

        print(f"Building profile from {len(image_paths)} image(s)...")

        features = []
        for path in image_paths:
            feat = self.extractor.extract_features(path)
            if feat is not None:
                features.append(feat)
            else:
                print(f"  Skipped: {path}")

        if len(features) == 0:
            print("ERROR: Could not extract features from any image.")
            return None

        # Average all image features → single profile vector, then re-normalize
        profile = np.mean(features, axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-8)

        print(f"✓ Profile built from {len(features)}/{len(image_paths)} images")
        print(f"  Profile shape : {profile.shape}")

        return profile

    def save_profile(self, profile: np.ndarray, save_path: str):
        """Save a user profile vector to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, profile)
        print(f"✓ Profile saved to {save_path}")

    def load_profile(self, load_path: str) -> np.ndarray:
        """Load a previously saved user profile."""
        return np.load(load_path)


def test_user_profile():
    """Test profile building on sample Polyvore images."""
    print("=" * 60)
    print("TESTING USER PROFILE BUILDER")
    print("=" * 60)

    images_dir = Path('data/polyvore/images')
    if not images_dir.exists():
        print("ERROR: Images directory not found.")
        return

    # Use first 7 images as a mock user upload
    sample_images = list(images_dir.glob('*.jpg'))[:7]
    if len(sample_images) == 0:
        print("ERROR: No images found.")
        return

    builder = UserProfileBuilder()
    profile = builder.build_profile(sample_images)

    if profile is not None:
        print(f"\n  ✓ Profile shape     : {profile.shape}")
        print(f"  ✓ Profile norm      : {np.linalg.norm(profile):.6f}  (should be ~1.0)")
        print(f"  ✓ Profile min/max   : [{profile.min():.4f}, {profile.max():.4f}]")

        builder.save_profile(profile, 'data/processed/user_profiles/test_user.npy')

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_user_profile()
