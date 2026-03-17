"""
Outfit Database Processing Script
Personalized Outfit Recommendation System

Processes all Polyvore training outfits:
  - Extracts ResNet50 features from each item image
  - Averages item features into one vector per outfit
  - Saves feature matrix + index to data/processed/

Outputs:
  data/processed/outfit_features.npy  — shape (N, 2048)
  data/processed/outfit_index.json    — maps row number -> outfit metadata
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction import FeatureExtractor


def build_outfit_database(
    data_path='data/polyvore/train_no_dup.json',
    images_dir='data/polyvore/images',
    output_dir='data/processed',
    model_name='resnet50',
    checkpoint_every=1000,
):
    """
    Build the outfit feature database from the Polyvore training set.

    Args:
        data_path:         Path to train_no_dup.json
        images_dir:        Path to folder containing .jpg images
        output_dir:        Where to save outputs
        model_name:        CNN model to use ('resnet50', 'vgg16', 'efficientnet')
        checkpoint_every:  Save a partial checkpoint every N outfits (resume-safe)
    """
    data_path = Path(data_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training data ---
    print(f"Loading {data_path}...")
    with open(data_path) as f:
        train_data = json.load(f)
    print(f"✓ Loaded {len(train_data)} outfits")

    # --- Check images directory ---
    if not images_dir.exists():
        print(f"ERROR: Images directory not found at {images_dir}")
        return None, None

    # --- Initialize feature extractor ---
    extractor = FeatureExtractor(model_name=model_name)

    outfit_features = []
    outfit_index = []
    skipped = 0

    print(f"\nProcessing outfits...")
    for outfit in tqdm(train_data, desc='Outfits', unit='outfit'):
        set_id = outfit['set_id']
        items = outfit['items']

        # Collect valid image paths for this outfit
        item_features = []
        for item in items:
            img_path = images_dir / f"{set_id}_{item['index']}.jpg"
            if not img_path.exists():
                continue
            feat = extractor.extract_features(img_path)
            if feat is not None:
                item_features.append(feat)

        # Skip outfit if no images could be loaded
        if len(item_features) == 0:
            skipped += 1
            continue

        # Average item features → single outfit vector, then re-normalize
        outfit_feat = np.mean(item_features, axis=0)
        outfit_feat = outfit_feat / (np.linalg.norm(outfit_feat) + 1e-8)

        outfit_features.append(outfit_feat)
        outfit_index.append({
            'row': len(outfit_index),
            'set_id': set_id,
            'name': outfit.get('name', ''),
            'num_items': len(item_features),
        })

        # Checkpoint save so progress isn't lost if the process is interrupted
        if len(outfit_index) % checkpoint_every == 0:
            _save_checkpoint(outfit_features, outfit_index, output_dir)
            print(f"  [checkpoint] {len(outfit_index)} outfits saved")

    # --- Final save ---
    features_array = np.array(outfit_features, dtype=np.float32)
    np.save(output_dir / 'outfit_features.npy', features_array)

    with open(output_dir / 'outfit_index.json', 'w') as f:
        json.dump(outfit_index, f, indent=2)

    print("\n" + "=" * 60)
    print("DATABASE BUILD COMPLETE")
    print("=" * 60)
    print(f"  Outfits processed : {len(outfit_index)}")
    print(f"  Outfits skipped   : {skipped} (no images found)")
    print(f"  Feature matrix    : {features_array.shape}")
    print(f"  Saved to          : {output_dir}/")
    print("=" * 60)

    return features_array, outfit_index


def _save_checkpoint(outfit_features, outfit_index, output_dir):
    """Save partial progress in case of interruption."""
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    np.save(checkpoint_dir / 'outfit_features_partial.npy',
            np.array(outfit_features, dtype=np.float32))
    with open(checkpoint_dir / 'outfit_index_partial.json', 'w') as f:
        json.dump(outfit_index, f)


def verify_database(output_dir='data/processed'):
    """Quick sanity check on the saved database."""
    output_dir = Path(output_dir)
    features_path = output_dir / 'outfit_features.npy'
    index_path = output_dir / 'outfit_index.json'

    print("\nVerifying database...")

    if not features_path.exists():
        print("ERROR: outfit_features.npy not found")
        return

    features = np.load(features_path)
    with open(index_path) as f:
        index = json.load(f)

    print(f"  ✓ Feature matrix shape : {features.shape}")
    print(f"  ✓ Index entries        : {len(index)}")
    print(f"  ✓ Rows match index     : {features.shape[0] == len(index)}")
    print(f"  ✓ All vectors normalized: {np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)}")

    print(f"\nSample entries:")
    for entry in index[:3]:
        print(f"  Row {entry['row']}: set_id={entry['set_id']}, "
              f"name='{entry['name']}', items={entry['num_items']}")


if __name__ == '__main__':
    build_outfit_database()
    verify_database()
