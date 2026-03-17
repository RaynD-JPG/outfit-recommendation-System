"""
Dataset Exploration Script
Personalized Outfit Recommendation System

Explore and visualize the Polyvore dataset.
"""

import json
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from collections import Counter


def load_polyvore_data():
    """Load Polyvore dataset JSON files."""
    data_dir = Path('data/polyvore')
    
    datasets = {}
    for split in ['train', 'valid', 'test']:
        json_path = data_dir / f'{split}_no_dup.json'
        if json_path.exists():
            with open(json_path) as f:
                datasets[split] = json.load(f)
            print(f"✓ Loaded {split}: {len(datasets[split])} outfits")
        else:
            print(f"✗ {split} data not found at {json_path}")
    
    return datasets


def analyze_dataset_stats(datasets):
    """Analyze and print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    for split, data in datasets.items():
        print(f"\n{split.upper()} SET:")
        print(f"  Total outfits: {len(data)}")
        
        # Count items per outfit
        items_per_outfit = [len(outfit['items']) for outfit in data]
        print(f"  Items per outfit:")
        print(f"    Min: {min(items_per_outfit)}")
        print(f"    Max: {max(items_per_outfit)}")
        print(f"    Mean: {np.mean(items_per_outfit):.2f}")
        print(f"    Median: {np.median(items_per_outfit):.0f}")
        
        # Count categories
        all_categories = []
        for outfit in data:
            for item in outfit['items']:
                if 'categoryid' in item:
                    all_categories.append(item['categoryid'])
        
        category_counts = Counter(all_categories)
        print(f"  Unique categories: {len(category_counts)}")
        print(f"  Most common categories:")
        for cat_id, count in category_counts.most_common(5):
            print(f"    Category {cat_id}: {count} items")


def visualize_sample_outfits(datasets, num_samples=5):
    """Visualize random sample outfits."""
    print("\n" + "=" * 60)
    print("VISUALIZING SAMPLE OUTFITS")
    print("=" * 60)
    
    images_dir = Path('data/polyvore/images')
    train_data = datasets.get('train', [])
    
    if not train_data:
        print("No training data found!")
        return
    
    # Select random outfits
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_data), min(num_samples, len(train_data)), replace=False)
    
    for idx in sample_indices:
        outfit = train_data[idx]
        visualize_outfit(outfit, images_dir)


def visualize_outfit(outfit, images_dir):
    """Visualize a single outfit."""
    print(f"\nOutfit: {outfit.get('name', 'Unnamed')}")
    print(f"  Set ID: {outfit['set_id']}")
    print(f"  Items: {len(outfit['items'])}")
    
    # Create figure
    num_items = len(outfit['items'])
    fig, axes = plt.subplots(1, num_items, figsize=(3 * num_items, 4))
    
    if num_items == 1:
        axes = [axes]
    
    for idx, item in enumerate(outfit['items']):
        filename = f"{outfit['set_id']}_{item['index']}.jpg"
        img_path = images_dir / filename
        
        if img_path.exists():
            img = Image.open(img_path)
            axes[idx].imshow(img)
            
            # Title with item name (truncated)
            title = item.get('name', 'Unknown')[:25]
            axes[idx].set_title(title, fontsize=8)
            axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, 'Image\nNot Found', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_dir = Path('experiments/figures')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"outfit_{outfit['set_id']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    
    plt.close()


def check_missing_images(datasets):
    """Check how many outfit images are missing."""
    print("\n" + "=" * 60)
    print("CHECKING IMAGE AVAILABILITY")
    print("=" * 60)
    
    images_dir = Path('data/polyvore/images')
    
    for split, data in datasets.items():
        print(f"\n{split.upper()}:")
        
        total_items = 0
        missing_items = 0
        
        for outfit in data:
            for item in outfit['items']:
                total_items += 1
                filename = f"{outfit['set_id']}_{item['index']}.jpg"
                img_path = images_dir / filename  # Uses actual filename!
                if not img_path.exists():
                    missing_items += 1
        
        print(f"  Total items: {total_items}")
        print(f"  Missing images: {missing_items}")
        print(f"  Availability: {(1 - missing_items/total_items)*100:.2f}%")


def create_outfit_size_distribution(datasets):
    """Create histogram of outfit sizes."""
    print("\n" + "=" * 60)
    print("CREATING OUTFIT SIZE DISTRIBUTION")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (split, data) in enumerate(datasets.items()):
        items_per_outfit = [len(outfit['items']) for outfit in data]
        
        axes[idx].hist(items_per_outfit, bins=range(1, 12), alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Number of Items per Outfit')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{split.upper()} Set Distribution')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path('experiments/figures/outfit_size_distribution.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved distribution to {save_path}")
    
    plt.close()


def main():
    """Main exploration function."""
    print("=" * 60)
    print("POLYVORE DATASET EXPLORATION")
    print("=" * 60)
    
    # Load data
    datasets = load_polyvore_data()
    
    if not datasets:
        print("\nERROR: No datasets loaded!")
        print("Please ensure Polyvore dataset is downloaded to data/polyvore/")
        return
    
    # Analyze statistics
    analyze_dataset_stats(datasets)
    
    # Check image availability
    check_missing_images(datasets)
    
    # Create visualizations
    create_outfit_size_distribution(datasets)
    
    # Visualize sample outfits
    visualize_sample_outfits(datasets, num_samples=5)
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE!")
    print("Check experiments/figures/ for visualizations")
    print("=" * 60)


if __name__ == '__main__':
    main()
