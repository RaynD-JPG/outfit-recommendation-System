"""
Recommendation Module
Personalized Outfit Recommendation System

Loads the pre-built outfit feature database and finds the top-K
most similar outfits to a given user style profile using cosine similarity.

Supports gender filtering (men/women) and minimum image count filtering.
"""

import json
import numpy as np
from pathlib import Path


class OutfitRecommender:
    """
    Recommends outfits based on cosine similarity to a user style profile.

    Usage:
        recommender = OutfitRecommender()
        recommender.load_database()
        results = recommender.recommend(user_profile, top_k=10, gender='men')
    """

    def __init__(
        self,
        features_path='data/processed/outfit_features.npy',
        index_path='data/processed/outfit_index.json',
        gender_map_path='data/processed/gender_map.json',
        images_dir='data/polyvore/images',
    ):
        self.features_path = Path(features_path)
        self.index_path = Path(index_path)
        self.gender_map_path = Path(gender_map_path)
        self.images_dir = Path(images_dir)

        self.outfit_features = None
        self.outfit_index = None
        self.gender_map = None

    def load_database(self):
        """Load the pre-built outfit feature database from disk."""
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Outfit features not found at {self.features_path}\n"
                "Run src/build_outfit_database.py first."
            )

        self.outfit_features = np.load(self.features_path)
        with open(self.index_path) as f:
            self.outfit_index = json.load(f)
        with open(self.gender_map_path) as f:
            self.gender_map = json.load(f)

        print(f"✓ Loaded outfit database")
        print(f"  Outfits : {self.outfit_features.shape[0]}")
        print(f"  Dim     : {self.outfit_features.shape[1]}")

    def recommend(
        self,
        user_profile: np.ndarray,
        top_k: int = 10,
        gender: str = None,
        min_images: int = 3,
    ) -> list:
        """
        Find the top-K most similar outfits to the user profile.

        Args:
            user_profile : Normalized 2048D user style vector
            top_k        : Number of results to return
            gender       : 'men' or 'women' — filters results to matching gender
            min_images   : Only return outfits with at least this many images

        Returns:
            List of dicts with set_id, name, score, num_items, image_paths
        """
        if self.outfit_features is None:
            raise RuntimeError("Database not loaded. Call load_database() first.")

        # Build a boolean mask for gender filtering
        if gender and self.gender_map:
            mask = np.array([
                self.gender_map.get(entry['set_id'], 'women') == gender
                for entry in self.outfit_index
            ])
            filtered_features = self.outfit_features[mask]
            filtered_index = [e for e, m in zip(self.outfit_index, mask) if m]
        else:
            filtered_features = self.outfit_features
            filtered_index = self.outfit_index

        if len(filtered_features) == 0:
            return []

        # Cosine similarity = dot product (vectors are pre-normalized)
        scores = filtered_features @ user_profile

        # Get top candidates (fetch more than needed so we can filter by image count)
        candidate_k = min(top_k * 5, len(scores))
        top_indices = np.argsort(scores)[::-1][:candidate_k]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break

            entry = filtered_index[idx]
            set_id = entry['set_id']
            image_paths = self._get_outfit_images(set_id, entry['num_items'])

            # Skip outfits without enough images
            if len(image_paths) < min_images:
                continue

            results.append({
                'set_id': set_id,
                'name': entry['name'],
                'score': float(scores[idx]),
                'num_items': entry['num_items'],
                'image_paths': image_paths,
            })

        return results

    def _get_outfit_images(self, set_id: str, num_items: int) -> list:
        """Return paths to existing item images for an outfit."""
        paths = []
        for i in range(1, num_items + 2):
            img_path = self.images_dir / f"{set_id}_{i}.jpg"
            if img_path.exists():
                paths.append(str(img_path))
        return paths


def print_results(results: list):
    print("\n" + "=" * 60)
    print(f"TOP {len(results)} RECOMMENDED OUTFITS")
    print("=" * 60)
    for rank, result in enumerate(results, 1):
        print(f"\n#{rank}  {result['name'] or 'Unnamed outfit'}")
        print(f"     set_id : {result['set_id']}")
        print(f"     score  : {result['score']:.4f}")
        print(f"     images : {len(result['image_paths'])}")


def test_recommender():
    print("=" * 60)
    print("TESTING OUTFIT RECOMMENDER")
    print("=" * 60)

    recommender = OutfitRecommender()
    recommender.load_database()

    mock_profile = recommender.outfit_features[:7].mean(axis=0)
    mock_profile = mock_profile / (np.linalg.norm(mock_profile) + 1e-8)

    print("\n-- Men's results --")
    results = recommender.recommend(mock_profile, top_k=5, gender='men', min_images=3)
    print_results(results)

    print("\n-- Women's results --")
    results = recommender.recommend(mock_profile, top_k=5, gender='women', min_images=3)
    print_results(results)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_recommender()
