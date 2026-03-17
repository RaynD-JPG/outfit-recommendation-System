"""
Evaluation Module
Personalized Outfit Recommendation System

Evaluates the system on two standard Polyvore benchmark tasks:

  1. Fill-in-the-Blank (FITB)
     Given a partial outfit + 4 candidates, pick the item that best completes it.
     Metric: Accuracy (fraction of questions answered correctly)

  2. Fashion Compatibility Prediction (FCP)
     Given a full outfit, predict whether it is compatible (1) or not (0).
     Metric: AUC (Area Under ROC Curve)

Both tasks require per-item features, not per-outfit features.
An item feature cache is built first so each image is only processed once.
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys

sys.path.insert(0, str(Path(__file__).parent))
from feature_extraction import FeatureExtractor


# ---------------------------------------------------------------------------
# Item Feature Cache
# ---------------------------------------------------------------------------

class ItemFeatureCache:
    """
    Extracts and caches per-item features to avoid redundant computation.
    Items are keyed by their image filename stem (e.g. '119704139_1').
    """

    def __init__(self, images_dir='data/polyvore/images', model_name='resnet50'):
        self.images_dir = Path(images_dir)
        self.extractor = FeatureExtractor(model_name=model_name)
        self._cache = {}  # item_id -> np.ndarray

    def get(self, item_id: str) -> np.ndarray:
        """
        Return feature vector for item_id (e.g. '119704139_1').
        Extracts and caches on first access.
        """
        if item_id not in self._cache:
            img_path = self.images_dir / f"{item_id}.jpg"
            if img_path.exists():
                feat = self.extractor.extract_features(img_path)
                self._cache[item_id] = feat  # may be None if extraction fails
            else:
                self._cache[item_id] = None
        return self._cache[item_id]

    def prefetch(self, item_ids: list):
        """Pre-load features for a list of item IDs (shows a progress bar)."""
        missing = [i for i in item_ids if i not in self._cache]
        if not missing:
            return
        print(f"Prefetching {len(missing)} item features...")
        for item_id in tqdm(missing, desc='Caching items', unit='item'):
            self.get(item_id)

    @property
    def size(self):
        return len(self._cache)


# ---------------------------------------------------------------------------
# Task 1: Fill-in-the-Blank
# ---------------------------------------------------------------------------

def evaluate_fill_in_blank(
    cache: ItemFeatureCache,
    fitb_path='data/polyvore/fill_in_blank_test.json',
) -> float:
    """
    Evaluate fill-in-the-blank accuracy.

    For each question:
      - Average the feature vectors of the known outfit items → query vector
      - Score each of the 4 answer candidates by cosine similarity to query
      - Predict the candidate with the highest score
      - answers[0] is always the correct answer

    Returns:
        Accuracy (0.0 – 1.0)
    """
    print("\n" + "=" * 60)
    print("TASK 1: FILL-IN-THE-BLANK")
    print("=" * 60)

    with open(fitb_path) as f:
        questions = json.load(f)

    print(f"Questions: {len(questions)}")

    correct = 0
    skipped = 0

    for q in tqdm(questions, desc='FITB', unit='q'):
        # Build query from known outfit items
        query_feats = [cache.get(item_id) for item_id in q['question']]
        query_feats = [f for f in query_feats if f is not None]

        if len(query_feats) == 0:
            skipped += 1
            continue

        query = np.mean(query_feats, axis=0)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Score each answer candidate
        best_score = -np.inf
        best_idx = -1

        for idx, answer_id in enumerate(q['answers']):
            feat = cache.get(answer_id)
            if feat is None:
                continue
            score = float(np.dot(query, feat))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == 0:  # answers[0] is always correct
            correct += 1

    answered = len(questions) - skipped
    accuracy = correct / answered if answered > 0 else 0.0

    print(f"\n  Questions answered : {answered} / {len(questions)}")
    print(f"  Correct            : {correct}")
    print(f"  Accuracy           : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    return accuracy


# ---------------------------------------------------------------------------
# Task 2: Fashion Compatibility Prediction
# ---------------------------------------------------------------------------

def evaluate_compatibility(
    cache: ItemFeatureCache,
    compat_path='data/polyvore/fashion_compatibility_prediction.txt',
) -> float:
    """
    Evaluate fashion compatibility prediction using AUC.

    For each outfit:
      - Extract features for all items
      - Score = average pairwise cosine similarity between all item pairs
        (higher score → more compatible)
      - Compute AUC against ground truth labels (1=compatible, 0=incompatible)

    Returns:
        AUC score (0.5 = random, 1.0 = perfect)
    """
    print("\n" + "=" * 60)
    print("TASK 2: FASHION COMPATIBILITY PREDICTION")
    print("=" * 60)

    labels = []
    scores = []
    skipped = 0

    with open(compat_path) as f:
        lines = f.readlines()

    print(f"Outfits: {len(lines)}")

    for line in tqdm(lines, desc='Compatibility', unit='outfit'):
        parts = line.strip().split()
        if len(parts) < 3:
            skipped += 1
            continue

        label = int(parts[0])
        item_ids = parts[1:]

        # Get features for all items in this outfit
        feats = [cache.get(item_id) for item_id in item_ids]
        feats = [f for f in feats if f is not None]

        if len(feats) < 2:
            skipped += 1
            continue

        # Average pairwise cosine similarity (dot product, vectors are normalized)
        feats_matrix = np.stack(feats)  # (n_items, 2048)
        sim_matrix = feats_matrix @ feats_matrix.T  # (n_items, n_items)

        # Take upper triangle (exclude diagonal self-similarity)
        n = sim_matrix.shape[0]
        upper = sim_matrix[np.triu_indices(n, k=1)]
        score = float(upper.mean())

        labels.append(label)
        scores.append(score)

    auc = roc_auc_score(labels, scores)

    print(f"\n  Outfits evaluated  : {len(labels)} / {len(lines)}")
    print(f"  Skipped            : {skipped}")
    print(f"  Positive (compat)  : {sum(labels)}")
    print(f"  Negative (incompat): {len(labels) - sum(labels)}")
    print(f"  AUC                : {auc:.4f}")

    return auc


# ---------------------------------------------------------------------------
# Run Both Evaluations
# ---------------------------------------------------------------------------

def run_evaluation(
    images_dir='data/polyvore/images',
    fitb_path='data/polyvore/fill_in_blank_test.json',
    compat_path='data/polyvore/fashion_compatibility_prediction.txt',
    model_name='resnet50',
):
    """Run both benchmark tasks and print a final summary."""
    print("=" * 60)
    print("EVALUATION — POLYVORE BENCHMARKS")
    print("=" * 60)

    # Build item feature cache (shared across both tasks)
    cache = ItemFeatureCache(images_dir=images_dir, model_name=model_name)

    # Collect all item IDs needed upfront and prefetch
    item_ids = set()

    with open(fitb_path) as f:
        for q in json.load(f):
            item_ids.update(q['question'])
            item_ids.update(q['answers'])

    with open(compat_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                item_ids.update(parts[1:])

    cache.prefetch(list(item_ids))

    # Run tasks
    fitb_accuracy = evaluate_fill_in_blank(cache, fitb_path)
    compat_auc = evaluate_compatibility(cache, compat_path)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Fill-in-the-Blank Accuracy : {fitb_accuracy:.4f}  ({fitb_accuracy * 100:.2f}%)")
    print(f"  Compatibility AUC          : {compat_auc:.4f}")
    print("=" * 60)
    print("\nBaseline reference (random chance):")
    print("  FITB Accuracy : 0.25  (1 in 4 random guess)")
    print("  AUC           : 0.50  (random classifier)")

    return {
        'fitb_accuracy': fitb_accuracy,
        'compatibility_auc': compat_auc,
    }


if __name__ == '__main__':
    run_evaluation()
