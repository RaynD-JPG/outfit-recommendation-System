# Outfit Recommendation System

A personalized outfit recommendation system.

## What It Does

1. User uploads 5–10 images of clothing they like
2. A pre-trained ResNet50 CNN extracts 2048-dimensional feature vectors from each image
3. Features are averaged to create a **User Style Profile**
4. The profile is compared against pre-computed Polyvore outfit features using cosine similarity
5. The system returns the **top-10 matching outfits** from the Polyvore dataset

## Dataset

Uses the [Polyvore dataset](https://github.com/xthan/polyvore-dataset):
- **Train:** 17,316 outfits
- **Validation:** 1,497 outfits
- **Test:** 3,076 outfits
- ~126,928 outfit item images

The dataset JSON files are included. Images must be downloaded separately from the Polyvore dataset repository.

## Tech Stack

- Python 3.12
- PyTorch + torchvision (ResNet50)
- NumPy, scikit-learn, Pillow
- Streamlit (web UI)

## Project Structure

```
outfit-recommendation-thesis/
├── src/
│   ├── feature_extraction.py     # FeatureExtractor class (ResNet50/VGG16/EfficientNet)
│   ├── build_outfit_database.py  # Processes Polyvore outfits → feature matrix
│   ├── user_profile.py           # Builds user style profile from uploaded images
│   ├── recommendation.py         # Cosine similarity retrieval against outfit database
│   ├── evaluation.py             # FITB accuracy + fashion compatibility AUC benchmarks
│   └── explore_dataset.py        # Dataset statistics and visualizations
├── app.py                        # Streamlit web app
├── data/
│   └── polyvore/                 # Dataset JSON files (images not included)
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download Polyvore images into `data/polyvore/images/`, then:

```bash
# Build the outfit feature database (required before running the app)
python src/build_outfit_database.py

# Run the web app
streamlit run app.py
```

## Evaluation Benchmarks

| Task | Metric | Random Baseline |
|------|--------|----------------|
| Fill-in-the-Blank (FITB) | Accuracy | 0.25 |
| Fashion Compatibility Prediction (FCP) | AUC | 0.50 |

```bash
python src/evaluation.py
```

## Status

- [x] Feature extraction pipeline (ResNet50)
- [x] Outfit database builder with checkpointing
- [x] User profile builder
- [x] Recommendation engine with gender filtering
- [x] Evaluation scripts (FITB + FCP)
- [x] Streamlit web app
- [ ] End-to-end testing
- [ ] Thesis writing (in progress)

**Deadline:** April 30, 2026
