# DeepFake Video Detection

A deep learning pipeline for classifying videos as **REAL** or **FAKE**, built on the
[DeepFake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) dataset.
Video frames are processed with **InceptionV3** for feature extraction, and a
**GRU-based recurrent model** classifies the resulting frame-feature sequences.

## Overview

- **Frame extraction & preprocessing** — video frames are read with OpenCV, resized to 224x224,
  and normalized.
- **Feature extraction** — InceptionV3 extracts per-frame visual features.
- **Sequence modeling** — a GRU network (with masking to handle variable-length sequences)
  learns temporal patterns across frames to classify a video as REAL or FAKE.
- **Training** — binary cross-entropy loss with the Adam optimizer; best weights are saved
  via a `ModelCheckpoint` callback.
- **Evaluation** — accuracy, precision, recall, F1-score, a confusion matrix, and an ROC curve.

## Project Structure

```
.
├── deepfake_detection.py   # Main pipeline: preprocessing, model, training, evaluation
└── README.md
```

## Requirements

```bash
pip install -U tensorflow
pip install git+https://github.com/tensorflow/docs
pip install opencv-python pandas numpy matplotlib scikit-learn imageio
```

## Dataset

Download the [DeepFake Detection Challenge dataset](https://www.kaggle.com/c/deepfake-detection-challenge)
from Kaggle and update `DATA_FOLDER` in `deepfake_detection.py` to point to your local copy:

```
DATA_FOLDER/
├── train_sample_videos/
│   ├── metadata.json
│   └── *.mp4
└── test_videos/
    └── *.mp4
```

Labels in `metadata.json` (`FAKE` / `REAL`) are mapped to `1` / `0` for binary classification.

## Usage

```bash
python deepfake_detection.py
```

This will:
1. Load and preprocess video frames.
2. Extract features with InceptionV3.
3. Train the GRU classification model.
4. Evaluate on the test set and print accuracy, precision, recall, and F1-score.
5. Plot the confusion matrix, ROC curve, and training/validation accuracy & loss curves.

## Model

| Layer         | Details                          |
|---------------|-----------------------------------|
| GRU           | 16 units, returns sequences       |
| GRU           | 8 units                           |
| Dropout       | 0.4                                |
| Dense         | 8 units, ReLU                      |
| Dense (output)| 1 unit, sigmoid                    |

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Decision threshold:** prediction ≥ 0.5 → FAKE, otherwise REAL

## Hyperparameters

| Parameter        | Value |
|-------------------|-------|
| `IMG_SIZE`         | 224   |
| `BATCH_SIZE`       | 64    |
| `EPOCHS`           | 10    |
| `MAX_SEQ_LENGTH`   | 20    |
| `NUM_FEATURES`     | 2048  |

## License

_Add a license of your choice (e.g. MIT)._
