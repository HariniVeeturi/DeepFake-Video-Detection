"""
DeepFake Detection Pipeline
Extracted and consolidated from project report.

NOTE: A few pieces referenced in the original report snippets were never
actually defined in the text you provided (train_test_split call that
produces train_data/test_data/train_labels/test_labels/test_videos,
the `load_video` function, the `play_video` function, and the
`feature_extractor` model). I've left TODO markers wherever the code
depends on them so you can drop in your original implementations
before pushing.
"""

# ----------------------------------------------------------------------
# Imports and Setup
# ----------------------------------------------------------------------
# !pip install -U --upgrade tensorflow
# !pip install git+https://github.com/tensorflow/docs

from tensorflow_docs.vis import embed
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os


# ----------------------------------------------------------------------
# Configuration of Training Arguments (hyperparameters)
# ----------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

DATA_FOLDER = "C:/Users/KBhagyaRekha/Downloads/deepfake-detection-challenge"
TRAIN_SAMPLE_FOLDER = "train_sample_videos"
TEST_FOLDER = "test_videos"


# ----------------------------------------------------------------------
# Dataset Loading and Splitting
# ----------------------------------------------------------------------
print(f"Train_samples:{len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test_samples:{len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")

# Load metadata
train_sample_metadata = pd.read_json(
    os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, "metadata.json")
).T
train_sample_metadata.head()

# TODO: label encoding (FAKE -> 1, REAL -> 0) and the actual
# train_test_split(...) call (90% train / 10% test) that produces
# train_data, test_data, train_labels, test_labels, test_videos
# were described in the report but not included as code — add them here.


# ----------------------------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------------------------
def preprocess_video(video_path, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)  # Resize frame
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)
    cap.release()
    return np.array(frames)


# Example usage: Extract frames from a sample video
sample_video = os.path.join(
    DATA_FOLDER, TRAIN_SAMPLE_FOLDER, train_sample_metadata.index[0]
)
processed_frames = preprocess_video(sample_video)

# Display first frame after preprocessing
plt.imshow(processed_frames[0])
plt.axis("off")
plt.show()


# ----------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------
frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model([frame_features_input, mask_input], output)

# Compile model with optimizer and loss function
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# ----------------------------------------------------------------------
# Training Setup and Training
# ----------------------------------------------------------------------
checkpoint = keras.callbacks.ModelCheckpoint(
    "model_weights.weights.h5",
    save_weights_only=True,
    save_best_only=True,
)

# Train the model
history = model.fit(
    [train_data[0], train_data[1]],
    train_labels,
    validation_data=([test_data[0], test_data[1]], test_labels),
    epochs=EPOCHS,
    batch_size=8,
    callbacks=[checkpoint],
)


# ----------------------------------------------------------------------
# Loading Best Checkpoint and Final Evaluation
# ----------------------------------------------------------------------
# Load the best checkpoint weights
model.load_weights(
    r"C:\Users\KBhagyaRekha\Downloads\deepfake-detection-challenge\model_weights.weights.h5"
)

# Evaluate model performance on test dataset
test_loss, test_accuracy = model.evaluate([test_data[0], test_data[1]], test_labels)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")


# ----------------------------------------------------------------------
# Metrics Computation
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }
    return metrics


# Example usage: Compute metrics on test set predictions
test_predictions = model.predict([test_data[0], test_data[1]])
test_predictions = (test_predictions >= 0.5).astype(int)  # Convert probabilities to binary labels
metrics_result = compute_metrics(test_labels, test_predictions)
print(metrics_result)


# ----------------------------------------------------------------------
# Test Evaluation Callback / Single-Video Prediction
# ----------------------------------------------------------------------
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])  # TODO: define feature_extractor
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    frames = load_video(os.path.join(DATA_FOLDER, TEST_FOLDER, path))  # TODO: define load_video
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]


# Select random test video for evaluation
test_video = np.random.choice(test_videos["video"].values.tolist())
print(f"Test video path: {test_video}")

if sequence_prediction(test_video) >= 0.5:
    print("The predicted class of the video is FAKE")
else:
    print("The predicted class of the video is REAL")

# Display video for reference
play_video(test_video, TEST_FOLDER)  # TODO: define play_video


# ----------------------------------------------------------------------
# Visual Explanation
# ----------------------------------------------------------------------

# --- Confusion Matrix ---
test_predictions = model.predict([test_data[0], test_data[1]])
test_predictions = (test_predictions >= 0.5).astype(int)

conf_matrix = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["REAL", "FAKE"])
disp.plot(cmap="Blues")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(test_labels, test_predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

# --- Training vs Validation Accuracy/Loss Curves ---
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()
