<div align="center">

# Real-Time Gesture Recognition for Sign Language | GestureBridge

**A vocabulary-agnostic pipeline that recognizes hand and body gestures from a live webcam feed using MediaPipe Holistic and an LSTM sequence classifier.**



<!-- TODO: replace with your demo GIF. Easiest method: record 20-30s of the app running,
     then drag the .mp4 into a new GitHub issue on this repo, copy the generated URL,
     and paste it below as ![demo](URL) -->


</div>

---

## Overview

Sign language is inaccessible to most people who don't use it, and that gap keeps millions of hearing- and speech-impaired people locked out of everyday conversation. This project builds the recognition layer that any such system needs: a model that watches a webcam feed and turns gestures into text in real time.

The key design decision is that the model classifies **motion, not stills**. Many signs are distinguished by movement rather than hand shape — the same handshape can mean different things depending on how it travels. So instead of labelling one frame at a time, the model consumes a 30-frame sequence of body, face, and hand landmarks and predicts what that whole sequence represents.

Currently trained on **TODO_N custom gestures**: `TODO_GESTURE_1`, `TODO_GESTURE_2`, `TODO_GESTURE_3`.

> **On the vocabulary:** the gestures in this model are my own, defined for the purpose of building and validating the pipeline — they are not authentic ASL or ISL signs. That's a deliberate scoping choice for a first version, and it means the interesting claim here is about the *pipeline*, not the vocabulary. Because the model learns from extracted landmarks rather than raw pixels, swapping in a real signed vocabulary is a data-collection problem, not a rearchitecture: point `collect_data.py` at a new gesture list and retrain. See [Limitations](#limitations).

---

## How It Works

```
Webcam frame
     │
     ▼
MediaPipe Holistic  ──►  pose + face + both hands landmarks
     │
     ▼
Flatten to a 1662-dim keypoint vector per frame
     │
     ▼
Rolling buffer of the last 30 frames  →  (30, 1662) sequence
     │
     ▼
Stacked LSTM  →  Dense layers  →  Softmax over TODO_N classes
     │
     ▼
Prediction (smoothed over the last 10 frames + confidence threshold)
```

### The feature vector

Every frame is reduced from raw pixels to a fixed-length vector of landmark coordinates. This is what makes the model small, fast, and largely independent of lighting, skin tone, and background:

| Component | Landmarks | Values each | Total |
|---|---|---|---|
| Pose | 33 | x, y, z, visibility | 132 |
| Face | 468 | x, y, z | 1404 |
| Left hand | 21 | x, y, z | 63 |
| Right hand | 21 | x, y, z | 63 |
| **Per frame** | | | **1662** |

When a hand is out of frame, its 63 values are zero-filled so the vector length stays constant.

### Model architecture

<!-- TODO: run model.summary() in your notebook and correct the layer sizes below if they differ -->

| Layer | Output shape | Notes |
|---|---|---|
| LSTM (64 units) | (30, 64) | `return_sequences=True` |
| LSTM (128 units) | (30, 128) | `return_sequences=True` |
| LSTM (64 units) | (64) | final sequence state |
| Dense (64) | (64) | ReLU |
| Dense (32) | (32) | ReLU |
| Dense (TODO_N) | (TODO_N) | Softmax |

Trained with categorical cross-entropy and the Adam optimizer for TODO_EPOCHS epochs. Total parameters: **TODO_PARAM_COUNT**.

### Prediction smoothing

A single frame-window prediction is noisy — the model flickers between classes during the transition into a gesture. Two guards are applied before anything is shown on screen:

1. The predicted class must exceed a confidence threshold of `TODO_THRESHOLD`.
2. The same class must win the last 10 consecutive predictions.

This trades a small amount of latency for a far more stable output.

---

## Results

<!-- TODO: fill from your notebook's evaluation cell. If you haven't computed a
     confusion matrix yet, sklearn.metrics.multilabel_confusion_matrix on your
     test split takes two lines and is worth adding. -->

| Metric | Value |
|---|---|
| Training sequences | TODO |
| Test sequences | TODO |
| Test accuracy | TODO% |
| Categorical accuracy (final epoch) | TODO% |

![Confusion matrix](assets/confusion_matrix.png)

**Where it struggles:** TODO — e.g. gestures sharing a similar starting pose get confused; accuracy drops when the signer stands further from the camera than during data collection; performance degrades under strong backlighting.

---

## Dataset

The dataset was collected locally rather than downloaded, using the included collection script.

- **Gestures:** TODO_N custom classes
- **Sequences per gesture:** TODO
- **Frames per sequence:** 30
- **Signers:** TODO (recorded by TODO people in TODO environments)

Each sequence is stored as 30 `.npy` files of shape `(1662,)` under `MP_Data/<gesture>/<sequence_id>/`. Raw video is never saved — only extracted landmarks — which keeps the dataset small and avoids storing identifiable footage.

> **On scope:** this is a small, self-collected dataset. It demonstrates that the pipeline works end to end, but it is not large or diverse enough to generalize across signers, camera positions, or lighting conditions.

---

## Repository Structure

```
sign-language-detector/
├── collect_data.py         # Webcam capture → landmark extraction → MP_Data/
├── train.py                # Builds and trains the LSTM, saves model weights
├── app.py                  # Real-time inference on live webcam feed
├── utils.py                # MediaPipe helpers, keypoint extraction, drawing
├── model/
│   └── action.h5           # Trained weights (committed so you can run inference directly)
├── notebooks/
│   └── exploration.ipynb   # Original end-to-end development notebook
├── assets/                 # Demo GIF, confusion matrix, architecture diagram
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

Python 3.9 is recommended — MediaPipe and TensorFlow are both version-sensitive, and 3.9 is the sweet spot where the pinned versions in `requirements.txt` install cleanly. A webcam is required.

### Installation

```bash
git clone https://github.com/smitthakkar11/sign-language-detector.git
cd sign-language-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the live detector

The trained weights are included, so this works immediately:

```bash
python app.py
```

Press `q` to quit.

### Train on your own gestures

The pipeline is not tied to any particular vocabulary — define whatever gesture set you need:

```bash
# 1. Edit the `actions` list in collect_data.py to your chosen gestures
python collect_data.py           # record sequences, following the on-screen countdown

# 2. Train
python train.py
```

Collection takes roughly TODO minutes per gesture. Vary your distance from the camera and your position in the frame between takes — models trained on perfectly consistent framing fail the moment you move.

---

## Limitations

Stated plainly, because they shape how the results above should be read:

- **The vocabulary is custom, not a real signed language.** The gestures were defined by me to build and test the pipeline. This is a gesture-recognition system demonstrated on sign-language-style input — not a sign language translator, and not usable by anyone who signs.
- **Small vocabulary.** TODO_N classes.
- **Signer dependence.** Trained on a small number of people, so accuracy on a new person will be meaningfully lower than the reported test accuracy.
- **Isolated gestures only.** The model classifies one gesture per 30-frame window. It does not segment continuous signing, and real signed languages have their own grammar and syntax that word-by-word mapping does not capture.
- **Fixed window length.** Gestures taking longer than 30 frames get truncated.
- **Test-split optimism.** The test sequences were recorded in the same session as the training sequences, so the reported accuracy is an upper bound on real-world performance.

---

## Roadmap

- [ ] Retrain on an authentic Indian Sign Language vocabulary with input from signers
- [ ] Expand to 20+ classes with multiple signers
- [ ] Port to TensorFlow.js for a fully in-browser demo (no server, full framerate)
- [ ] Replace the fixed 30-frame window with continuous gesture segmentation
- [ ] Compare the LSTM against a Transformer encoder and a 1D-CNN baseline
- [ ] Add text-to-speech so recognized gestures are spoken aloud

---

## Built With

- [MediaPipe Holistic](https://developers.google.com/mediapipe) — pose, face, and hand landmark extraction
- [TensorFlow / Keras](https://www.tensorflow.org/) — LSTM sequence model
- [OpenCV](https://opencv.org/) — video capture and overlay rendering
- NumPy, scikit-learn

---

<div align="center">
Built by <a href="https://github.com/smitthakkar11">Smit Thakkar</a>
</div>
