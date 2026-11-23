import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import librosa
import joblib

from sklearn.ensemble import ExtraTreesClassifier


# ==========================================
# Absolute Pitch AI v0.2
# Automatic Lightweight Training
# ==========================================

VERSION = "0.2.0"

MODEL_PATH = Path("models/pitch_ai.pkl")

# 44.1 kHz supports the full MIDI 0-127 range, including G9.
SAMPLE_RATE = 44100
FFT_SIZE = 65536

# CPU負荷を抑える設定
TRAIN_VARIANTS = 6
TREE_COUNT = 40
MAX_DEPTH = 18
CPU_WORKERS = 1

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def midi_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_frequency(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ==========================================
# Feature extraction
# ==========================================

def extract_features(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < FFT_SIZE:
        audio = np.pad(audio, (0, FFT_SIZE - len(audio)))
    else:
        start = (len(audio) - FFT_SIZE) // 2
        audio = audio[start:start + FFT_SIZE]

    audio = audio * np.hanning(len(audio))

    spectrum = np.abs(np.fft.rfft(audio, n=FFT_SIZE))

    frequencies = np.fft.rfftfreq(
        FFT_SIZE,
        1 / SAMPLE_RATE
    )

    spectrum /= np.max(spectrum) + 1e-10

    features = []

    # 128音それぞれの倍音特徴
    for midi in range(128):
        fundamental = midi_frequency(midi)

        score = 0.0

        for harmonic in range(1, 13):
            target = fundamental * harmonic

            if target >= SAMPLE_RATE / 2:
                break

            index = int(round(target / (SAMPLE_RATE / FFT_SIZE)))

            left = max(0, index - 1)
            right = min(len(spectrum), index + 2)

            peak = np.max(spectrum[left:right])

            score += peak / (harmonic ** 1.15)

        features.append(score)

    # スペクトル全体の特徴
    centroid = np.sum(frequencies * spectrum) / (
        np.sum(spectrum) + 1e-10
    )

    flatness = np.exp(
        np.mean(np.log(spectrum + 1e-10))
    ) / (np.mean(spectrum) + 1e-10)

    features.extend([
        centroid / (SAMPLE_RATE / 2),
        flatness,
        np.std(spectrum),
        np.mean(spectrum)
    ])

    return np.asarray(features, dtype=np.float32)


# ==========================================
# Synthetic dataset generation
# ==========================================

def generate_sample(midi, rng):
    """
    音色や倍音構成をランダムに変えた
    学習用単音データを生成する。
    """

    duration = 1.0
    count = int(SAMPLE_RATE * duration)

    t = np.arange(count) / SAMPLE_RATE

    base = midi_frequency(midi)

    # 微小なピッチ変化
    detune = rng.uniform(-0.0015, 0.0015)
    frequency = base * (1 + detune)

    # 音色のランダム化
    signal = np.zeros(count, dtype=np.float32)

    harmonics = rng.uniform(0.5, 1.5)

    for h in range(1, 18):
        f = frequency * h

        if f >= SAMPLE_RATE / 2:
            break

        amplitude = (
            rng.uniform(0.5, 1.5)
            / (h ** harmonics)
        )

        phase = rng.uniform(0, 2 * np.pi)

        signal += (
            amplitude *
            np.sin(2 * np.pi * f * t + phase)
        ).astype(np.float32)

    # ランダムなエンベロープ
    attack = rng.uniform(0.005, 0.08)
    decay = rng.uniform(0.1, 0.5)

    envelope = np.ones(count, dtype=np.float32)

    attack_count = max(1, int(attack * SAMPLE_RATE))

    envelope[:attack_count] = np.linspace(
        0, 1, attack_count
    )

    envelope *= np.exp(-t / decay)

    signal *= envelope

    # ノイズを少し追加
    noise_level = rng.uniform(0.0001, 0.015)

    signal += rng.normal(
        0, noise_level, count
    ).astype(np.float32)

    # 音量をランダム化
    signal *= rng.uniform(0.2, 1.0)

    return signal


def generate_dataset():
    print("================================")
    print(" Automatic Dataset Generation")
    print("================================")

    rng = np.random.default_rng(42)

    X = []
    y = []

    total = 128 * TRAIN_VARIANTS

    for midi in range(128):
        for variant in range(TRAIN_VARIANTS):

            audio = generate_sample(midi, rng)

            features = extract_features(audio)

            X.append(features)
            y.append(midi)

        if midi % 8 == 0 or midi == 127:
            progress = (midi + 1) / 128 * 100

            print(
                f"\rGenerating: {progress:5.1f}% "
                f"({(midi + 1) * TRAIN_VARIANTS}/{total})",
                end="",
                flush=True
            )

    print()

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int16)
    )


# ==========================================
# Automatic training
# ==========================================

def train():
    start = time.perf_counter()

    X, y = generate_dataset()

    print()
    print("================================")
    print(" Lightweight AI Training")
    print("================================")

    model = ExtraTreesClassifier(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=CPU_WORKERS,
        random_state=42
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH,
        compress=1
    )

    elapsed = time.perf_counter() - start

    print()
    print("Training completed!")
    print(f"Samples: {len(X)}")
    print(f"Classes: {len(set(y))}")
    print(f"Trees: {TREE_COUNT}")
    print(f"CPU workers: {CPU_WORKERS}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Saved: {MODEL_PATH}")

    return model


def load_model():
    if MODEL_PATH.exists():
        print("[AI] Loading trained model...")
        return joblib.load(MODEL_PATH)

    print("[AI] No model found.")
    print("[AI] Starting automatic training...")

    return train()


# ==========================================
# DSP analysis
# ==========================================

def dsp_predict(features):
    scores = features[:128]

    return int(np.argmax(scores))


# ==========================================
# Audio prediction
# ==========================================

def predict(audio_path, model):
    print()
    print("================================")
    print(" Absolute Pitch AI")
    print("================================")

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("音声ファイルが空です")

    features = extract_features(audio)

    dsp_note = dsp_predict(features)

    ml_probabilities = model.predict_proba(
        features.reshape(1, -1)
    )[0]

    ml_scores = np.zeros(128, dtype=np.float32)

    for index, midi in enumerate(model.classes_):
        ml_scores[int(midi)] = ml_probabilities[index]

    # Hybrid
    dsp_scores = features[:128].copy()
    dsp_scores /= np.max(dsp_scores) + 1e-10

    combined = (
        dsp_scores * 0.5
        + ml_scores * 0.5
    )

    result = int(np.argmax(combined))

    print(f"Result: {midi_name(result)}")
    print(f"MIDI: {result}")
    print(f"Frequency: {midi_frequency(result):.3f} Hz")

    print()
    print(f"DSP result: {midi_name(dsp_note)}")

    ml_note = int(model.classes_[np.argmax(ml_probabilities)])

    print(f"ML result: {midi_name(ml_note)}")

    print()
    print("Top 5:")

    ranking = np.argsort(combined)[::-1][:5]

    for index, midi in enumerate(ranking, 1):
        print(
            f"{index}. {midi_name(midi):5s} "
            f"MIDI={midi:3d} "
            f"Score={combined[midi]:.5f}"
        )


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Absolute Pitch AI"
    )

    parser.add_argument(
        "audio",
        nargs="?",
        help="Audio file to analyze"
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining"
    )

    args = parser.parse_args()

    if args.retrain and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    model = load_model()

    if args.audio:
        predict(args.audio, model)
    else:
        print()
        print("AI is ready!")
        print("Usage:")
        print('python main.py "test.wav"')


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nCancelled.")

    except Exception as error:
        print("\n[ERROR]", error)
        sys.exit(1)