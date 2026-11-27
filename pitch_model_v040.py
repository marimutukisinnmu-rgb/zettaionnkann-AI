import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import pitch_dsp_v040 as dsp

VERSION = dsp.VERSION
BASE_DIR = dsp.BASE_DIR
MODEL_PATH = dsp.MODEL_PATH
SAMPLE_RATE = dsp.SAMPLE_RATE
MAX_ANALYSIS_SECONDS = dsp.MAX_ANALYSIS_SECONDS
TRAIN_VARIANTS = dsp.TRAIN_VARIANTS
TREE_COUNT = dsp.TREE_COUNT
MAX_DEPTH = dsp.MAX_DEPTH
MIN_SAMPLES_LEAF = dsp.MIN_SAMPLES_LEAF
CPU_WORKERS = dsp.CPU_WORKERS
CANDIDATE_FEATURE_COUNT = dsp.CANDIDATE_FEATURE_COUNT
CANDIDATE_FEATURE_NAMES = dsp.CANDIDATE_FEATURE_NAMES
midi_frequency = dsp.midi_frequency
_prepare_audio = dsp._prepare_audio
_analyze_tier0 = dsp._analyze_tier0
_generate_candidates = dsp._generate_candidates
_candidate_feature_matrix = dsp._candidate_feature_matrix

def generate_sample(midi, rng):
    """Generate varied synthetic monophonic tones for the reranker."""
    duration = MAX_ANALYSIS_SECONDS
    count = int(SAMPLE_RATE * duration)
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    base = midi_frequency(midi)
    detune = rng.uniform(-0.0015, 0.0015)
    frequency = base * (1.0 + detune)
    signal = np.zeros(count, dtype=np.float32)
    rolloff = rng.uniform(0.55, 1.8)
    fundamental_gain = rng.uniform(0.15, 1.4)
    partial_count = int(rng.integers(5, 19))
    inharmonicity = rng.uniform(0.0, 0.00015)
    for partial in range(1, partial_count + 1):
        partial_frequency = frequency * partial * np.sqrt(1.0 + inharmonicity * partial * partial)
        if partial_frequency >= SAMPLE_RATE / 2:
            break
        amplitude = rng.uniform(0.2, 1.4) / partial ** rolloff
        if partial == 1:
            amplitude *= fundamental_gain
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += (amplitude * np.sin(2.0 * np.pi * partial_frequency * t + phase)).astype(np.float32)
    attack = rng.uniform(0.003, 0.1)
    attack_count = max(2, int(attack * SAMPLE_RATE))
    envelope = np.ones(count, dtype=np.float32)
    envelope[:attack_count] = np.linspace(0.0, 1.0, attack_count)
    decay = rng.uniform(0.45, 2.2)
    sustain_level = rng.uniform(0.35, 0.95)
    envelope *= sustain_level + (1.0 - sustain_level) * np.exp(-t / decay)
    signal *= envelope
    noise = rng.uniform(1e-05, 0.002)
    signal += rng.normal(0.0, noise, count).astype(np.float32)
    signal *= rng.uniform(0.15, 1.0)
    return signal

def _training_parts(signal):
    prepared = _prepare_audio(signal)
    return _analyze_tier0(prepared)

def train():
    started = time.perf_counter()
    rng = np.random.default_rng(20261127)
    feature_rows = []
    labels = []
    candidate_coverage = 0
    candidate_total = 0
    audio_count = 128 * TRAIN_VARIANTS
    print("=" * 52)
    print(" Absolute Pitch AI v0.4 Candidate Reranker Training")
    print("=" * 52)
    for midi in range(128):
        for _ in range(TRAIN_VARIANTS):
            audio = generate_sample(midi, rng)
            parts = _training_parts(audio)
            candidates = _generate_candidates(parts)
            candidate_total += 1
            if midi in candidates:
                candidate_coverage += 1
            else:
                candidates.append(midi)
                candidates.sort()
            matrix = _candidate_feature_matrix(parts, candidates, len(audio) / SAMPLE_RATE)
            feature_rows.append(matrix)
            labels.append(np.asarray([int(candidate == midi) for candidate in candidates], dtype=np.int8))
        if midi % 4 == 0 or midi == 127:
            done = (midi + 1) * TRAIN_VARIANTS
            print(f"\rGenerating candidates: {done / audio_count * 100:5.1f}% ({done}/{audio_count})", end="", flush=True)
    print()
    X = np.vstack(feature_rows).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.int8, copy=False)
    print("Training ExtraTrees candidate reranker...")
    model = ExtraTreesClassifier(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=0.9,
        class_weight={0: 1.0, 1: 4.0},
        n_jobs=CPU_WORKERS,
        random_state=42,
    )
    model.fit(X, y)
    payload = {
        "version": VERSION,
        "feature_count": CANDIDATE_FEATURE_COUNT,
        "feature_names": CANDIDATE_FEATURE_NAMES,
        "model": model,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH, compress=1)
    elapsed = time.perf_counter() - started
    print("Training completed!")
    print(f"Audio samples:       {audio_count}")
    print(f"Candidate rows:      {len(y)}")
    print(f"Features/candidate:  {CANDIDATE_FEATURE_COUNT}")
    print(f"Candidate recall:    {candidate_coverage}/{candidate_total} ({candidate_coverage / max(candidate_total, 1) * 100:.2f}%)")
    print(f"Positive rows:       {int(np.sum(y))}")
    print(f"Trees:               {TREE_COUNT}")
    print(f"Training time:       {elapsed:.2f} seconds")
    print(f"Saved:               {MODEL_PATH}")
    print("Note: synthetic-data training is not a real-instrument accuracy guarantee.")
    return model

def load_model():
    if MODEL_PATH.exists():
        print("[AI] Loading v0.4 candidate reranker...")
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict) and payload.get("version") == VERSION and payload.get("feature_count") == CANDIDATE_FEATURE_COUNT and "model" in payload:
            return payload["model"]
        print("[AI] Existing model format is incompatible; rebuilding v0.4 model.")
    else:
        print("[AI] No v0.4 model found. Starting automatic training...")
    return train()
