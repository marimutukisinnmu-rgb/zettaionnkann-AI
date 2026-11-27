import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


# ============================================================
# Absolute Pitch AI v0.3 - Multi-analysis hybrid engine
# ============================================================

VERSION = "0.3.0"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pitch_ai.pkl"

SAMPLE_RATE = 44100
# The long analysis window is about 2.97 seconds. Zero-padding to this
# size makes the frequency grid denser; the actual low-frequency
# resolution still depends on the duration of real audio in the window.
FFT_SIZE = 131072
ATTACK_FFT_SIZE = 32768
MAX_ANALYSIS_SECONDS = 2.5
ATTACK_SECONDS = 0.35
SUSTAIN_OFFSET_SECONDS = 0.15
CQT_HOP_LENGTH = 4096
CQT_BINS = 128
CQT_BINS_PER_OCTAVE = 12

TRAIN_VARIANTS = 6
TREE_COUNT = 48
MAX_DEPTH = 20
CPU_WORKERS = 1

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]
MIDI_MIN_FREQUENCY = 440.0 * 2 ** ((0 - 69) / 12)
FEATURE_COUNT = 8 * 128 + 8


def midi_name(midi):
    midi = int(midi)
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_frequency(midi):
    return 440.0 * 2 ** ((int(midi) - 69) / 12)


def normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum > 1e-12:
        values = values / maximum
    return values.astype(np.float32, copy=False)


def _prepare_audio(audio):
    """Remove leading/trailing near-silence and retain up to 2.5 seconds."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("音声ファイルが空です")

    audio = np.nan_to_num(audio, copy=False)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        raise ValueError("音声に有効な信号がありません（ほぼ無音です）")

    active = np.flatnonzero(np.abs(audio) >= peak * 0.005)
    if active.size:
        start = int(active[0])
        end = int(active[-1]) + 1
        audio = audio[start:end]

    max_samples = int(MAX_ANALYSIS_SECONDS * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    audio = audio - np.mean(audio, dtype=np.float64)
    return audio.astype(np.float32, copy=False)


def _read_peak(spectrum, center_index, radius=1):
    left = max(0, center_index - radius)
    right = min(len(spectrum), center_index + radius + 1)
    if right <= left:
        return 0.0
    return float(np.max(spectrum[left:right]))


def _fft_components(audio, n_fft=FFT_SIZE):
    """Return direct-bin, harmonic-template, odd-harmonic and spectral stats."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 16:
        zeros = np.zeros(128, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy(), np.zeros(8, dtype=np.float32)

    # Keep the observed duration unchanged. n_fft may be larger than the
    # signal (zero-padding), but this does not manufacture new information.
    window = np.hanning(len(audio)).astype(np.float32)
    windowed = audio * window
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)).astype(np.float32)
    spectrum = normalize_scores(spectrum)

    bin_hz = SAMPLE_RATE / n_fft
    direct = np.zeros(128, dtype=np.float32)
    harmonic = np.zeros(128, dtype=np.float32)
    odd_harmonic = np.zeros(128, dtype=np.float32)

    for midi in range(128):
        fundamental = midi_frequency(midi)
        base_index = int(round(fundamental / bin_hz))
        direct[midi] = _read_peak(spectrum, base_index, radius=1)

        all_score = 0.0
        odd_score = 0.0
        for partial in range(1, 13):
            target = fundamental * partial
            if target >= SAMPLE_RATE / 2:
                break
            index = int(round(target / bin_hz))
            peak = _read_peak(spectrum, index, radius=1)
            weighted = peak / (partial ** 1.15)
            all_score += weighted
            if partial % 2 == 1:
                odd_score += weighted

        harmonic[midi] = all_score
        odd_harmonic[midi] = odd_score

    # Descriptive spectral statistics, useful to the classifier but not
    # interpreted as a separate pitch estimate.
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    spectrum_sum = float(np.sum(spectrum)) + 1e-10
    centroid = float(np.sum(frequencies * spectrum) / spectrum_sum)
    flatness = float(
        np.exp(np.mean(np.log(spectrum + 1e-10)))
        / (np.mean(spectrum) + 1e-10)
    )
    probability = spectrum / spectrum_sum
    spectral_entropy = float(-np.sum(probability * np.log(probability + 1e-10)))
    stats = np.asarray([
        centroid / (SAMPLE_RATE / 2),
        flatness,
        float(np.std(spectrum)),
        float(np.mean(spectrum)),
        float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        float(np.max(np.abs(audio))),
        min(len(audio) / (MAX_ANALYSIS_SECONDS * SAMPLE_RATE), 1.0),
        spectral_entropy / np.log(max(len(spectrum), 2)),
    ], dtype=np.float32)

    return (
        normalize_scores(direct),
        normalize_scores(harmonic),
        normalize_scores(odd_harmonic),
        np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _periodicity_features(audio):
    """FFT autocorrelation and YIN difference-function pitch estimates."""
    signal = np.asarray(audio, dtype=np.float32)
    signal = signal - np.mean(signal, dtype=np.float64)
    n = len(signal)

    autocorrelation_scores = np.zeros(128, dtype=np.float32)
    yin_scores = np.zeros(128, dtype=np.float32)
    if n < 32 or float(np.max(np.abs(signal))) < 1e-8:
        return autocorrelation_scores, yin_scores

    # FFT autocorrelation. Correct for the decreasing overlap at larger lags;
    # otherwise low-frequency notes are unfairly beaten by tiny lags.
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(signal, n=n_fft)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[:n]
    overlap = np.maximum(n - np.arange(n), 1)
    correlation = correlation / overlap
    correlation = correlation / (float(correlation[0]) + 1e-12)

    max_lag = min(n - 2, int(np.ceil(SAMPLE_RATE / MIDI_MIN_FREQUENCY)))
    if max_lag < 4:
        return autocorrelation_scores, yin_scores

    # Estimate the first full waveform period: find the first local trough,
    # then the first positive local peak after it. This avoids selecting the
    # near-1.0 correlation values at tiny lags for bass notes.
    # At very high pitches only a few samples represent one period, so this
    # estimator is intentionally limited to its more reliable lower range.
    max_acf_frequency = 2000.0
    min_period_lag = max(3, int(SAMPLE_RATE / max_acf_frequency))
    search = np.arange(min_period_lag + 1, max_lag, dtype=np.int64)
    trough_mask = (
        (correlation[search] <= correlation[search - 1])
        & (correlation[search] < correlation[search + 1])
    )
    troughs = search[trough_mask]

    if troughs.size:
        first_trough = int(troughs[0])
        peak_search = np.arange(first_trough + 1, max_lag, dtype=np.int64)
        peak_mask = (
            (correlation[peak_search] >= correlation[peak_search - 1])
            & (correlation[peak_search] > correlation[peak_search + 1])
        )
        peaks = peak_search[peak_mask]
        peaks = peaks[correlation[peaks] > 0.05]
        if peaks.size:
            # For sustained single-note audio, the earliest strong peak
            # following the first trough is the fundamental period estimate.
            peak_lag = int(peaks[0])
            estimated_frequency = SAMPLE_RATE / peak_lag
            estimated_midi = 69.0 + 12.0 * np.log2(estimated_frequency / 440.0)
            if -0.5 <= estimated_midi <= 72.5:
                midi_axis = np.arange(128, dtype=np.float64)
                distance = (midi_axis - estimated_midi) / 0.8
                strength = float(np.clip(correlation[peak_lag], 0.0, 1.0))
                autocorrelation_scores = (
                    np.exp(-0.5 * distance ** 2) * strength
                ).astype(np.float32)

    # librosa.yin implements the difference function and cumulative-mean
    # normalized difference (CMND). Limit its reliable upper range: at the
    # very highest MIDI notes the period is only 3-4 samples and YIN can make
    # octave errors. FFT and CQT handle that upper range more reliably.
    yin_frame_length = 32768
    yin_signal = signal
    if len(yin_signal) < yin_frame_length:
        yin_signal = np.pad(yin_signal, (0, yin_frame_length - len(yin_signal)))
    try:
        yin_frequencies = librosa.yin(
            yin_signal,
            fmin=MIDI_MIN_FREQUENCY,
            fmax=6000.0,
            sr=SAMPLE_RATE,
            frame_length=yin_frame_length,
            hop_length=8192,
            trough_threshold=0.1,
            center=False,
        )
        yin_frequencies = np.asarray(yin_frequencies, dtype=np.float64)
        yin_frequencies = yin_frequencies[
            np.isfinite(yin_frequencies) & (yin_frequencies > 0)
        ]
        if yin_frequencies.size:
            estimated_midi = 69.0 + 12.0 * np.log2(yin_frequencies / 440.0)
            midi_axis = np.arange(128, dtype=np.float64)
            # Average frame-wise likelihoods so mild pitch drift does not
            # force the entire recording to one arbitrary frame.
            distances = (midi_axis[:, None] - estimated_midi[None, :]) / 0.75
            yin_scores = np.mean(np.exp(-0.5 * distances ** 2), axis=1).astype(np.float32)
    except Exception as error:
        warnings.warn(f"YIN analysis skipped: {error}", RuntimeWarning)

    return autocorrelation_scores, normalize_scores(yin_scores)


def _cqt_features(audio):
    """Map the CQT bins directly to MIDI 0..127."""
    signal = np.asarray(audio, dtype=np.float32)
    # Keep enough samples for the lowest CQT filters on very short inputs.
    min_samples = int(0.5 * SAMPLE_RATE)
    if len(signal) < min_samples:
        signal = np.pad(signal, (0, min_samples - len(signal)))

    with warnings.catch_warnings():
        # librosa can emit this warning for short internal resampled CQT
        # bands; the caller's actual input is still padded and analyzed.
        warnings.filterwarnings(
            "ignore",
            message=r"n_fft=.*too large for input signal of length=.*",
            category=UserWarning,
        )
        cqt = librosa.cqt(
            signal,
            sr=SAMPLE_RATE,
            hop_length=CQT_HOP_LENGTH,
            fmin=MIDI_MIN_FREQUENCY,
            n_bins=CQT_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
            tuning=0.0,
            filter_scale=1.0,
            pad_mode="constant",
        )
    magnitude = np.abs(cqt)
    # RMS over time is less sensitive than a single-frame maximum.
    energies = np.sqrt(np.mean(magnitude ** 2, axis=1))
    return normalize_scores(energies)


def extract_feature_parts(audio):
    """Extract all complementary pitch-analysis feature groups."""
    signal = _prepare_audio(audio)

    attack_end = min(len(signal), max(16, int(ATTACK_SECONDS * SAMPLE_RATE)))
    attack = signal[:attack_end]

    sustain_start = min(
        len(signal) - 1,
        int(SUSTAIN_OFFSET_SECONDS * SAMPLE_RATE),
    )
    sustain = signal[sustain_start:]
    if len(sustain) < int(0.2 * SAMPLE_RATE):
        sustain = signal

    fundamental, harmonic, odd_harmonic, stats = _fft_components(
        signal, FFT_SIZE
    )
    attack_fundamental, attack_harmonic, _, _ = _fft_components(
        attack, ATTACK_FFT_SIZE
    )
    _, sustain_harmonic, _, _ = _fft_components(sustain, FFT_SIZE)

    autocorrelation, yin = _periodicity_features(signal)
    cqt = _cqt_features(signal)

    # Periodicity estimators become less dependable as the period approaches
    # only a few samples. Let FFT/CQT dominate when their pitch region is high.
    cqt_peak = int(np.argmax(cqt))
    if cqt_peak > 72:
        autocorrelation = np.zeros(128, dtype=np.float32)
    if cqt_peak > 108:
        yin = np.zeros(128, dtype=np.float32)

    parts = {
        "fft": fundamental,
        "harmonic": harmonic,
        "odd_harmonic": odd_harmonic,
        "attack": attack_harmonic,
        "sustain": sustain_harmonic,
        "autocorrelation": autocorrelation,
        "yin": yin,
        "cqt": cqt,
        "stats": stats,
    }
    return parts


def extract_features(audio):
    """Return the fixed-size feature vector used by ExtraTrees."""
    parts = extract_feature_parts(audio)
    features = np.concatenate([
        parts["fft"],
        parts["harmonic"],
        parts["odd_harmonic"],
        parts["attack"],
        parts["sustain"],
        parts["autocorrelation"],
        parts["yin"],
        parts["cqt"],
        parts["stats"],
    ]).astype(np.float32, copy=False)

    if features.size != FEATURE_COUNT:
        raise RuntimeError(
            f"特徴量のサイズが不正です: {features.size} != {FEATURE_COUNT}"
        )
    return features


def generate_sample(midi, rng):
    """Generate a 2.5-second synthetic training tone with varied harmonics."""
    duration = MAX_ANALYSIS_SECONDS
    count = int(SAMPLE_RATE * duration)
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE

    base = midi_frequency(midi)
    detune = rng.uniform(-0.0015, 0.0015)
    frequency = base * (1.0 + detune)
    signal = np.zeros(count, dtype=np.float32)
    harmonic_rolloff = rng.uniform(0.65, 1.7)

    for partial in range(1, 19):
        partial_frequency = frequency * partial
        if partial_frequency >= SAMPLE_RATE / 2:
            break
        amplitude = rng.uniform(0.35, 1.3) / (partial ** harmonic_rolloff)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += (
            amplitude * np.sin(2.0 * np.pi * partial_frequency * t + phase)
        ).astype(np.float32)

    attack = rng.uniform(0.005, 0.08)
    attack_count = max(2, int(attack * SAMPLE_RATE))
    envelope = np.ones(count, dtype=np.float32)
    envelope[:attack_count] = np.linspace(0.0, 1.0, attack_count)
    decay = rng.uniform(0.5, 2.0)
    sustain_level = rng.uniform(0.55, 0.9)
    envelope *= sustain_level + (1.0 - sustain_level) * np.exp(-t / decay)
    signal *= envelope

    noise_level = rng.uniform(0.00005, 0.004)
    signal += rng.normal(0.0, noise_level, count).astype(np.float32)
    signal *= rng.uniform(0.2, 1.0)
    return signal


def generate_dataset():
    print("=" * 44)
    print(" Multi-analysis synthetic dataset generation")
    print("=" * 44)

    rng = np.random.default_rng(42)
    features = []
    labels = []
    total = 128 * TRAIN_VARIANTS

    for midi in range(128):
        for _ in range(TRAIN_VARIANTS):
            sample = generate_sample(midi, rng)
            features.append(extract_features(sample))
            labels.append(midi)

        if midi % 4 == 0 or midi == 127:
            done = (midi + 1) * TRAIN_VARIANTS
            progress = done / total * 100.0
            print(
                f"\rGenerating: {progress:5.1f}% ({done}/{total})",
                end="",
                flush=True,
            )

    print()
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int16),
    )


def train():
    started = time.perf_counter()
    X, y = generate_dataset()

    print()
    print("=" * 44)
    print(" ExtraTrees multi-analysis training")
    print("=" * 44)
    model = ExtraTreesClassifier(
        n_estimators=TREE_COUNT,
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=CPU_WORKERS,
        random_state=42,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=1)

    elapsed = time.perf_counter() - started
    print("Training completed!")
    print(f"Version:       {VERSION}")
    print(f"Samples:       {len(X)}")
    print(f"Classes:       {len(set(y))}")
    print(f"Features:      {X.shape[1]}")
    print(f"Trees:         {TREE_COUNT}")
    print(f"CPU workers:   {CPU_WORKERS}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Saved:         {MODEL_PATH}")
    return model


def load_model():
    if MODEL_PATH.exists():
        print("[AI] Loading trained model...")
        model = joblib.load(MODEL_PATH)
        if getattr(model, "n_features_in_", None) == FEATURE_COUNT:
            return model
        print("[AI] Existing model uses an old feature layout.")
        print("[AI] Rebuilding the model for v0.3...")
        return train()

    print("[AI] No model found. Starting automatic training...")
    return train()


def dsp_predict(features):
    return int(np.argmax(features[:128]))


def _fuse_dsp(parts):
    # The weights are deliberately explicit so individual analyzers can be
    # benchmarked and adjusted independently in later iterations.
    weights = {
        "fft": 0.20,
        "harmonic": 0.15,
        "odd_harmonic": 0.05,
        "attack": 0.05,
        "sustain": 0.05,
        "autocorrelation": 0.05,
        "yin": 0.20,
        "cqt": 0.25,
    }
    combined = np.zeros(128, dtype=np.float32)
    for name, weight in weights.items():
        combined += normalize_scores(parts[name]) * weight
    return combined


def octave_correction(scores, odd_harmonic, harmonic):
    """Conservatively favor a lower octave when its odd partials support it."""
    corrected = np.asarray(scores, dtype=np.float32).copy()
    changed = False

    for upper in range(12, 128):
        lower = upper - 12
        # Odd partials of the lower note (1x, 3x, 5x...) cannot be explained
        # by the even-only harmonic series of the upper-octave candidate.
        odd_support = float(odd_harmonic[lower])
        harmonic_support = float(harmonic[lower])
        upper_support = float(harmonic[upper])

        if (
            odd_support >= 0.28
            and harmonic_support >= upper_support * 0.72
            and corrected[lower] >= corrected[upper] * 0.55
        ):
            boost = min(0.025, 0.035 * odd_support)
            corrected[lower] += boost
            corrected[upper] *= 0.985
            changed = True

    return corrected, changed


def _print_method_result(label, scores):
    note = int(np.argmax(scores))
    print(f"{label:20s}: {midi_name(note):5s} MIDI={note:3d}")


def predict(audio_path, model):
    print()
    print("=" * 48)
    print(f" Absolute Pitch AI v{VERSION}")
    print("=" * 48)

    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )
    parts = extract_feature_parts(audio)
    features = np.concatenate([
        parts["fft"],
        parts["harmonic"],
        parts["odd_harmonic"],
        parts["attack"],
        parts["sustain"],
        parts["autocorrelation"],
        parts["yin"],
        parts["cqt"],
        parts["stats"],
    ]).astype(np.float32, copy=False)

    if features.size != FEATURE_COUNT:
        raise RuntimeError(
            f"特徴量のサイズが不正です: {features.size} != {FEATURE_COUNT}"
        )

    dsp_scores = _fuse_dsp(parts)
    ml_probabilities = model.predict_proba(features.reshape(1, -1))[0]
    ml_scores = np.zeros(128, dtype=np.float32)
    for index, midi in enumerate(model.classes_):
        ml_scores[int(midi)] = float(ml_probabilities[index])

    # Keep DSP as the stronger source so a small synthetic ML model cannot
    # completely override independent signal-processing evidence.
    combined = dsp_scores * 0.80 + ml_scores * 0.20
    combined, corrected = octave_correction(
        combined,
        parts["odd_harmonic"],
        parts["harmonic"],
    )

    result = int(np.argmax(combined))
    print(f"Result: {midi_name(result)}")
    print(f"MIDI: {result}")
    print(f"Frequency: {midi_frequency(result):.3f} Hz")
    print(f"Octave correction: {'applied' if corrected else 'no change'}")

    print()
    print("Individual analyzer results:")
    _print_method_result("FFT fundamental", parts["fft"])
    _print_method_result("Harmonic template", parts["harmonic"])
    _print_method_result("Autocorrelation", parts["autocorrelation"])
    _print_method_result("YIN", parts["yin"])
    _print_method_result("CQT", parts["cqt"])
    _print_method_result("Attack analysis", parts["attack"])
    _print_method_result("Sustain analysis", parts["sustain"])
    _print_method_result("DSP fusion", dsp_scores)
    _print_method_result("ML", ml_scores)

    print()
    print("Top 5:")
    ranking = np.argsort(combined)[::-1][:5]
    for index, midi in enumerate(ranking, 1):
        print(
            f"{index}. {midi_name(midi):5s} "
            f"MIDI={midi:3d} Score={combined[midi]:.5f}"
        )

    return {
        "midi": result,
        "note": midi_name(result),
        "frequency": midi_frequency(result),
        "scores": combined,
        "analyzers": {
            name: int(np.argmax(parts[name]))
            for name in (
                "fft", "harmonic", "autocorrelation", "yin", "cqt",
                "attack", "sustain"
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Absolute Pitch AI - FFT + Autocorrelation + YIN + CQT"
    )
    parser.add_argument("audio", nargs="?", help="Audio file to analyze")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force rebuilding the v0.3 multi-analysis model",
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
        print("Rebuild model:")
        print("python main.py --retrain")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as error:
        print("\n[ERROR]", error)
        sys.exit(1)
