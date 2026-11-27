import warnings
from pathlib import Path

import librosa
import numpy as np

VERSION = "0.4.2"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pitch_reranker_v042.pkl"  # Independent v0.4.2 model; never overwrite v0.4.0/v0.4.1
SAMPLE_RATE = 44100
FFT_SIZE = 131072
ATTACK_FFT_SIZE = 32768
MAX_ANALYSIS_SECONDS = 2.5
ATTACK_SECONDS = 0.35
SUSTAIN_OFFSET_SECONDS = 0.15
CQT_HOP_LENGTH = 4096
CQT_BINS = 128
CQT_BINS_PER_OCTAVE = 12
TRAIN_VARIANTS = 8
TREE_COUNT = 96
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 2
CPU_WORKERS = 1
YIN_FRAME_LENGTH = 32768
YIN_FMIN = 440.0 * 2 ** ((0 - 69) / 12)
YIN_FMAX = 2200.0
YIN_HIGH_MIN_MIDI = 96
YIN_HIGH_FMIN = 440.0 * 2 ** ((YIN_HIGH_MIN_MIDI - 69) / 12)
YIN_HIGH_FMAX = 440.0 * 2 ** ((127 - 69) / 12)
HIGH_YIN_TRIGGER = 0.035
HIGH_YIN_SCORE_SCALE = 0.65
YIN_THRESHOLD = 0.20
CANDIDATE_TOP_K = 4
SPECTRAL_PEAK_CANDIDATES = 16
SPECTRAL_PEAK_MIN_RELATIVE = 0.05
SPECTRAL_PEAK_MIN_ABSOLUTE = 0.020
SPECTRAL_MATCH_TOLERANCE_CENTS = 35.0
SPECTRAL_HARMONIC_FEATURE_LIMIT = 8
CANDIDATE_OFFSETS = (-38, -34, -31, -28, -24, -19, -12, 0, 12, 19, 24, 28, 31, 34, 38)
LOW_CONF_RELATIVE_MARGIN = 0.14
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MIDI_FREQUENCIES = 440.0 * np.power(2.0, (np.arange(128) - 69) / 12.0)
MIDI_MIN_FREQUENCY = float(MIDI_FREQUENCIES[0])
HARMONIC_NUMBERS = np.arange(1, 17, dtype=np.float64)
CANDIDATE_FEATURE_NAMES = ("midi_norm", "pc_sin", "pc_cos", "fft", "harmonic", "odd_harmonic", "harmonic_coverage", "yin", "autocorrelation", "lower_fft", "lower_harmonic", "lower_odd", "upper_harmonic", "fft_rank", "harmonic_rank", "yin_rank", "acf_rank", "pitch_class_support", "distance_to_yin", "distance_to_fft", "duration_norm", "rms_norm", "yin_global_confidence", "spectral_fundamental_band", "spectral_upper_harmonic_band", "spectral_subharmonic_band", "spectral_subharmonic_ratio", "high_yin_candidate_support")
CANDIDATE_FEATURE_COUNT = len(CANDIDATE_FEATURE_NAMES)

class InvalidAudioError(ValueError):
    pass

def midi_name(midi):
    midi = int(midi)
    return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'

def midi_frequency(midi):
    return float(MIDI_FREQUENCIES[int(midi)])

def normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum > 1e-12:
        values = values / maximum
    return values.astype(np.float32, copy=False)

def _prepare_audio(audio):
    """Validate, trim low-level leading/trailing silence, and cap duration."""
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise InvalidAudioError('empty audio')
    if not np.all(np.isfinite(signal)):
        raise InvalidAudioError('non-finite samples')
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-08:
        raise InvalidAudioError('silent audio')
    active = np.flatnonzero(np.abs(signal) >= peak * 0.005)
    if active.size == 0:
        raise InvalidAudioError('no active samples')
    start = int(active[0])
    end = int(active[-1]) + 1
    if end - start < int(0.05 * SAMPLE_RATE):
        raise InvalidAudioError('active audio shorter than 50 ms')
    signal = signal[start:end]
    if signal.size > int(MAX_ANALYSIS_SECONDS * SAMPLE_RATE):
        signal = signal[:int(MAX_ANALYSIS_SECONDS * SAMPLE_RATE)]
    signal = signal - np.mean(signal, dtype=np.float64)
    if float(np.max(np.abs(signal))) < 1e-08:
        raise InvalidAudioError('signal became silent after DC removal')
    return signal.astype(np.float32, copy=False)

def _read_peak(spectrum, center_index, radius=1):
    left = max(0, int(center_index) - int(radius))
    right = min(len(spectrum), int(center_index) + int(radius) + 1)
    if right <= left:
        return 0.0
    return float(np.max(spectrum[left:right]))

def _spectral_analysis(audio, n_fft=FFT_SIZE, include_stats=True):
    """Compute one spectrum and derive pitch/partial features from it."""
    signal = np.asarray(audio, dtype=np.float32)
    if signal.size < 16:
        zeros = np.zeros(128, dtype=np.float32)
        return {'spectrum': np.zeros(n_fft // 2 + 1, dtype=np.float32), 'fft': zeros.copy(), 'harmonic': zeros.copy(), 'odd_harmonic': zeros.copy(), 'coverage': zeros.copy(), 'stats': np.zeros(4, dtype=np.float32)}
    windowed = signal * np.hanning(len(signal)).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windowed, n=n_fft)).astype(np.float32)
    spectrum = normalize_scores(magnitude)
    bin_hz = SAMPLE_RATE / n_fft
    targets = MIDI_FREQUENCIES[:, None] * HARMONIC_NUMBERS[None, :]
    valid = targets < SAMPLE_RATE / 2
    indices = np.rint(targets / bin_hz).astype(np.int64)
    indices = np.clip(indices, 0, len(spectrum) - 1)
    harmonic_peaks = np.zeros_like(targets, dtype=np.float32)
    for offset in (-2, -1, 0, 1, 2):
        shifted = np.clip(indices + offset, 0, len(spectrum) - 1)
        harmonic_peaks = np.maximum(harmonic_peaks, spectrum[shifted])
    harmonic_peaks *= valid
    direct = harmonic_peaks[:, 0].copy()
    weighted = harmonic_peaks / np.power(HARMONIC_NUMBERS[None, :], 1.15)
    harmonic = np.sum(weighted, axis=1)
    odd_mask = HARMONIC_NUMBERS.astype(np.int64) % 2 == 1
    odd_harmonic = np.sum(weighted[:, odd_mask], axis=1)
    valid_count = np.maximum(np.sum(valid, axis=1), 1)
    coverage = np.sum((harmonic_peaks > 0.025) & valid, axis=1) / valid_count
    if include_stats:
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
        spectrum_sum = float(np.sum(spectrum)) + 1e-10
        centroid = float(np.sum(frequencies * spectrum) / spectrum_sum)
        rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
        stats = np.asarray([centroid / (SAMPLE_RATE / 2), min(len(signal) / (MAX_ANALYSIS_SECONDS * SAMPLE_RATE), 1.0), min(rms * 4.0, 1.0), min(float(np.max(np.abs(signal))) * 2.0, 1.0)], dtype=np.float32)
    else:
        stats = np.zeros(4, dtype=np.float32)
    return {'spectrum': spectrum, 'fft': normalize_scores(direct), 'harmonic': normalize_scores(harmonic), 'odd_harmonic': normalize_scores(odd_harmonic), 'coverage': np.clip(coverage, 0.0, 1.0).astype(np.float32), 'stats': np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)}

def _yin_frame(signal, frame_length=YIN_FRAME_LENGTH, fmin=YIN_FMIN, fmax=YIN_FMAX):
    """A compact YIN implementation with CMND confidence and autocorrelation."""
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size))
    elif x.size > frame_length:
        start = (x.size - frame_length) // 2
        x = x[start:start + frame_length]
    x = x - np.mean(x, dtype=np.float64)
    if float(np.max(np.abs(x))) < 1e-08:
        return ([], None)
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(fft * np.conjugate(fft), n=n_fft)[:n]
    tau_min = max(2, int(np.floor(SAMPLE_RATE / fmax)))
    tau_max = min(n - 2, int(np.ceil(SAMPLE_RATE / fmin)))
    if tau_max <= tau_min:
        return ([], None)
    prefix = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    taus = np.arange(1, tau_max + 1, dtype=np.int64)
    left_energy = prefix[n - taus]
    right_energy = prefix[n] - prefix[taus]
    difference = left_energy + right_energy - 2.0 * acf[taus]
    difference = np.maximum(difference, 0.0)
    cumulative = np.cumsum(difference) + 1e-12
    cmnd = difference * taus / cumulative
    lag_values = np.arange(1, tau_max + 1)
    selected = (lag_values >= tau_min) & (lag_values <= tau_max)
    candidates_idx = np.flatnonzero(selected)
    lag_indices = np.arange(1, len(cmnd) - 1, dtype=np.int64)
    minima_mask = (lag_indices + 1 >= tau_min) & (lag_indices + 1 <= tau_max) & (cmnd[lag_indices] <= cmnd[lag_indices - 1]) & (cmnd[lag_indices] < cmnd[lag_indices + 1])
    local_minima = lag_indices[minima_mask]
    if local_minima.size == 0:
        local_minima = np.asarray([int(candidates_idx[np.argmin(cmnd[candidates_idx])])], dtype=np.int64)
    below_threshold = local_minima[cmnd[local_minima] <= YIN_THRESHOLD]
    if not below_threshold.size:
        # No CMND minimum passed the YIN threshold: abstain instead of
        # turning an unreliable periodicity estimate into a pitch vote.
        return ([], 0.0)
    primary = int(below_threshold[0])
    remaining = local_minima[local_minima != primary]
    remaining = remaining[np.argsort(cmnd[remaining])]
    local_minima = np.concatenate(([primary], remaining[:8]))
    results = []
    seen_notes = set()
    for candidate_order, idx in enumerate(local_minima):
        idx = int(idx)
        cmnd_value = float(cmnd[idx])
        if cmnd_value > 0.45 and results:
            continue
        refined_lag = float(idx + 1)
        if 0 < idx < len(cmnd) - 1:
            left, center, right = (float(cmnd[idx - 1]), float(cmnd[idx]), float(cmnd[idx + 1]))
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                refined_lag += 0.5 * (left - right) / denominator
        frequency = SAMPLE_RATE / max(refined_lag, 1e-06)
        if frequency < fmin or frequency > fmax:
            continue
        midi_float = 69.0 + 12.0 * np.log2(frequency / 440.0)
        midi = int(np.clip(np.rint(midi_float), 0, 127))
        if midi in seen_notes:
            continue
        seen_notes.add(midi)
        lag = int(np.clip(round(refined_lag), 1, n - 1))
        denom = np.sqrt(max(left_energy[lag - 1] * right_energy[lag - 1], 1e-18))
        acf_strength = float(np.clip(acf[lag] / denom, -1.0, 1.0))
        confidence = float(np.clip(1.0 - cmnd_value, 0.0, 1.0))
        priority = 1.0 if candidate_order == 0 else 0.55
        results.append({'midi': midi, 'midi_float': float(midi_float), 'frequency': float(frequency), 'cmnd': cmnd_value, 'confidence': confidence, 'priority': priority, 'acf': acf_strength})
        if len(results) >= 3:
            break
    global_confidence = results[0]['confidence'] if results else 0.0
    return (results, global_confidence)

def _periodicity_analysis(signal, multi_frame=False, fmin=YIN_FMIN, fmax=YIN_FMAX, candidate_priority=1.0):
    """Run YIN over one or three frames within a configurable frequency band."""
    x = np.asarray(signal, dtype=np.float32)
    if multi_frame and len(x) > YIN_FRAME_LENGTH:
        starts = [0, (len(x) - YIN_FRAME_LENGTH) // 2, len(x) - YIN_FRAME_LENGTH]
    else:
        starts = [max(0, (len(x) - YIN_FRAME_LENGTH) // 2)]
    yin_scores = np.zeros(128, dtype=np.float32)
    acf_scores = np.zeros(128, dtype=np.float32)
    all_candidates = []
    frame_confidences = []
    for start in starts:
        frame = x[start:start + YIN_FRAME_LENGTH]
        candidates, frame_confidence = _yin_frame(frame, fmin=fmin, fmax=fmax)
        if frame_confidence is not None:
            frame_confidences.append(frame_confidence)
        for candidate in candidates:
            midi = candidate['midi']
            confidence = candidate['confidence']
            priority = candidate.get('priority', 1.0) * candidate_priority
            width = 0.7
            note_axis = np.arange(128, dtype=np.float32)
            likelihood = np.exp(-0.5 * ((note_axis - candidate['midi_float']) / width) ** 2)
            yin_scores = np.maximum(yin_scores, likelihood * confidence * priority)
            acf_scores[midi] = max(acf_scores[midi], max(0.0, candidate['acf']) * priority)
            all_candidates.append({**candidate, 'priority': priority})
    best_by_note = {}
    for item in all_candidates:
        current = best_by_note.get(item['midi'])
        if current is None or item['confidence'] * item.get('priority', 1.0) > current['confidence'] * current.get('priority', 1.0):
            best_by_note[item['midi']] = item
    candidates = sorted(best_by_note.values(), key=lambda item: item['confidence'] * item.get('priority', 1.0), reverse=True)[:6]
    global_confidence = float(np.median(frame_confidences)) if frame_confidences else 0.0
    return {'yin': normalize_scores(yin_scores), 'acf': normalize_scores(acf_scores), 'yin_candidates': candidates, 'yin_confidence': global_confidence}

def _cqt_analysis(signal):
    x = np.asarray(signal, dtype=np.float32)
    minimum = int(0.5 * SAMPLE_RATE)
    if len(x) < minimum:
        x = np.pad(x, (0, minimum - len(x)))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='n_fft=.*too large for input signal of length=.*', category=UserWarning)
        cqt = librosa.cqt(x, sr=SAMPLE_RATE, hop_length=CQT_HOP_LENGTH, fmin=MIDI_MIN_FREQUENCY, n_bins=CQT_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE, tuning=0.0, filter_scale=1.0, pad_mode='constant')
    magnitude = np.abs(cqt)
    energies = np.sqrt(np.mean(magnitude ** 2, axis=1))
    return normalize_scores(energies)

def _top_indices(scores, count=CANDIDATE_TOP_K):
    values = np.asarray(scores, dtype=np.float32)
    if values.size != 128 or float(np.max(values)) <= 1e-10:
        return []
    return [int(value) for value in np.argsort(values)[::-1][:count]]

def _spectral_band_peak(spectrum, frequency, tolerance_cents=SPECTRAL_MATCH_TOLERANCE_CENTS):
    """Return the strongest FFT magnitude within a cents-based frequency window."""
    if frequency <= 0.0 or frequency >= SAMPLE_RATE / 2:
        return 0.0
    bin_hz = SAMPLE_RATE / FFT_SIZE
    ratio = 2.0 ** (float(tolerance_cents) / 1200.0)
    low_hz = frequency / ratio
    high_hz = frequency * ratio
    left = max(1, int(np.floor(low_hz / bin_hz)))
    right = min(len(spectrum), int(np.ceil(high_hz / bin_hz)) + 1)
    if right <= left:
        return 0.0
    return float(np.max(spectrum[left:right]))


def _spectral_candidate_evidence(spectrum, midi):
    """Extract frequency-tolerant fundamental, partial, and subharmonic evidence."""
    base_frequency = midi_frequency(midi)
    fundamental = _spectral_band_peak(spectrum, base_frequency)
    harmonic_values = []
    harmonic_weights = []
    for harmonic in range(2, SPECTRAL_HARMONIC_FEATURE_LIMIT + 1):
        target = base_frequency * harmonic
        if target >= SAMPLE_RATE / 2:
            break
        weight = 1.0 / (harmonic ** 1.15)
        harmonic_values.append(_spectral_band_peak(spectrum, target) * weight)
        harmonic_weights.append(weight)
    upper_harmonic = (
        float(sum(harmonic_values) / sum(harmonic_weights))
        if harmonic_weights else 0.0
    )
    subharmonic_values = []
    for divisor in range(2, SPECTRAL_HARMONIC_FEATURE_LIMIT + 1):
        target = base_frequency / divisor
        if target < MIDI_MIN_FREQUENCY:
            continue
        subharmonic_values.append(_spectral_band_peak(spectrum, target))
    subharmonic = max(subharmonic_values, default=0.0)
    subharmonic_ratio = subharmonic / (fundamental + subharmonic + 1e-6)
    return (
        float(fundamental),
        float(upper_harmonic),
        float(subharmonic),
        float(np.clip(subharmonic_ratio, 0.0, 1.0)),
    )


def _parabolic_peak_offset(spectrum, peak_bin):
    """Estimate a sub-bin FFT peak position using a log-magnitude parabola."""
    index = int(peak_bin)
    if index <= 0 or index >= len(spectrum) - 1:
        return 0.0
    left, center, right = np.log(np.maximum(spectrum[index - 1:index + 2], 1e-12))
    denominator = float(left - 2.0 * center + right)
    if abs(denominator) <= 1e-12:
        return 0.0
    offset = 0.5 * float(left - right) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def _spectral_peak_candidates(parts, max_candidates=SPECTRAL_PEAK_CANDIDATES):
    """Map refined FFT peaks to MIDI candidates, reserving room for high notes.

    Half of the initial slots are reserved for MIDI 96-127 candidates so a
    dense lower-frequency spectrum cannot crowd all high-register peaks out.
    """
    spectrum = np.asarray(parts.get('spectrum', ()), dtype=np.float32)
    if spectrum.size < 3 or float(np.max(spectrum)) <= 1e-10:
        return []
    bin_hz = SAMPLE_RATE / FFT_SIZE
    peak_mask = (spectrum[1:-1] >= spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
    peak_bins = np.flatnonzero(peak_mask) + 1
    if peak_bins.size == 0:
        return []
    peak_values = spectrum[peak_bins]
    threshold = max(
        SPECTRAL_PEAK_MIN_ABSOLUTE,
        float(np.max(spectrum)) * SPECTRAL_PEAK_MIN_RELATIVE,
    )
    eligible = peak_values >= threshold
    peak_bins = peak_bins[eligible]
    peak_values = peak_values[eligible]
    if peak_bins.size == 0:
        return []
    order = np.argsort(peak_values)[::-1]
    low_candidates = []
    high_candidates = []
    ordered_candidates = []
    seen = set()
    min_frequency = MIDI_FREQUENCIES[0]
    max_frequency = MIDI_FREQUENCIES[127]
    for order_index in order:
        peak_bin = int(peak_bins[order_index])
        refined_bin = peak_bin + _parabolic_peak_offset(spectrum, peak_bin)
        frequency = refined_bin * bin_hz
        if frequency < min_frequency or frequency > max_frequency:
            continue
        midi_float = 69.0 + 12.0 * np.log2(frequency / 440.0)
        midi = int(np.rint(midi_float))
        if not 0 <= midi <= 127 or abs(midi_float - midi) > 0.50:
            continue
        if midi in seen:
            continue
        seen.add(midi)
        ordered_candidates.append(midi)
        if midi >= YIN_HIGH_MIN_MIDI:
            high_candidates.append(midi)
        else:
            low_candidates.append(midi)
    low_quota = max_candidates // 2
    high_quota = max_candidates - low_quota
    selected = low_candidates[:low_quota] + high_candidates[:high_quota]
    selected_set = set(selected)
    for midi in ordered_candidates:
        if len(selected) >= max_candidates:
            break
        if midi not in selected_set:
            selected.append(midi)
            selected_set.add(midi)
    return selected

def _generate_candidates(parts):
    candidates = set()
    for key in ('fft', 'harmonic', 'yin', 'acf', 'cqt'):
        scores = parts.get(key)
        if scores is None:
            continue
        for top in _top_indices(scores):
            for offset in CANDIDATE_OFFSETS:
                candidate = top + offset
                if 0 <= candidate <= 127:
                    candidates.add(candidate)
    # Direct spectral peaks provide an additional path for high notes that
    # may be crowded out of an analyzer's Top-K list by lower harmonics.
    candidates.update(_spectral_peak_candidates(parts))
    return sorted(candidates)

def _rank_array(scores):
    array = np.asarray(scores, dtype=np.float32)
    order = np.argsort(array)[::-1]
    ranks = np.empty(128, dtype=np.float32)
    ranks[order] = np.linspace(1.0, 0.0, 128, dtype=np.float32)
    if float(np.max(array)) <= 1e-10:
        ranks.fill(0.0)
    return ranks

def _candidate_feature_matrix(parts, candidates, duration_seconds):
    fft_rank = _rank_array(parts['fft'])
    harmonic_rank = _rank_array(parts['harmonic'])
    yin_rank = _rank_array(parts['yin'])
    acf_rank = _rank_array(parts['acf'])
    fft_top = int(np.argmax(parts['fft'])) if np.max(parts['fft']) > 0 else 0
    yin_top = int(np.argmax(parts['yin'])) if np.max(parts['yin']) > 0 else 0
    pitch_class_support = np.zeros(128, dtype=np.float32)
    family_scores = (parts['fft'], parts['harmonic'], parts['yin'])
    for midi in range(128):
        pc = midi % 12
        pitch_class_support[midi] = sum((float(np.max(score[pc::12])) for score in family_scores)) / len(family_scores)
    spectrum = np.asarray(parts.get('spectrum', ()), dtype=np.float32)
    high_yin = np.asarray(parts.get('yin_high', np.zeros(128)), dtype=np.float32)
    rows = []
    for midi in candidates:
        lower = midi - 12
        upper = midi + 12
        pc_angle = 2.0 * np.pi * (midi % 12) / 12.0
        spectral_evidence = _spectral_candidate_evidence(spectrum, midi) if spectrum.size else (0.0, 0.0, 0.0, 0.0)
        row = [
            midi / 127.0,
            float(np.sin(pc_angle)),
            float(np.cos(pc_angle)),
            float(parts['fft'][midi]),
            float(parts['harmonic'][midi]),
            float(parts['odd_harmonic'][midi]),
            float(parts['coverage'][midi]),
            float(parts['yin'][midi]),
            float(parts['acf'][midi]),
            float(parts['fft'][lower]) if lower >= 0 else 0.0,
            float(parts['harmonic'][lower]) if lower >= 0 else 0.0,
            float(parts['odd_harmonic'][lower]) if lower >= 0 else 0.0,
            float(parts['harmonic'][upper]) if upper <= 127 else 0.0,
            float(fft_rank[midi]),
            float(harmonic_rank[midi]),
            float(yin_rank[midi]),
            float(acf_rank[midi]),
            float(pitch_class_support[midi]),
            abs(midi - yin_top) / 127.0,
            abs(midi - fft_top) / 127.0,
            min(duration_seconds / MAX_ANALYSIS_SECONDS, 1.0),
            float(parts['stats'][2]),
            float(parts['yin_confidence']),
            spectral_evidence[0],
            spectral_evidence[1],
            spectral_evidence[2],
            spectral_evidence[3],
            float(high_yin[midi]),
        ]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)

def _analyze_tier0(signal):
    spectral = _spectral_analysis(signal, FFT_SIZE, include_stats=True)
    periodic = _periodicity_analysis(signal, multi_frame=False)
    return {**spectral, **periodic, 'cqt': np.zeros(128, dtype=np.float32), 'attack': np.zeros(128, dtype=np.float32), 'sustain': np.zeros(128, dtype=np.float32), 'tier': 0}

def _augment_high_band_yin(signal, parts):
    """Run a separate high-register YIN pass only when the spectrum supports it."""
    updated = dict(parts)
    updated['yin_high'] = np.zeros(128, dtype=np.float32)
    updated['acf_high'] = np.zeros(128, dtype=np.float32)
    updated['high_yin_confidence'] = 0.0
    spectrum = np.asarray(parts.get('spectrum', ()), dtype=np.float32)
    if spectrum.size < 3:
        return updated
    bin_hz = SAMPLE_RATE / FFT_SIZE
    low_bin = max(1, int(np.floor(YIN_HIGH_FMIN / bin_hz)))
    high_bin = min(len(spectrum), int(np.ceil(YIN_HIGH_FMAX / bin_hz)) + 1)
    high_band_peak = float(np.max(spectrum[low_bin:high_bin])) if high_bin > low_bin else 0.0
    if high_band_peak < HIGH_YIN_TRIGGER:
        return updated
    high = _periodicity_analysis(
        signal,
        multi_frame=False,
        fmin=YIN_HIGH_FMIN,
        fmax=YIN_HIGH_FMAX,
        candidate_priority=HIGH_YIN_SCORE_SCALE,
    )
    updated['yin_high'] = high['yin']
    updated['acf_high'] = high['acf']
    updated['high_yin_confidence'] = float(high['yin_confidence'])
    updated['yin'] = np.maximum(
        np.asarray(parts['yin'], dtype=np.float32),
        high['yin'] * HIGH_YIN_SCORE_SCALE,
    )
    updated['acf'] = np.maximum(
        np.asarray(parts['acf'], dtype=np.float32),
        high['acf'] * HIGH_YIN_SCORE_SCALE,
    )
    merged = list(parts.get('yin_candidates', []))
    by_note = {item['midi']: item for item in merged}
    for item in high.get('yin_candidates', []):
        current = by_note.get(item['midi'])
        if current is None or item['confidence'] * item.get('priority', 1.0) > current['confidence'] * current.get('priority', 1.0):
            by_note[item['midi']] = item
    updated['yin_candidates'] = sorted(
        by_note.values(),
        key=lambda item: item['confidence'] * item.get('priority', 1.0),
        reverse=True,
    )[:8]
    return updated


def _analyze_tier1(signal, tier0):
    augmented = _augment_high_band_yin(signal, tier0)
    cqt = _cqt_analysis(signal)
    return {**augmented, 'cqt': cqt, 'tier': 1}

def _analyze_tier2(signal, parts):
    attack_end = min(len(signal), max(16, int(ATTACK_SECONDS * SAMPLE_RATE)))
    attack_signal = signal[:attack_end]
    sustain_start = min(len(signal) - 1, int(SUSTAIN_OFFSET_SECONDS * SAMPLE_RATE))
    sustain_signal = signal[sustain_start:]
    if len(sustain_signal) < int(0.2 * SAMPLE_RATE):
        sustain_signal = signal
    attack = _spectral_analysis(attack_signal, ATTACK_FFT_SIZE, include_stats=False)
    sustain = _spectral_analysis(sustain_signal, FFT_SIZE, include_stats=False)
    multi_periodic = _periodicity_analysis(signal, multi_frame=True)
    updated = dict(parts)
    updated['attack'] = attack['harmonic']
    updated['sustain'] = sustain['harmonic']
    updated['yin_multi'] = np.maximum(
        multi_periodic['yin'],
        np.asarray(parts.get('yin_high', np.zeros(128)), dtype=np.float32) * HIGH_YIN_SCORE_SCALE,
    )
    updated['yin_confidence_multi'] = max(
        float(multi_periodic['yin_confidence']),
        float(parts.get('high_yin_confidence', 0.0)),
    )
    updated['tier'] = 2
    return updated
