import time
from pathlib import Path

import librosa
import numpy as np

import pitch_dsp_v041 as dsp

VERSION = dsp.VERSION
SAMPLE_RATE = dsp.SAMPLE_RATE
MAX_ANALYSIS_SECONDS = dsp.MAX_ANALYSIS_SECONDS
LOW_CONF_RELATIVE_MARGIN = dsp.LOW_CONF_RELATIVE_MARGIN
InvalidAudioError = dsp.InvalidAudioError
midi_name = dsp.midi_name
midi_frequency = dsp.midi_frequency
normalize_scores = dsp.normalize_scores
_prepare_audio = dsp._prepare_audio
_analyze_tier0 = dsp._analyze_tier0
_analyze_tier1 = dsp._analyze_tier1
_analyze_tier2 = dsp._analyze_tier2
_generate_candidates = dsp._generate_candidates
_candidate_feature_matrix = dsp._candidate_feature_matrix

def _score_margin(scores):
    values = np.asarray(scores, dtype=np.float32)
    if values.size < 2:
        return 1.0
    ranking = np.argsort(values)[::-1]
    best = float(values[ranking[0]])
    second = float(values[ranking[1]])
    return (best - second) / (abs(best) + 1e-09)

def _tier0_early_decision(parts):
    """Only early-exit when independent periodic and spectral estimates agree."""
    if not parts['yin_candidates']:
        return None
    yin = parts['yin_candidates'][0]
    yin_midi = int(yin['midi'])
    fft_midi = int(np.argmax(parts['fft']))
    harmonic_midi = int(np.argmax(parts['harmonic']))
    if parts['yin_confidence'] < 0.93 or yin['cmnd'] > 0.12:
        return None
    if not yin_midi == fft_midi == harmonic_midi:
        return None
    if _score_margin(parts['harmonic']) < 0.08:
        return None
    return {'midi': yin_midi, 'confidence': float(min(parts['yin_confidence'], 0.99)), 'margin': _score_margin(parts['harmonic']), 'method': 'Tier0 consensus'}

def _fallback_candidate_score(parts, candidates):
    """Conservative non-neural fallback used only if the ranker is unavailable."""
    weights = {'yin': 0.42, 'harmonic': 0.28, 'fft': 0.18, 'acf': 0.07, 'cqt': 0.05}
    values = np.zeros(len(candidates), dtype=np.float32)
    for idx, midi in enumerate(candidates):
        values[idx] = sum((weights[key] * float(parts[key][midi]) for key in weights))
    return normalize_scores(values)

def _rank_candidates(parts, candidates, model, duration_seconds):
    matrix = _candidate_feature_matrix(parts, candidates, duration_seconds)
    if model is None:
        scores = _fallback_candidate_score(parts, candidates)
    else:
        probabilities = model.predict_proba(matrix)
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else None
        if positive_index is None:
            scores = _fallback_candidate_score(parts, candidates)
        else:
            scores = probabilities[:, positive_index].astype(np.float32)
            cqt_evidence = np.asarray([parts['cqt'][midi] for midi in candidates], dtype=np.float32)
            if float(np.max(cqt_evidence)) > 1e-10:
                scores = 0.9 * normalize_scores(scores) + 0.1 * normalize_scores(cqt_evidence)
    return scores

def _apply_tier2(scores, candidates, parts):
    if 'attack' not in parts or 'sustain' not in parts or 'yin_multi' not in parts:
        return scores
    detail = np.zeros(len(candidates), dtype=np.float32)
    for idx, midi in enumerate(candidates):
        detail[idx] = 0.18 * float(parts['cqt'][midi]) + 0.18 * float(parts['attack'][midi]) + 0.28 * float(parts['sustain'][midi]) + 0.36 * float(parts['yin_multi'][midi])
    detail = normalize_scores(detail)
    base = normalize_scores(scores)
    return base * 0.82 + detail * 0.18

def analyze_audio(audio, model=None, source_name='<memory>'):
    started = time.perf_counter()
    signal = _prepare_audio(audio)
    duration = len(signal) / SAMPLE_RATE
    t0 = time.perf_counter()
    tier0 = _analyze_tier0(signal)
    tier0_ms = (time.perf_counter() - t0) * 1000.0
    early = _tier0_early_decision(tier0)
    if early is not None:
        midi = early['midi']
        total_ms = (time.perf_counter() - started) * 1000.0
        return {'status': 'OK', 'midi': midi, 'note': midi_name(midi), 'frequency': midi_frequency(midi), 'confidence': early['confidence'], 'margin': early['margin'], 'confidence_kind': 'YIN CMND + exact consensus (heuristic)', 'tier': 0, 'method': early['method'], 'top_candidates': [{'midi': midi, 'note': midi_name(midi), 'score': 1.0}], 'analyzers': {'fft': int(np.argmax(tier0['fft'])), 'harmonic': int(np.argmax(tier0['harmonic'])), 'yin': int(tier0['yin_candidates'][0]['midi']), 'autocorrelation': int(np.argmax(tier0['acf'])) if np.max(tier0['acf']) > 0 else None, 'cqt': None, 'attack': None, 'sustain': None}, 'timings_ms': {'tier0': tier0_ms, 'tier1': 0.0, 'tier2': 0.0, 'total': total_ms}, 'duration': duration, 'source': source_name}
    t1 = time.perf_counter()
    parts = _analyze_tier1(signal, tier0)
    candidates = _generate_candidates(parts)
    if not candidates:
        raise RuntimeError('候補音高を生成できませんでした')
    rank_scores = _rank_candidates(parts, candidates, model, duration)
    ranking = np.argsort(rank_scores)[::-1]
    best_index = int(ranking[0])
    second_index = int(ranking[1]) if len(ranking) > 1 else best_index
    tier1_ms = (time.perf_counter() - t1) * 1000.0
    relative_margin = _score_margin(rank_scores)
    low_conf = relative_margin < LOW_CONF_RELATIVE_MARGIN
    yin_top = int(np.argmax(parts['yin'])) if np.max(parts['yin']) > 0 else None
    rank_top = candidates[best_index]
    if yin_top is not None and abs(yin_top - rank_top) >= 12:
        low_conf = True
    tier2_ms = 0.0
    if low_conf:
        t2 = time.perf_counter()
        parts = _analyze_tier2(signal, parts)
        refined_scores = _apply_tier2(rank_scores, candidates, parts)
        rank_scores = refined_scores
        tier2_ms = (time.perf_counter() - t2) * 1000.0
        ranking = np.argsort(rank_scores)[::-1]
        best_index = int(ranking[0])
        second_index = int(ranking[1]) if len(ranking) > 1 else best_index
        relative_margin = _score_margin(rank_scores)
        yin_multi_top = int(np.argmax(parts['yin_multi'])) if float(np.max(parts['yin_multi'])) > 0 else None
        low_conf = relative_margin < LOW_CONF_RELATIVE_MARGIN
        if yin_multi_top is not None and abs(yin_multi_top - int(candidates[best_index])) >= 12:
            low_conf = True
    chosen = int(candidates[best_index])
    second = int(candidates[second_index])
    top_order = ranking[:5]
    top_candidates = [{'midi': int(candidates[index]), 'note': midi_name(candidates[index]), 'score': float(rank_scores[index])} for index in top_order]
    total_ms = (time.perf_counter() - started) * 1000.0
    return {'status': 'LOW_CONF' if low_conf else 'OK', 'midi': chosen, 'note': midi_name(chosen), 'frequency': midi_frequency(chosen), 'confidence': float(max(0.0, min(1.0, relative_margin))), 'margin': float(relative_margin), 'confidence_kind': 'relative candidate-score margin (not calibrated probability)', 'tier': int(parts['tier']), 'method': 'Candidate reranker' if model is not None else 'DSP fallback', 'top_candidates': top_candidates, 'analyzers': {'fft': int(np.argmax(parts['fft'])) if np.max(parts['fft']) > 0 else None, 'harmonic': int(np.argmax(parts['harmonic'])) if np.max(parts['harmonic']) > 0 else None, 'yin': int(np.argmax(parts['yin'])) if np.max(parts['yin']) > 0 else None, 'autocorrelation': int(np.argmax(parts['acf'])) if np.max(parts['acf']) > 0 else None, 'cqt': int(np.argmax(parts['cqt'])) if np.max(parts['cqt']) > 0 else None, 'attack': int(np.argmax(parts['attack'])) if np.max(parts['attack']) > 0 else None, 'sustain': int(np.argmax(parts['sustain'])) if np.max(parts['sustain']) > 0 else None}, 'timings_ms': {'tier0': tier0_ms, 'tier1': tier1_ms, 'tier2': tier2_ms, 'total': total_ms}, 'duration': duration, 'source': source_name, 'second_candidate_midi': second}

def analyze_file(audio_path, model=None):
    """Load a WAV and return a structured prediction or SKIP result."""
    path = Path(audio_path)
    load_started = time.perf_counter()
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        result = analyze_audio(audio, model=model, source_name=str(path))
        result['timings_ms']['load'] = load_ms
        result['timings_ms']['total_with_load'] = result['timings_ms']['total'] + load_ms
        result['audio_path'] = str(path)
        return result
    except InvalidAudioError as error:
        load_ms = (time.perf_counter() - load_started) * 1000.0
        return {'status': 'SKIP', 'midi': None, 'note': None, 'reason': str(error), 'confidence': 0.0, 'margin': 0.0, 'timings_ms': {'load': load_ms, 'total_with_load': load_ms}, 'audio_path': str(path)}

def _format_analyzer(name, midi):
    if midi is None:
        return f'{name:20s}: N/A'
    return f'{name:20s}: {midi_name(midi):5s} MIDI={int(midi):3d}'

def print_result(result):
    print()
    print('=' * 48)
    print(f' Absolute Pitch AI v{VERSION}')
    print('=' * 48)
    if result['status'] == 'SKIP':
        print('有効な音声ではありません')
        print(f"Reason: {result.get('reason', 'invalid audio')}")
        return
    if result['status'] == 'LOW_CONF':
        print('Result: LOW_CONF')
        print(f"Best candidate: {result['note']}")
    else:
        print(f"Result: {result['note']}")
    print(f"MIDI: {result['midi']}")
    print(f"Frequency: {result['frequency']:.3f} Hz")
    print(f"Status: {result['status']}")
    print(f"Confidence score: {result['confidence']:.4f} ({result['confidence_kind']})")
    print(f"Analysis tier: {result['tier']}")
    print(f"Method: {result['method']}")
    print()
    print('Individual analyzer results:')
    analyzers = result.get('analyzers', {})
    _print_order = (('FFT fundamental', 'fft'), ('Harmonic template', 'harmonic'), ('Autocorrelation', 'autocorrelation'), ('YIN', 'yin'), ('CQT', 'cqt'), ('Attack analysis', 'attack'), ('Sustain analysis', 'sustain'))
    for label, key in _print_order:
        print(_format_analyzer(label, analyzers.get(key)))
    print()
    print('Top candidates:')
    for index, item in enumerate(result.get('top_candidates', []), 1):
        print(f"{index}. {item['note']:5s} MIDI={item['midi']:3d} Score={item['score']:.5f}")
    timings = result.get('timings_ms', {})
    print()
    print(f"Timing: Tier0={timings.get('tier0', 0):.1f}ms Tier1={timings.get('tier1', 0):.1f}ms Tier2={timings.get('tier2', 0):.1f}ms Total={timings.get('total_with_load', timings.get('total', 0)):.1f}ms")

def predict(audio_path, model):
    """Backward-compatible convenience wrapper: analyze and print."""
    result = analyze_file(audio_path, model)
    print_result(result)
    return result
