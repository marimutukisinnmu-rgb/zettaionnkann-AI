from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WAV_DIR = Path(r"E:\rise-\temp\midi音\wav")
RUNS_DIR = PROJECT_DIR / "bench_runs"
NOTE_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10}
NOTE_RE = re.compile(r"^(C#|D#|F#|G#|A#|C|D|E|F|G|A|B)(-?\d+)$")


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_expected_midi(name):
    match = NOTE_RE.fullmatch(name.strip())
    if not match:
        return None
    pitch, octave_text = match.groups()
    midi = (int(octave_text) + 1) * 12 + NOTE_PC[pitch]
    return midi if 0 <= midi <= 127 else None


def range_name(midi):
    if midi is None:
        return "unknown"
    if 1 <= midi <= 23:
        return "MIDI_1_23"
    if 24 <= midi <= 95:
        return "MIDI_24_95"
    if 96 <= midi <= 127:
        return "MIDI_96_127"
    return "MIDI_0_not_in_standard_test" if midi == 0 else "unknown"


def classify_error(expected, predicted):
    delta = predicted - expected
    distance = abs(delta)
    if distance == 0:
        return "correct"
    if distance % 12 == 0:
        return f"octave_{'up' if delta > 0 else 'down'}_{distance // 12}"
    if distance == 1:
        return f"semitone_{'up' if delta > 0 else 'down'}"
    harmonic_steps = {round(12 * math.log2(n)) for n in range(2, 17) if n & (n - 1)}
    if distance in harmonic_steps:
        return f"harmonic_lock_{'up' if delta > 0 else 'down'}_{distance}"
    return "other"


def collect_targets(wav_dir):
    instruments = defaultdict(list)
    for path in wav_dir.glob("*.wav"):
        parts = path.stem.split(" - ")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        instruments[parts[0]].append((path, int(parts[1]), parts[2].strip()))
    return {
        key: sorted(values, key=lambda row: (row[1], row[0].name.lower()))
        for key, values in sorted(instruments.items(), key=lambda row: int(row[0]))
    }


def make_run_dir(base):
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / f"v0.4.0_{stamp}"
    suffix = 1
    while path.exists():
        path = base / f"v0.4.0_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir()
    return path


def time_stats(values):
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def bucket_summary(total, bucket):
    skipped = bucket.get("SKIP", 0)
    denominator = max(0, total - skipped)
    correct = bucket.get("correct", 0)
    return {
        "total": total,
        **dict(bucket),
        "valid_denominator": denominator,
        "accuracy_all_percent": correct / total * 100 if total else 0.0,
        "accuracy_valid_percent": correct / denominator * 100 if denominator else 0.0,
        "coverage_percent": denominator / total * 100 if total else 0.0,
    }


def run_benchmark(pitch_ai, model, instruments, wav_dir, run_dir):
    total_files = sum(len(entries) for entries in instruments.values())
    counts = Counter()
    instrument_counts = defaultdict(Counter)
    range_counts = defaultdict(Counter)
    errors = Counter()
    skip_reasons = Counter()
    timings = defaultdict(list)
    jsonl_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "input_manifest.csv"
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    file_index = 0

    print("=" * 72)
    print(" Absolute Pitch AI v0.4 - Adaptive DSP Benchmark")
    print("=" * 72)
    print(f"Version:       {pitch_ai.VERSION}")
    print(f"WAV directory: {wav_dir}")
    print(f"Instruments:   {len(instruments)}")
    print(f"Test files:    {total_files}")
    print(f"Run directory: {run_dir}")
    print("Statuses: OK / NG / LOW_CONF / SKIP / ERROR")
    print("=" * 72)

    with jsonl_path.open("w", encoding="utf-8", buffering=1) as jsonl, manifest_path.open("w", encoding="utf-8-sig", newline="") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(["instrument", "note_number", "filename", "expected_note", "expected_midi", "size_bytes", "mtime_ns"])
        for instrument, entries in instruments.items():
            print(f"\n[Instrument {instrument}]")
            print(f"Files: {len(entries)}")
            bucket = instrument_counts[instrument]
            for index, (path, note_number, expected_name) in enumerate(entries, 1):
                file_index += 1
                expected = parse_expected_midi(expected_name)
                started = time.perf_counter()
                result = None
                error_message = None
                try:
                    result = pitch_ai.analyze_file(path, model)
                    status = result.get("status", "ERROR")
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    status = "ERROR"
                    error_message = f"{type(exc).__name__}: {exc}"

                wall_ms = (time.perf_counter() - started) * 1000
                predicted = result.get("midi") if result else None
                reason = result.get("reason") if result else error_message
                correct = False
                signed_error = None
                error_class = None
                range_key = range_name(expected)

                if status == "SKIP":
                    counts["SKIP"] += 1
                    bucket["SKIP"] += 1
                    range_counts[range_key]["SKIP"] += 1
                    skip_reasons[str(reason or "unspecified")] += 1
                elif status == "ERROR" or expected is None or predicted is None:
                    status = "ERROR"
                    counts["ERROR"] += 1
                    bucket["ERROR"] += 1
                    range_counts[range_key]["ERROR"] += 1
                    if expected is None:
                        reason = reason or "unparseable expected note label"
                else:
                    correct = int(predicted) == expected
                    signed_error = int(predicted) - expected
                    error_class = classify_error(expected, int(predicted))
                    counts["valid"] += 1
                    bucket["valid"] += 1
                    range_counts[range_key]["valid"] += 1
                    if correct:
                        counts["correct"] += 1
                        bucket["correct"] += 1
                        range_counts[range_key]["correct"] += 1
                    else:
                        errors[error_class] += 1
                        bucket[error_class] += 1
                        range_counts[range_key][error_class] += 1
                    if status == "LOW_CONF":
                        counts["LOW_CONF"] += 1
                        bucket["LOW_CONF"] += 1
                        range_counts[range_key]["LOW_CONF"] += 1
                    elif correct:
                        status = "OK"
                        counts["OK"] += 1
                        bucket["OK"] += 1
                        range_counts[range_key]["OK"] += 1
                    else:
                        status = "NG"
                        counts["NG"] += 1
                        bucket["NG"] += 1
                        range_counts[range_key]["NG"] += 1

                timing_data = result.get("timings_ms", {}) if result else {}
                if file_index > 3 and status != "SKIP":
                    timings["total"].append(float(timing_data.get("total_with_load", wall_ms)))
                    for key in ("load", "tier0", "tier1", "tier2"):
                        if key in timing_data:
                            timings[key].append(float(timing_data[key]))

                try:
                    stat = path.stat()
                    size_bytes, mtime_ns = stat.st_size, stat.st_mtime_ns
                except OSError:
                    size_bytes, mtime_ns = None, None
                writer.writerow([instrument, note_number, path.name, expected_name, expected, size_bytes, mtime_ns])

                row = {
                    "instrument": instrument,
                    "note_number": note_number,
                    "filename": path.name,
                    "expected_note": expected_name,
                    "expected_midi": expected,
                    "predicted_note": result.get("note") if result else None,
                    "predicted_midi": predicted,
                    "status": status,
                    "is_correct": bool(correct),
                    "signed_error": signed_error,
                    "error_class": error_class,
                    "confidence_relative_margin": result.get("confidence") if result else None,
                    "margin": result.get("margin") if result else None,
                    "tier": result.get("tier") if result else None,
                    "analyzers": result.get("analyzers") if result else None,
                    "top_candidates": result.get("top_candidates") if result else None,
                    "reason": reason,
                    "duration_seconds": result.get("duration") if result else None,
                    "timings_ms": timing_data,
                    "wall_ms": wall_ms,
                    "range": range_key,
                }
                jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(f"\n[{index}/{len(entries)}] {path.name}")
                if result and status in ("OK", "LOW_CONF"):
                    pitch_ai.print_result(result)
                    label = result.get("note", "?")
                elif status == "SKIP":
                    print("有効な音声ではありません")
                    print(f"Reason: {reason}")
                    label = "-"
                else:
                    print(f"[ERROR] {reason}")
                    label = "ERROR"
                print(f"  Expected: {expected_name} | Predicted: {label} | {status}", flush=True)

            summary = bucket_summary(len(entries), bucket)
            print(f"\n[Instrument {instrument} Summary]")
            for key in ("OK", "NG", "LOW_CONF", "SKIP", "ERROR"):
                print(f"  {key:12s}: {bucket.get(key, 0)}")
            print(f"  Accuracy all:   {summary['accuracy_all_percent']:.2f}%")
            print(f"  Accuracy valid: {summary['accuracy_valid_percent']:.2f}%")

    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    accuracy_all = counts["correct"] / total_files * 100 if total_files else 0.0
    valid_denominator = total_files - counts["SKIP"]
    accuracy_valid = counts["correct"] / valid_denominator * 100 if valid_denominator else 0.0
    coverage = valid_denominator / total_files * 100 if total_files else 0.0

    range_summary = {}
    for key, bucket in range_counts.items():
        total = sum(bucket.get(name, 0) for name in ("OK", "NG", "LOW_CONF", "SKIP", "ERROR"))
        range_summary[key] = bucket_summary(total, bucket)
    instrument_summary = {
        key: bucket_summary(len(instruments[key]), bucket)
        for key, bucket in instrument_counts.items()
    }

    def package_ver(name):
        try:
            return package_version(name)
        except PackageNotFoundError:
            return None

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": {name: package_ver(name) for name in ("numpy", "librosa", "scikit-learn", "joblib", "soundfile")},
    }
    summary = {
        "version": "0.4.0",
        "environment": environment,
        "created_at": datetime.now().astimezone().isoformat(),
        "wav_directory": str(wav_dir),
        "run_directory": str(run_dir),
        "counts": dict(counts),
        "total_files": total_files,
        "accuracy_all_percent": accuracy_all,
        "accuracy_valid_percent": accuracy_valid,
        "coverage_percent": coverage,
        "range_summaries": range_summary,
        "instrument_summaries": instrument_summary,
        "error_types": dict(errors),
        "skip_reasons": dict(skip_reasons),
        "timings": {key: time_stats(values) for key, values in timings.items()},
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "notes": [
            "Accuracy(valid) includes every non-SKIP file; ERROR counts as a failed attempt.",
            "LOW_CONF is scored by its best candidate and counted separately.",
            "The first three files are excluded from per-file timing summaries as warm-up.",
            "The manifest records file size and modification time, not a content hash.",
            "The benchmark uses rendered MIDI WAV files and does not establish performance on acoustic recordings.",
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(" BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"Instruments:       {len(instruments)}")
    print(f"Total files:       {total_files}")
    for key in ("OK", "NG", "LOW_CONF", "SKIP", "ERROR"):
        print(f"{key:18s}: {counts.get(key, 0)}")
    print(f"Correct guesses:   {counts['correct']}")
    print(f"Accuracy (all):    {accuracy_all:.2f}%")
    print(f"Accuracy (valid):  {accuracy_valid:.2f}%")
    print(f"Coverage:          {coverage:.2f}%")
    print(f"Wall time:         {wall_seconds:.2f} seconds")
    print(f"CPU time:          {cpu_seconds:.2f} seconds")
    for key, values in timings.items():
        stats = time_stats(values)
        print(f"Time {key:10s}: mean={stats['mean_ms']:.2f}ms median={stats['median_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms")
    print(f"TXT:               {run_dir / 'benchmark_results.txt'}")
    print(f"JSONL:             {jsonl_path}")
    print(f"SUMMARY:           {run_dir / 'summary.json'}")
    print("=" * 72)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Absolute Pitch AI v0.4 benchmark")
    parser.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    if not args.wav_dir.is_dir():
        print(f"[ERROR] WAV directory not found: {args.wav_dir}")
        return 2
    instruments = collect_targets(args.wav_dir)
    if not instruments:
        print("[ERROR] No matching WAV files were found.")
        print("Expected filename format: (instrument) - (note_number) - (note_name).wav")
        return 2
    run_dir = make_run_dir(args.runs_dir)
    log_path = run_dir / "benchmark_results.txt"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        output = Tee(sys.stdout, log_file)
        with redirect_stdout(output):
            print("[Benchmark] Starting v0.4...")
            print(f"[Benchmark] Run folder: {run_dir}")
            project_path = str(PROJECT_DIR)
            if project_path not in sys.path:
                sys.path.insert(0, project_path)
            import main as pitch_ai
            pitch_ai.MODEL_PATH = PROJECT_DIR / "models" / "pitch_reranker_v040.pkl"
            print("[Benchmark] Loading v0.4 candidate reranker...")
            model = pitch_ai.load_model()
            print("[Benchmark] Model ready. Starting tests...")
            run_benchmark(pitch_ai, model, instruments, args.wav_dir, run_dir)
    return 0


def make_run_dir(base):
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base / f"v0.4.0_{stamp}"
    suffix = 1
    while folder.exists():
        folder = base / f"v0.4.0_{stamp}_{suffix:02d}"
        suffix += 1
    folder.mkdir()
    return folder


def time_stats(values):
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {"count": len(values), "mean_ms": float(statistics.mean(values)), "median_ms": float(statistics.median(values)), "p95_ms": float(np.percentile(values, 95))}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[CANCELLED] Benchmark interrupted. The run folder preserves completed records.")
        raise SystemExit(130)
