from pathlib import Path
import re
import csv
import subprocess
import sys

# Absolute Pitch AI benchmark for instrument 000.
WAV_DIR = Path(r"E:\rise-\temp\midi音\wav")
INSTRUMENT_ID = "000"
OUTPUT_CSV = Path("results_000.csv")
MAIN_SCRIPT = Path(__file__).with_name("main.py")

FILENAME_PATTERN = re.compile(
    r"^(\d+)\s*-\s*(\d+)\s*-\s*(.+)\.wav$",
    re.IGNORECASE,
)
RESULT_PATTERN = re.compile(r"^\s*Result:\s*(\S+)\s*$", re.MULTILINE)


def collect_targets():
    targets = []

    for wav_path in sorted(WAV_DIR.glob("*.wav")):
        match = FILENAME_PATTERN.match(wav_path.name)
        if not match:
            continue

        instrument_id, notes_number, expected_note = match.groups()
        if instrument_id != INSTRUMENT_ID:
            continue

        targets.append(
            (wav_path, int(notes_number), expected_note.strip())
        )

    return targets


def predict(wav_path):
    process = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT), str(wav_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    output = process.stdout + "\n" + process.stderr
    match = RESULT_PATTERN.search(output)

    if process.returncode != 0:
        error = output.strip().splitlines()
        return "ERROR", error[-1] if error else f"Exit code {process.returncode}"

    if not match:
        return "NO_RESULT", "Could not find 'Result:' in program output"

    return match.group(1), ""


def main():
    if not MAIN_SCRIPT.is_file():
        print(f"[ERROR] main.py not found: {MAIN_SCRIPT}")
        raise SystemExit(1)

    if not WAV_DIR.is_dir():
        print(f"[ERROR] WAV directory not found: {WAV_DIR}")
        raise SystemExit(1)

    targets = collect_targets()

    print("=" * 60)
    print(" Absolute Pitch AI - Benchmark")
    print("=" * 60)
    print(f"Instrument: {INSTRUMENT_ID}")
    print(f"Directory:  {WAV_DIR}")
    print(f"Test files: {len(targets)}")
    print("=" * 60)

    if not targets:
        print("[ERROR] No matching WAV files were found.")
        print("Expected filename format: 000 - 001 - C#-1.wav")
        raise SystemExit(1)

    results = []

    for index, (wav_path, notes_number, expected_note) in enumerate(targets, 1):
        print(f"[{index}/{len(targets)}] {wav_path.name}", flush=True)

        predicted_note, error = predict(wav_path)
        correct = predicted_note == expected_note

        status = "OK" if correct else "NG"
        print(
            f"  Expected: {expected_note} | "
            f"Predicted: {predicted_note} | {status}"
        )

        results.append(
            {
                "instrument": INSTRUMENT_ID,
                "notes_number": notes_number,
                "file": wav_path.name,
                "expected": expected_note,
                "predicted": predicted_note,
                "correct": correct,
                "error": error,
            }
        )

    total = len(results)
    correct_count = sum(row["correct"] for row in results)
    accuracy = correct_count / total * 100 if total else 0.0

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "instrument",
            "notes_number",
            "file",
            "expected",
            "predicted",
            "correct",
            "error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(" BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Instrument: {INSTRUMENT_ID}")
    print(f"Total:      {total}")
    print(f"Correct:    {correct_count}")
    print(f"Incorrect:  {total - correct_count}")
    print(f"Accuracy:   {accuracy:.2f}%")
    print(f"CSV:        {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
