from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import io
import re
import sys


# ==========================================
# Absolute Pitch AI - Multi-Instrument Benchmark
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
WAV_DIR = Path(r"E:\rise-\temp\midi音\wav")
OUTPUT_TXT = PROJECT_DIR / "benchmark_results.txt"
MAIN_SCRIPT = PROJECT_DIR / "main.py"

FILENAME_PATTERN = re.compile(
    r"^(\d+)\s*-\s*(\d+)\s*-\s*(.+)\.wav$",
    re.IGNORECASE,
)
RESULT_PATTERN = re.compile(
    r"^\s*Result:\s*(\S+)\s*$",
    re.MULTILINE,
)


class Tee:
    """Write the same text to multiple streams and flush immediately."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def collect_targets():
    """Collect WAV files and group them by instrument number."""
    instruments = defaultdict(list)

    for wav_path in WAV_DIR.glob("*.wav"):
        match = FILENAME_PATTERN.match(wav_path.name)
        if not match:
            continue

        instrument_id, notes_number, expected_note = match.groups()
        instruments[instrument_id].append(
            (wav_path, int(notes_number), expected_note.strip())
        )

    for files in instruments.values():
        files.sort(key=lambda item: (item[1], item[0].name.lower()))

    return dict(
        sorted(
            instruments.items(),
            key=lambda item: (int(item[0]), item[0]),
        )
    )


def run_benchmark(pitch_ai, model, instruments):
    total_files = sum(len(files) for files in instruments.values())
    total_correct = 0
    total_tested = 0

    print("=" * 72)
    print(" Absolute Pitch AI - Multi-Instrument Benchmark")
    print("=" * 72)
    print(f"WAV directory: {WAV_DIR}")
    print(f"Instruments:   {len(instruments)}")
    print(f"Test files:    {total_files}")
    print("Output:        ./benchmark_results.txt")
    print("=" * 72)

    for instrument_id, files in instruments.items():
        instrument_correct = 0
        instrument_tested = 0

        print()
        print("-" * 72)
        print(f"[Instrument {instrument_id}]")
        print(f"Files: {len(files)}")
        print("-" * 72)

        for index, (wav_path, notes_number, expected_note) in enumerate(
            files, start=1
        ):
            print()
            print(f"[{index}/{len(files)}] {wav_path.name}", flush=True)

            captured_output = io.StringIO()
            predicted_note = "ERROR"

            try:
                # main.py's own detailed output is mirrored to the console,
                # the TXT log, and this temporary buffer for result parsing.
                with redirect_stdout(Tee(sys.stdout, captured_output)):
                    pitch_ai.predict(str(wav_path), model)

                match = RESULT_PATTERN.search(captured_output.getvalue())
                if match:
                    predicted_note = match.group(1)
                else:
                    predicted_note = "NO_RESULT"
                    print("  [ERROR] Could not find 'Result:' in prediction output.")

            except Exception as error:
                print(f"  [ERROR] {type(error).__name__}: {error}")

            correct = predicted_note == expected_note
            status = "OK" if correct else "NG"

            print(
                f"  Expected: {expected_note} | "
                f"Predicted: {predicted_note} | {status}"
            )

            instrument_tested += 1
            total_tested += 1

            if correct:
                instrument_correct += 1
                total_correct += 1

        accuracy = (
            instrument_correct / instrument_tested * 100
            if instrument_tested
            else 0.0
        )

        print()
        print(f"[Instrument {instrument_id} Summary]")
        print(f"  Total:      {instrument_tested}")
        print(f"  Correct:    {instrument_correct}")
        print(f"  Incorrect:  {instrument_tested - instrument_correct}")
        print(f"  Accuracy:   {accuracy:.2f}%")

    overall_accuracy = (
        total_correct / total_tested * 100 if total_tested else 0.0
    )

    print()
    print("=" * 72)
    print(" BENCHMARK COMPLETE")
    print("=" * 72)
    print(f"Instruments: {len(instruments)}")
    print(f"Total:       {total_tested}")
    print(f"Correct:     {total_correct}")
    print(f"Incorrect:   {total_tested - total_correct}")
    print(f"Accuracy:    {overall_accuracy:.2f}%")
    print(f"TXT:         {OUTPUT_TXT}")
    print("=" * 72)


def main():
    # "w" starts a fresh log for each benchmark run.
    # Tee mirrors every print to the console and the TXT file.
    with OUTPUT_TXT.open("w", encoding="utf-8", buffering=1) as log_file:
        output_stream = Tee(sys.stdout, log_file)

        with redirect_stdout(output_stream):
            try:
                print("[Benchmark] Starting...")
                print(f"[Benchmark] Log file: {OUTPUT_TXT}")
                print()

                if not MAIN_SCRIPT.is_file():
                    print(f"[ERROR] main.py not found: {MAIN_SCRIPT}")
                    return

                if not WAV_DIR.is_dir():
                    print(f"[ERROR] WAV directory not found: {WAV_DIR}")
                    return

                instruments = collect_targets()

                if not instruments:
                    print("[ERROR] No matching WAV files were found.")
                    print(
                        "Expected filename format: "
                        "(instrument) - (notes number) - (note).wav"
                    )
                    return

                # Import and load the AI once, then reuse the same model
                # for every WAV file. This avoids a Python/model startup
                # for each individual audio file.
                project_path = str(PROJECT_DIR)
                if project_path not in sys.path:
                    sys.path.insert(0, project_path)

                import main as pitch_ai

                # main.py normally uses a path relative to the current
                # working directory. Anchor it to the project directory.
                pitch_ai.MODEL_PATH = PROJECT_DIR / "models" / "pitch_ai.pkl"

                print("[Benchmark] Loading AI model once...")
                model = pitch_ai.load_model()
                print("[Benchmark] Model ready. Starting tests...")

                run_benchmark(pitch_ai, model, instruments)

            except KeyboardInterrupt:
                print()
                print("[CANCELLED] Benchmark interrupted by user.")
                print("[CANCELLED] Results up to this point are saved in the TXT file.")


if __name__ == "__main__":
    main()
