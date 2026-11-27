import argparse
import sys

import pitch_dsp_v040 as pitch_dsp
import pitch_model_v040 as pitch_model
import pitch_runtime_v040 as pitch_runtime

VERSION = pitch_dsp.VERSION
SAMPLE_RATE = pitch_dsp.SAMPLE_RATE
MODEL_PATH = pitch_model.MODEL_PATH


def load_model():
    """Load or train v0.4's separate candidate-reranker model."""
    pitch_model.MODEL_PATH = MODEL_PATH
    return pitch_model.load_model()


def analyze_file(audio_path, model=None):
    return pitch_runtime.analyze_file(audio_path, model)


def analyze_audio(audio, model=None, source_name="<memory>"):
    return pitch_runtime.analyze_audio(audio, model, source_name)


def print_result(result):
    return pitch_runtime.print_result(result)


def predict(audio_path, model):
    """Keep the original benchmark.py calling convention."""
    return pitch_runtime.predict(audio_path, model)


def main():
    parser = argparse.ArgumentParser(
        description="Absolute Pitch AI v0.4 - Adaptive DSP + ExtraTrees reranker (no deep learning)"
    )
    parser.add_argument("audio", nargs="?", help="Audio file to analyze")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Rebuild the v0.4 candidate reranker",
    )
    args = parser.parse_args()

    if args.retrain and MODEL_PATH.exists():
        MODEL_PATH.unlink()

    model = load_model()
    if args.audio:
        result = analyze_file(args.audio, model)
        print_result(result)
        return 0

    print()
    print("AI is ready!")
    print(f"Version: {VERSION}")
    print("Usage:")
    print('python main.py "test.wav"')
    print("Rebuild model:")
    print("python main.py --retrain")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
    except Exception as error:
        print("\n[ERROR]", error)
        raise SystemExit(1)
