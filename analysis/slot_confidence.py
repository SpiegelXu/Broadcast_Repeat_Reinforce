import argparse
from pathlib import Path

import pandas as pd


SLOT_NAMES = ("H", "T", "O")


def parse_args():
    parser = argparse.ArgumentParser(
        description="read unmasked_positions.csv, calculate slot confidence。"
    )
    parser.add_argument("--input", required=True, )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def read_one_file(csv_path):
    frame = pd.read_csv(csv_path)
    frame = frame[frame["digit_type"].isin(SLOT_NAMES)].copy()
    sample_columns = ["sample_id"] if "sample_id" in frame.columns else []
    frame = frame.sort_values(sample_columns + ["step", "canvas_pos"])

    frame["slot_reveal_index"] = (
        frame.groupby(sample_columns + ["digit_type"]).cumcount() + 1
    )
    return frame[["digit_type", "slot_reveal_index", "confidence"]]


def analyze(input_path):
    if input_path.is_file():
        csv_paths = [input_path]
    else:
        csv_paths = sorted(input_path.rglob("unmasked_positions.csv"))

    records = pd.concat(
        [read_one_file(csv_path) for csv_path in csv_paths],
        ignore_index=True,
    )

    
    summary = (
        records.groupby(["slot_reveal_index", "digit_type"])["confidence"]
        .mean()
        .unstack("digit_type")
        .reindex(columns=SLOT_NAMES)
        .reset_index()
    )
    summary.columns.name = None
    return summary, len(csv_paths)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    run_dir = input_path.parent if input_path.is_file() else input_path
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "slot_reveal_confidence.csv"
    )

    summary, file_count = analyze(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"Read {file_count} unmasked_positions.csv files")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
