import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from time_series.repetition import compute_r_value_from_tokens, compute_slot_repetition
from time_series.serialize import SerializerSettings, serialize_arr


SLOT_NAMES = ("H", "T", "O")


def parse_args():
    parser = argparse.ArgumentParser(
        description="read samples.csv calculate final sample repetition and ground truth repetition。"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", default=None
    )
    return parser.parse_args()


def discrete_tokens(values, scaler, settings):
    values = np.asarray(values, dtype=float)
    normalized = (values - scaler["min_"]) / scaler["q"]
    token_string = serialize_arr(normalized, settings)
    return [token.strip() for token in token_string.split(settings.time_sep) if token.strip()]


def repetition_rates(values, scaler, settings):
    tokens = discrete_tokens(values, scaler, settings)
    return {
        "all": compute_r_value_from_tokens(tokens),
        **compute_slot_repetition(tokens),
    }


def analyze(run_dir):
    samples = pd.read_csv(run_dir / "samples.csv")

    with (run_dir / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (run_dir / "data.json").open(encoding="utf-8") as file:
        data = json.load(file)

    settings = SerializerSettings(**config["forecast"]["settings"])
    scaler_by_series = {
        int(series_id): scaler
        for series_id, scaler in zip(data["series_ids"], data["scalers"])
    }

    prediction_rates = {name: [] for name in ("all", *SLOT_NAMES)}
    target_rates = {name: [] for name in ("all", *SLOT_NAMES)}

    for series_id, series_frame in samples.groupby("series_id", sort=False):
        series_frame = series_frame.sort_values("time_index")
        scaler = scaler_by_series[int(series_id)]

        target = (
            series_frame.drop_duplicates("time_index")
            .sort_values("time_index")["target"]
            .to_numpy(dtype=float)
        )
        rates = repetition_rates(target, scaler, settings)
        for name in rates:
            target_rates[name].append(rates[name])

        for _, sample_frame in series_frame.groupby("sample_id", sort=False):
            prediction = sample_frame.sort_values("time_index")["prediction"]
            if prediction.isna().any():
                continue
            rates = repetition_rates(prediction.to_numpy(dtype=float), scaler, settings)
            for name in rates:
                prediction_rates[name].append(rates[name])

    rows = []
    for name in ("all", *SLOT_NAMES):
        r_pred = float(np.mean(prediction_rates[name]))
        r_gt = float(np.mean(target_rates[name]))
        rows.append({
            "metric": "R" if name == "all" else f"R_{name}",
            "prediction_repetition": r_pred,
            "ground_truth_repetition": r_gt,
            "repetition_excess": r_pred - r_gt,
            "prediction_sample_count": len(prediction_rates[name]),
            "ground_truth_series_count": len(target_rates[name]),
        })

    slot_pred = float(np.mean([rows[index]["prediction_repetition"] for index in range(1, 4)]))
    slot_gt = float(np.mean([rows[index]["ground_truth_repetition"] for index in range(1, 4)]))
    rows.append({
        "metric": "R_slot_mean",
        "prediction_repetition": slot_pred,
        "ground_truth_repetition": slot_gt,
        "repetition_excess": slot_pred - slot_gt,
        "prediction_sample_count": len(prediction_rates["H"]),
        "ground_truth_series_count": len(target_rates["H"]),
    })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    run_dir = input_path.parent if input_path.is_file() else input_path
    output_path = (Path(args.output).expanduser().resolve()
                   if args.output else run_dir / "final_repetition.csv")

    summary = analyze(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
