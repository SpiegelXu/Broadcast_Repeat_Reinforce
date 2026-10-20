#!/usr/bin/env python3
"""Recompute forecasting MAE directly from saved sampling metadata.

The script recursively discovers ``metadata.json`` files, deserializes every
saved completion, applies the saved scaler, and compares it with ``target``.
It reports both per-sample MAE and the original mainline-style MAE after
combining samples within each series.

Example
-------
python compute_mae_from_metadata.py \
    --input /data1/xujianheng/llada_mainline_look_ahead/runs/RUN_NAME \
    --output /data1/xujianheng/llada_mainline_look_ahead/outputs/RUN_NAME_mae
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively recompute MAE from sampling metadata.json files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Run root, series directory, sample directory, or metadata.json file.",
    )
    parser.add_argument(
        "--output",
        default="mae_from_metadata",
        help="Output directory for sample_metrics.csv, series_metrics.csv, and summary.json.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("energy_hybrid", "median", "mean"),
        default=None,
        help="Sample aggregation. Default: read config.yaml, otherwise energy_hybrid.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Zero-based energy rank used by energy_hybrid. Default: config.yaml or 2.",
    )
    parser.add_argument("--base", type=int, default=None, help="Serialization base; default 10.")
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help="Serialization precision; default config value or metadata digits_per_value.",
    )
    parser.add_argument(
        "--time-sep",
        default=None,
        help="Time-step separator; default config value or comma.",
    )
    parser.add_argument(
        "--no-half-bin-correction",
        action="store_true",
        help="Disable the +0.5/base**precision correction used by deserialize_str.",
    )
    parser.add_argument(
        "--strict-length",
        action="store_true",
        help="Reject completions shorter than target instead of padding with the last value.",
    )
    return parser.parse_args()


def discover_metadata(paths):
    files = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("metadata.json")
        else:
            raise FileNotFoundError(path)
        # Only sample-level metadata contains target/scaler/completions.  A run
        # root can also contain metadata.json, but it has no series identity.
        files.extend(
            candidate
            for candidate in candidates
            if candidate.parent.name.startswith("samples_")
            and candidate.parent.parent.name.startswith("series_")
        )
    return sorted(set(file.resolve() for file in files))


def find_named_parent(path, prefix):
    """Return the nearest parent whose directory name starts with ``prefix``."""
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent
    return None


def resolve_run_and_sample_ids(metadata_path, metadata):
    """Read IDs from metadata when present, otherwise decode the directory layout."""
    series_dir = find_named_parent(metadata_path, "series_")
    sample_dir = find_named_parent(metadata_path, "samples_")

    if "series_id" in metadata:
        series_id = int(metadata["series_id"])
    elif series_dir is not None:
        series_id = int(series_dir.name.removeprefix("series_"))
    else:
        raise ValueError(f"Cannot infer series_id from {metadata_path}")

    if metadata.get("run_id") is not None:
        run_id = str(metadata["run_id"])
    elif series_dir is not None:
        run_id = series_dir.parent.name
    else:
        run_id = metadata_path.parent.name

    sample_ids = metadata.get("sample_ids") or []
    if sample_ids:
        sample_ids = [int(sample_id) for sample_id in sample_ids]
    elif metadata.get("sample_start") is not None:
        sample_start = int(metadata["sample_start"])
        sample_ids = list(range(sample_start, sample_start + len(metadata.get("completions", []))))
    elif sample_dir is not None:
        match = re.fullmatch(r"samples_(\d+)_(\d+)", sample_dir.name)
        if match is None:
            raise ValueError(f"Cannot parse sample directory {sample_dir}")
        sample_start = int(match.group(1))
        sample_ids = list(range(sample_start, sample_start + len(metadata.get("completions", []))))
    else:
        sample_ids = list(range(len(metadata.get("completions", []))))

    return run_id, series_id, sample_ids


def find_config(input_paths):
    try:
        import yaml
    except ImportError:
        return {}

    candidates = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            path = path.parent
        candidates.append(path / "config.yaml")
        candidates.extend(parent / "config.yaml" for parent in path.parents)

    for config_path in candidates:
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
    return {}


def deserialize_completion(text, base, precision, time_sep, half_bin_correction):
    """Match the numerical convention used by time_series.serialize.deserialize_str."""
    values = []
    for raw_token in text.split(time_sep):
        token = raw_token.strip()
        if not token:
            continue

        sign = -1.0 if token.startswith("-") else 1.0
        if token[:1] in ("+", "-"):
            token = token[1:]

        digits = []
        for char in token:
            if char.isdigit():
                digit = int(char)
                if digit >= base:
                    break
                digits.append(digit)
            elif char == ".":
                continue
            else:
                break
        if not digits:
            break

        value = 0.0
        for digit in digits:
            value = value * base + digit
        value /= base ** precision
        if half_bin_correction:
            value += 0.5 / base ** precision
        values.append(sign * value)

    return np.asarray(values, dtype=float)


def match_target_length(prediction, target_length, strict_length):
    if prediction.size == 0:
        return None, "failed"
    if prediction.size < target_length:
        if strict_length:
            return None, "failed_short"
        prediction = np.concatenate(
            [prediction, np.full(target_length - prediction.size, prediction[-1])]
        )
        return prediction, "padded"
    if prediction.size > target_length:
        return prediction[:target_length], "truncated"
    return prediction, "ok"


def aggregate_samples(samples, method, k):
    samples = np.asarray(samples, dtype=float)
    if samples.shape[0] == 1:
        return samples[0].copy(), "single_sample"
    if method == "median":
        return np.median(samples, axis=0), "ok"
    if method == "mean":
        return samples.mean(axis=0), "ok"
    if k >= samples.shape[0]:
        return None, f"need_at_least_{k + 1}_samples"

    differences = np.diff(samples, axis=1)
    energy = np.sum(differences ** 2, axis=1)
    selected_index = np.argsort(-energy, kind="stable")[k]
    selected = samples[selected_index].copy()

    sorted_indices = np.argsort(np.abs(differences), axis=0, kind="stable")
    median_indices = sorted_indices[samples.shape[0] // 2]
    time_indices = np.arange(differences.shape[1])
    median_signed_difference = differences[median_indices, time_indices]

    result = selected.copy()
    result[1:] += median_signed_difference
    return result, "ok"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    metadata_files = discover_metadata(args.input)
    if not metadata_files:
        raise FileNotFoundError("No metadata.json files found under the supplied input.")

    config = find_config(args.input)
    forecast_config = config.get("forecast", {})
    settings_config = forecast_config.get("settings", {})
    aggregation = args.aggregation or forecast_config.get("aggregation", "energy_hybrid")
    k = args.k if args.k is not None else int(forecast_config.get("K", 2))
    base = args.base if args.base is not None else int(settings_config.get("base", 10))
    time_sep = args.time_sep or settings_config.get("time_sep", ",")
    half_bin = not args.no_half_bin_correction

    sample_rows = []
    grouped_samples = defaultdict(list)
    series_info = {}

    for metadata_path in metadata_files:
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

        run_id, series_id, sample_ids = resolve_run_and_sample_ids(metadata_path, metadata)
        target = np.asarray(metadata["target"], dtype=float)
        scaler = metadata["scaler"]
        scaler_min = float(scaler["min_"])
        scaler_q = float(scaler["q"])
        completions = metadata.get("completions", [])
        metadata_precision = metadata.get("sampler_config", {}).get("digits_per_value", 3)
        precision = args.precision if args.precision is not None else int(
            settings_config.get("prec", metadata_precision)
        )

        series_key = (run_id, series_id)
        series_info[series_key] = {
            "target": target,
            "scaler_min": scaler_min,
            "scaler_q": scaler_q,
        }

        for completion_index, completion in enumerate(completions):
            sample_id = sample_ids[completion_index]
            normalized = deserialize_completion(
                completion, base, precision, time_sep, half_bin
            )
            raw_length = int(normalized.size)
            normalized, status = match_target_length(
                normalized, len(target), args.strict_length
            )

            row = {
                "run_id": run_id,
                "series_id": series_id,
                "sample_id": sample_id,
                "metadata_file": str(metadata_path),
                "raw_length": raw_length,
                "target_length": len(target),
                "status": status,
                "sample_mae": "",
            }
            if normalized is not None:
                prediction = normalized * scaler_q + scaler_min
                sample_mae = float(np.mean(np.abs(prediction - target)))
                row["sample_mae"] = sample_mae
                grouped_samples[series_key].append((sample_id, prediction))
            sample_rows.append(row)

    series_rows = []
    series_maes = []
    for series_key in sorted(series_info):
        run_id, series_id = series_key
        info = series_info[series_key]
        target = info["target"]
        samples = sorted(grouped_samples.get(series_key, []), key=lambda item: item[0])
        unique_samples = {}
        for sample_id, prediction in samples:
            unique_samples[sample_id] = prediction
        predictions = list(unique_samples.values())

        combined, status = aggregate_samples(predictions, aggregation, k) if predictions else (None, "no_valid_samples")
        series_mae = ""
        if combined is not None:
            series_mae = float(np.mean(np.abs(combined - target)))
            series_maes.append(series_mae)

        series_rows.append({
            "run_id": run_id,
            "series_id": series_id,
            "target_length": len(target),
            "valid_samples": len(predictions),
            "aggregation": aggregation,
            "K": k,
            "status": status,
            "series_mae": series_mae,
        })

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "sample_metrics.csv",
        sample_rows,
        (
            "run_id", "series_id", "sample_id", "metadata_file", "raw_length",
            "target_length", "status", "sample_mae",
        ),
    )
    write_csv(
        output_dir / "series_metrics.csv",
        series_rows,
        (
            "run_id", "series_id", "target_length", "valid_samples",
            "aggregation", "K", "status", "series_mae",
        ),
    )

    sample_maes = [float(row["sample_mae"]) for row in sample_rows if row["sample_mae"] != ""]
    summary = {
        "input": [str(Path(path).expanduser().resolve()) for path in args.input],
        "metadata_files": len(metadata_files),
        "series_total": len(series_rows),
        "series_with_aggregated_mae": len(series_maes),
        "valid_samples": len(sample_maes),
        "aggregation": aggregation,
        "K": k,
        "base": base,
        "precision_override": args.precision,
        "time_sep": time_sep,
        "half_bin_correction": half_bin,
        "strict_length": args.strict_length,
        "mae": float(np.mean(series_maes)) if series_maes else None,
        "sample_mae_mean": float(np.mean(sample_maes)) if sample_maes else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"metadata files: {summary['metadata_files']}")
    print(f"valid samples: {summary['valid_samples']}")
    print(f"series with MAE: {summary['series_with_aggregated_mae']}/{summary['series_total']}")
    print(f"aggregated MAE: {summary['mae']}")
    print(f"mean sample MAE: {summary['sample_mae_mean']}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
