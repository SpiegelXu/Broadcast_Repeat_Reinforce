"""Aggregate hidden-state distance by Transformer layer.

For every layer and group, the script first averages over rollouts separately
at each diffusion step, then averages those step-level means across steps:

    m[t, l, g] = mean_rollout distance[t, l, g]
    d[l, g]    = mean_step m[t, l, g]

Thus ``hidden_index`` is the analysis axis and ``step`` is merged as repeated
evidence. The default groups include overall convergence (``all``) and the
three numerical slots H/T/U.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GROUPS = ("all", "slot_H", "slot_T", "slot_U")
GROUP_MAP = {
    "slot_0": "slot_H",
    "slot_1": "slot_T",
    "slot_2": "slot_U",
    "slot_3": "comma",
    "slot_O": "slot_U",
    "H": "slot_H",
    "T": "slot_T",
    "O": "slot_U",
    "U": "slot_U",
}
ROLLOUT_COLUMNS = ["run_id", "series_id", "sample_id", "batch_id", "branch"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average hidden-state distances over diffusion steps and report "
            "overall/H/T/U convergence as a function of Transformer layer."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory, one hidden_distance.csv file, or a directory tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RUN_DIR/hidden_convergence_analysis).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help=(
            "Groups after normalization (default: all slot_H slot_T slot_U). "
            "Use slot_3 or comma to include comma positions."
        ),
    )
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=0.25,
        help="Deepest fraction of layers used for the late-layer slope.",
    )
    return parser.parse_args()


def find_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("hidden_distance.csv"))


def path_ids(path: Path) -> tuple[float, float]:
    series_match = re.search(r"series_(\d+)", str(path))
    sample_match = re.search(r"samples_(\d+)_(\d+)", str(path))
    series_id = int(series_match.group(1)) if series_match else np.nan
    sample_id = int(sample_match.group(1)) if sample_match else np.nan
    return series_id, sample_id


def normalize_file(path: Path, requested_groups: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "hidden_index" not in frame.columns and "layer" in frame.columns:
        frame = frame.rename(columns={"layer": "hidden_index"})
    if "distance" not in frame.columns and "cosine_distance" in frame.columns:
        frame = frame.rename(columns={"cosine_distance": "distance"})

    required = {"step", "hidden_index", "group", "distance"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Available columns: {list(frame.columns)}"
        )

    frame["group"] = frame["group"].astype(str).str.strip().replace(GROUP_MAP)
    frame = frame[frame["group"].isin(requested_groups)].copy()
    for column in ("step", "hidden_index", "distance"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["step", "hidden_index", "group", "distance"])

    series_from_path, sample_from_path = path_ids(path)
    defaults = {
        "run_id": "unknown",
        "series_id": series_from_path,
        "sample_id": sample_from_path,
        "batch_id": 0,
        "branch": "full",
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value

    return frame[
        ROLLOUT_COLUMNS + ["step", "hidden_index", "group", "distance"]
    ]


def load_all(path: Path, requested_groups: list[str]) -> tuple[pd.DataFrame, int]:
    files = find_files(path)
    if not files:
        raise FileNotFoundError(f"No hidden_distance.csv found under {path}")
    frames = [normalize_file(file, requested_groups) for file in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"None of the requested groups {requested_groups} were found.")
    return pd.concat(frames, ignore_index=True), len(files)


def summarize(values: pd.Series, prefix: str = "") -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    names = {
        "count": "count",
        "mean": "mean",
        "median": "median",
        "q25": "q25",
        "q75": "q75",
        "std": "std",
    }
    if prefix:
        names = {key: f"{prefix}_{value}" for key, value in names.items()}
    if values.empty:
        return {name: np.nan for name in names.values()}
    return {
        names["count"]: int(len(values)),
        names["mean"]: float(values.mean()),
        names["median"]: float(values.median()),
        names["q25"]: float(values.quantile(0.25)),
        names["q75"]: float(values.quantile(0.75)),
        names["std"]: float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def collapse_duplicate_rows(data: pd.DataFrame) -> pd.DataFrame:
    keys = ROLLOUT_COLUMNS + ["step", "group", "hidden_index"]
    return (
        data.groupby(keys, dropna=False, sort=False)["distance"]
        .mean()
        .reset_index()
    )


def build_step_layer_mean(observations: pd.DataFrame) -> pd.DataFrame:
    """Mean across rollouts for every step, layer, and group."""
    rows = []
    for identity, frame in observations.groupby(
        ["branch", "step", "group", "hidden_index"],
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "branch": identity[0],
                "step": identity[1],
                "group": identity[2],
                "hidden_index": identity[3],
                **summarize(frame["distance"], prefix="rollout"),
            }
        )
    return pd.DataFrame(rows)


def build_layer_mean(step_layer: pd.DataFrame) -> pd.DataFrame:
    """Mean the step-level means so each diffusion step receives equal weight."""
    rows = []
    for identity, frame in step_layer.groupby(
        ["branch", "group", "hidden_index"], dropna=False, sort=True
    ):
        rows.append(
            {
                "branch": identity[0],
                "group": identity[1],
                "hidden_index": identity[2],
                **summarize(frame["rollout_mean"], prefix="step"),
                "mean_rollout_count_per_step": float(
                    frame["rollout_count"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_wide_layer_table(layer_mean: pd.DataFrame) -> pd.DataFrame:
    wide = layer_mean.pivot_table(
        index=["branch", "hidden_index"],
        columns="group",
        values="step_mean",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(["branch", "hidden_index"])


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def build_convergence_summary(
    layer_mean: pd.DataFrame, late_fraction: float
) -> pd.DataFrame:
    rows = []
    for identity, frame in layer_mean.groupby(
        ["branch", "group"], dropna=False, sort=True
    ):
        frame = frame.sort_values("hidden_index")
        x = frame["hidden_index"].to_numpy(dtype=float)
        y = frame["step_mean"].to_numpy(dtype=float)
        peak_index = int(np.argmax(y))
        peak = float(y[peak_index])
        final = float(y[-1])
        late_count = min(len(y), max(2, math.ceil(len(y) * late_fraction)))
        rows.append(
            {
                "branch": identity[0],
                "group": identity[1],
                "num_hidden_indices": len(y),
                "initial_hidden_index": float(x[0]),
                "initial_mean_distance": float(y[0]),
                "peak_hidden_index": float(x[peak_index]),
                "peak_mean_distance": peak,
                "final_hidden_index": float(x[-1]),
                "final_mean_distance": final,
                "peak_to_final_reduction": (
                    (peak - final) / peak if peak > 0 else np.nan
                ),
                "all_layer_slope": slope(x, y),
                "post_peak_slope": slope(x[peak_index:], y[peak_index:]),
                "late_layer_slope": slope(x[-late_count:], y[-late_count:]),
            }
        )
    return pd.DataFrame(rows)


def build_layer_delta(layer_mean: pd.DataFrame) -> pd.DataFrame:
    ordered = layer_mean.sort_values(["branch", "group", "hidden_index"]).copy()
    ordered["delta_step_mean"] = ordered.groupby(
        ["branch", "group"], dropna=False, sort=False
    )["step_mean"].diff()
    return ordered[
        ["branch", "group", "hidden_index", "step_mean", "delta_step_mean"]
    ]


def main() -> None:
    args = parse_args()
    if not 0 < args.late_fraction <= 1:
        raise ValueError("--late-fraction must be in (0, 1].")

    requested_groups = [GROUP_MAP.get(group, group) for group in args.groups]
    input_path = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (input_path.parent if input_path.is_file() else input_path)
        / "hidden_convergence_analysis"
    )

    data, file_count = load_all(input_path, requested_groups)
    observations = collapse_duplicate_rows(data)
    step_layer = build_step_layer_mean(observations)
    layer_mean = build_layer_mean(step_layer)
    layer_wide = make_wide_layer_table(layer_mean)
    convergence = build_convergence_summary(layer_mean, args.late_fraction)
    layer_delta = build_layer_delta(layer_mean)

    output_dir.mkdir(parents=True, exist_ok=True)
    step_layer.to_csv(output_dir / "mean_distance_by_step_and_layer.csv", index=False)
    layer_mean.to_csv(output_dir / "mean_distance_by_layer.csv", index=False)
    layer_wide.to_csv(output_dir / "mean_distance_by_layer_wide.csv", index=False)
    layer_delta.to_csv(output_dir / "layer_mean_delta.csv", index=False)
    convergence.to_csv(output_dir / "layer_convergence_summary.csv", index=False)

    print(f"Input files: {file_count}")
    print(f"Observations: {len(observations)}")
    print(f"Groups: {', '.join(sorted(data['group'].unique()))}")
    print("For each layer: mean across rollouts within step, then mean across steps.")
    print(f"Output directory: {output_dir}")
    print("\nLayer convergence summary")
    print(convergence.to_string(index=False))


if __name__ == "__main__":
    main()
