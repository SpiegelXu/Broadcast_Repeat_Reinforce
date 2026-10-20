from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .serialize import serialize_arr, deserialize_str, SerializerSettings
from .llada_lens import llada_completion_fn, prepare_input

from .repetition import (
    compute_r_value_from_tokens,
    compute_slot_repetition,
)


@dataclass
class Scaler:
    min_: float = 0.0
    q: float = 0.98
    
    def transform(self, x):
        return (np.asarray(x) - self.min_) / self.q

    def inv_transform(self, x):
        return np.asarray(x) * self.q + self.min_


def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    history = np.asarray(history)
    history = history[~np.isnan(history)]
    if basic:
        min_ = 0.0
        q = max(np.quantile(np.abs(history), alpha), 0.01) * 4
    else:
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha) * (1 + 1e-3)
        if q == 0:
            q = 1.0
    return Scaler(float(min_), float(q))


def truncate_input(input_arr, input_str, settings, steps, tokenizer, sampling_config, context_length):
    input_str_chunks = input_str.rstrip(settings.time_sep).split(settings.time_sep)
    for i in range(len(input_str_chunks)):
        truncated_input_str = settings.time_sep.join(input_str_chunks[i:]) + settings.time_sep
        input_ids, _, _, max_new_tokens = prepare_input(
            truncated_input_str, steps, settings, tokenizer, sampling_config
        )
        if len(input_ids) + max_new_tokens <= context_length:
            return input_arr[i:], truncated_input_str
    raise ValueError("the length of prompt and the forecast area exceeds the context_length.")



def handle_prediction(pred, expected_length, strict=False):
    if pred is None or len(pred) == 0:
        return None
    if len(pred) < expected_length:
        if strict:
            return None
        pred = np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
    return pred[:expected_length]


def process_to_discrete_tokens(arr, scaler, settings):
    
    serialized_str = serialize_arr(scaler.transform(arr), settings)
    
    return np.array([t.strip() for t in serialized_str.split(settings.time_sep) if t.strip()])



def compute_repetition_metrics(preds, test, scalers, settings):
    """
    Compute repetition metrics for ground-truth and predicted sequences,
    including overall and digit-slot (H, T, O) repetition rates and their
    excess over ground truth.
    """
    
    metric_names = ("all", "H", "T", "O")
    gt_values = {name: [] for name in metric_names}
    pred_values = {name: [] for name in metric_names}
    for test_arr, series_preds, scaler in zip(test, preds, scalers):
        target_tokens = process_to_discrete_tokens(test_arr, scaler, settings)
        target_rates = {"all": compute_r_value_from_tokens(target_tokens),
                        **compute_slot_repetition(target_tokens)}
        for name in metric_names:
            gt_values[name].append(target_rates[name])
        for pred in series_preds:
            if pred is not None:
                pred_tokens = process_to_discrete_tokens(pred, scaler, settings)
                pred_rates = {"all": compute_r_value_from_tokens(pred_tokens),
                              **compute_slot_repetition(pred_tokens)}
                for name in metric_names:
                    pred_values[name].append(pred_rates[name])

    metrics = {}
    for name in metric_names:
        prefix = "R" if name == "all" else f"R_{name}"
        r_gt = float(np.mean(gt_values[name]))
        r_pred = float(np.mean(pred_values[name]))
        metrics[f"{prefix}_gt"] = r_gt
        metrics[f"{prefix}_pred"] = r_pred
        metrics[f"{prefix}_excess"] = r_pred - r_gt

    metrics["R_slot_mean_gt"] = float(np.mean([metrics[f"R_{slot}_gt"] for slot in ("H", "T", "O")]))
    metrics["R_slot_mean_pred"] = float(np.mean([metrics[f"R_{slot}_pred"] for slot in ("H", "T", "O")]))
    metrics["R_slot_mean_excess"] = metrics["R_slot_mean_pred"] - metrics["R_slot_mean_gt"]
    return metrics


def generate_llada_predictions(completion_fn, input_strs, steps, settings, scalers,
                               num_samples=1, temp=0.7, strict_handling=False,
                               series_ids=None, targets=None, **kwargs):
    completions_list, preds, prediction_status = [], [], []
    
    targets = [None] * len(input_strs) if targets is None else targets
    for series_id, input_str, scaler, target, series_steps in zip(series_ids, input_strs, scalers, targets, steps):
       
        completions = completion_fn(input_str=input_str, steps=series_steps, settings=settings,
                                    num_samples=num_samples, temp=temp,
                                    series_id=series_id, target=target, scaler=scaler, **kwargs)
        completions_list.append(completions)
        
        series_preds = []
        
        for sample_id, completion in enumerate(completions):
            raw_pred = deserialize_str(completion, settings, ignore_last=False)
            raw_length = 0 if raw_pred is None else len(raw_pred)
            pred = handle_prediction(raw_pred, series_steps, strict_handling)
            series_preds.append(None if pred is None else scaler.inv_transform(pred))
            status = "failed" if pred is None else ("padded" if raw_length < series_steps else "ok")
            prediction_status.append({"series_id": series_id, "sample_id": sample_id,
                                      "raw_length": raw_length, "expected_length": series_steps,
                                      "status": status, "truncated": raw_length > series_steps})
        preds.append(series_preds)
        
    return preds, completions_list, prediction_status


def multi_sample(samples, K=2, method="energy_hybrid"):
    """
    Combine multiple forecast samples to improve prediction accuracy.
    The energy-hybrid method selects a high-energy sample and adds
    median-magnitude signed increments as perturbations, which may reduce MAE.

    For studying intrinsic repetition behavior, use raw individual samples
    without this post-processing to avoid altering the repetition patterns.

    """
    combined_series, select_data = [], []
    
    for s in samples:
        s_tensor = torch.tensor(s.dropna().values, dtype=torch.float32)
        if s_tensor.shape[0] == 0:
            raise ValueError("no successful generated data, please check the prediction_status")
        if method == "median":
            result = torch.tensor(np.median(s_tensor.numpy(), axis=0))
            selected = result.clone()
        elif method == "mean":
            result = s_tensor.mean(dim=0)
            selected = result.clone()
        else:
            if K >= s_tensor.shape[0]:
                raise ValueError("sample number is not enough to use K")
            differences = torch.diff(s_tensor, dim=-1)
            energy = torch.sum(differences ** 2, dim=1)
            select_indices = torch.argsort(energy, descending=True)[K]
            selected = s_tensor[select_indices].clone()
            sorted_indices = torch.argsort(torch.abs(differences), dim=0)
            median_indices = sorted_indices[s_tensor.shape[0] // 2]
            final_median_signed = differences[median_indices, torch.arange(differences.shape[1])]
            result = selected.clone()
            result[1:] += final_median_signed
        combined_series.append(result)
        select_data.append(selected)
    
    if len({len(x) for x in combined_series}) == 1:
        return torch.stack(combined_series), torch.stack(select_data)
    return combined_series, select_data


def get_llada_predictions_data(train, test, settings, num_samples=3, temp=1.0,
                               alpha=1.0, beta=0.3, basic=False, *, model, tokenizer,
                               sampling_config, analysis_config, svd_config,
                               batch_size=1, context_length=8192, aggregation="energy_hybrid",
                               K=2, strict_handling=False, series_ids=None, on_sample=None):
    
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if isinstance(train, pd.Series) or np.isscalar(train[0]):
        train, test = [train], [test]
        
    train = [np.asarray(x) for x in train]
    test = [np.asarray(x) for x in test]
    # [DATA] 每条序列使用自己的预测长度。
    steps = [len(t) for t in test]
    series_ids = list(range(len(train))) if series_ids is None else series_ids
    scalers = [get_scaler(x, alpha, beta, basic) for x in train]
    input_strs = [serialize_arr(scaler.transform(x), settings) for x, scaler in zip(train, scalers)]
    input_arrs, input_strs = zip(*[
        truncate_input(x, text, settings, series_steps, tokenizer, sampling_config, context_length)
        for x, text, series_steps in zip(train, input_strs, steps)
    ])
    
    preds, completions_list, prediction_status = generate_llada_predictions(
        llada_completion_fn, input_strs, steps, settings, scalers, targets=test,
        num_samples=num_samples, temp=temp, strict_handling=strict_handling, series_ids=series_ids,
        model=model, tokenizer=tokenizer, sampling_config=sampling_config,
        analysis_config=analysis_config, svd_config=svd_config, batch_size=batch_size,
        on_sample=on_sample,
    )
    
    
    samples = [pd.DataFrame([p if p is not None else np.full(series_steps, np.nan) for p in row])
               for row, series_steps in zip(preds, steps)]
    medians = [s.median(axis=0) for s in samples]
    combined_series, select_data_tensor = multi_sample(samples, K=K, method=aggregation)
    
    if isinstance(combined_series, torch.Tensor):
        final_output = combined_series.numpy()
        select_sequence = select_data_tensor.numpy()
    else:
        final_output = [x.numpy() for x in combined_series]
        select_sequence = [x.numpy() for x in select_data_tensor]
        
    maes = np.asarray([np.mean(np.abs(pred - target)) for pred, target in zip(final_output, test)])
    return {
        "samples": samples, "median": medians, "completions_list": completions_list,
        "input_strs": list(input_strs), "input_arrs": input_arrs,
        "final_output": final_output, "select_sequence": select_sequence,
        "scalers": [{"min_": s.min_, "q": s.q} for s in scalers],
        "prediction_status": prediction_status, "series_ids": series_ids,
        "maes": maes, "mae": float(np.mean(maes)),
        **compute_repetition_metrics(preds, test, scalers, settings),
    }
