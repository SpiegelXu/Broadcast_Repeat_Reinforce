from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge_config(base, updates):
    config = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = merge_config(config[key], value)
        else:
            config[key] = deepcopy(value)
    return config


def load_config(config_path, overrides=()):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    if parent is not None:
        config = merge_config(load_config(config_path.parent / parent), config)

    for item in overrides:
        key, value = item.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise KeyError(key)
        target[parts[-1]] = yaml.safe_load(value)

    validate_config(config)
    return config


def validate_config(config):
    """只检查会导致错误实验或不完整预测的几个条件。"""
    sampling, forecast = config["sampling"], config["forecast"]
    assert sampling["diffusion_steps"] > 0 and sampling["block_size"] > 0
    assert forecast["num_samples"] > 0 and forecast["batch_size"] > 0
    assert sampling["cfg_scale"] >= 0
    # [CFG] null 表示按数据原有 future 长度运行；非 null 仅作为额外上限。
    if config["data"]["forecast_horizon"] is not None:
        assert config["data"]["forecast_horizon"] > 0
    # [DATA] null 明确表示完整 future；数值时才施加相对 train 长度上限。
    if config["data"]["max_forecast_ratio"] is not None:
        assert 0 < config["data"]["max_forecast_ratio"] <= 1
    # [SAMPLER] 两个旧 token-queue 策略现在进入同一 YAML 选择入口。
    assert sampling["sampler"] in {
        "MDLMSampler", "MDLMLensSampler", "MDLMtruncateLensSampler", "MDLMswitchLensSampler"
    }
    assert sampling["token_queue"]["repeat_window"] >= 2
    if sampling["sampler"] in {"MDLMtruncateLensSampler", "MDLMswitchLensSampler"}:
        assert sampling["one_step_per_mask"]
        # [SAMPLER] stochastic transfer 可能一次抽中多个 token，会破坏“连续 step queue”定义。
        assert not sampling["stochastic_transfer"]
    assert sampling["cfg_keep_policy"] in {"prompt_token_overlap", "explicit_token_ids"}
    assert sampling["cfg_mask_scope"] in {"initial_known", "history"}
    assert sampling["length_mode"] in {"legacy_average", "fixed_slots"}
    assert sampling["remasking"] in {"low_confidence", "random"}
    assert config["svd"]["rank_mode"] in {"fixed", "fraction"}
    assert config["svd"]["branches"] in {"full", "both"}
    if config["svd"]["enabled"]:
        assert config["svd"]["rank"] > 0
        assert 0 < config["svd"]["rank_fraction"] <= 1
    if forecast["aggregation"] == "energy_hybrid":
        assert 0 <= forecast["K"] < forecast["num_samples"]
    else:
        assert forecast["aggregation"] in {"median", "mean"}
    # [ANALYSIS] look-ahead/slot convergence 是 token 分析；不要求 hidden-state 开关。
    lookahead = config["analysis"]["lookahead"]
    if lookahead["save_panel"] and lookahead["enabled"]:
        assert lookahead["save_metrics"]
    assert lookahead["early_steps"] >= 0 and lookahead["late_stride"] > 0
    assert lookahead["max_panels"] > 0

    heatmap = config["analysis"]["attention_heatmap"]
    assert heatmap["source"] in {"qk", "svd", "both"}
    assert heatmap["layer_parity"] in {"all", "odd", "even"}
    assert heatmap["head_parity"] in {"all", "odd", "even"}
    assert heatmap["layer_stride"] > 0 and heatmap["head_stride"] > 0
    assert heatmap["query_scope"] in {"all", "history", "forecast", "masked", "unmasked"}
    assert heatmap["key_scope"] in {"all", "history", "forecast", "masked", "unmasked"}
    assert heatmap["norm"] in {"linear", "power", "log"}
    if heatmap["enabled"] and heatmap["source"] in {"svd", "both"}:
        assert config["svd"]["enabled"] and config["svd"]["record_matrices"]


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
