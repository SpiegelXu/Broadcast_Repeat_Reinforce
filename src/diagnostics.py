"""分析只返回 rows/tensors，实际落盘由 ResultWriter 完成。"""

import math

import torch
import torch.nn.functional as F

from .attention_analysis import compute_shifted_c_gamma_CA, analyze_anchor_transitions
from .attention_analysis import (
    analyze_single_layer,
    analyze_adjacent_layers,
)
from .heatmap_analysis import analyze_attention_heatmaps
from .lookahead_analysis import analyze_lookahead
from .spectral_norm import analyze_beta_S_with_fisher_split


def compute_attention_weights(q, k):
    if q.shape[-3] != k.shape[-3]:
        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
    return torch.softmax(q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)


def _mean_cosine_distance(tensor):
    """
        H_bar = mean_i(H_i)

        distance =
            sum_i ||H_i - H_bar||_2
            / ((N - 1) * ||H_bar||_2)
    """
    n = tensor.shape[0]

    if n < 2:
        return float("nan")

    tensor = tensor.float()

    h_bar = tensor.mean(
        dim=0,
        keepdim=True,
    )

    distances = torch.linalg.vector_norm(
        tensor - h_bar,
        ord=2,
        dim=-1,
    )

    h_bar_norm = torch.linalg.vector_norm(
        h_bar.squeeze(0),
        ord=2,
    )

    relative_distance = (
        distances.sum()
        / (
            (n - 1)
            * h_bar_norm.clamp_min(1e-12)
        )
    )

    return relative_distance.item()

def hidden_state_analysis(result, config):
    distance_rows, rank_rows = [], []
    B = len(result.metadata["prompt_lens"])
    branches = [("full", 0)]
    if result.metadata["cfg_scale"] > 0 and "sparse" in config["branches"]:
        branches.append(("sparse", B))
    for step, hidden_states in result.all_step_hidden_state_record.items():
        for branch, offset in branches:
            for j, p_len in enumerate(result.metadata["prompt_lens"]):
                for layer, hidden in enumerate(hidden_states):
                    tensor = hidden[j + offset, p_len:p_len + result.metadata["max_new_tokens"]].float()
                    row = {"step": step, "batch_id": j, "branch": branch, "hidden_index": layer}
                    if config["hidden_state_analysis"]:
                        distance_rows.append({**row, "group": "all", "distance": _mean_cosine_distance(tensor)})
                        for slot in range(result.metadata["group_size"]):
                            distance_rows.append({**row, "group": f"slot_{slot}",
                                                  "distance": _mean_cosine_distance(tensor[slot::result.metadata["group_size"]])})
                    if config["hidden_state_rank"]:
                        S = torch.linalg.svdvals(tensor)
                        numerical_rank = int((S > S.max() * 1e-6).sum())
                        S_norm = S / S[0].clamp_min(1e-10)
                        rank_rows.append({**row, "rank": numerical_rank, "full_rank": min(tensor.shape),
                                          "rank_ratio": numerical_rank / min(tensor.shape),
                                          "sv_mean": S_norm.mean().item(), "sv_std": S_norm.std().item(),
                                          "sv_min": S_norm.min().item(), "sv_max": S_norm.max().item()})
    return distance_rows, rank_rows

'''
#############################################################
####################This is the cpu version##################
#############################################################

def attention_statistics(model, result, config):
    blocks = [block for block in model.get_blocks() if block.my_captured_steps
              and (config["layers"] is None or block.layer_id in config["layers"])]
    
    attention_rows, gain_rows = [], []
    
    if not blocks:
        return attention_rows, gain_rows
    
    layer_ids = [block.layer_id for block in blocks]
    B = len(result.metadata["prompt_lens"])
    branches = [("full", 0)]
    
    if result.metadata["cfg_scale"] > 0 and "sparse" in config["branches"]:
        branches.append(("sparse", B))
        
    anchor_config = dict(config["anchor"])
    exclude_mask_keys = anchor_config.pop("exclude_mask_keys")
    
    for step_idx, step in enumerate(blocks[0].my_captured_steps):
        # [HEATMAP] capture 可能是两类分析的并集；这里仍遵守原 analysis.steps。
        if config["steps"] is not None and step not in config["steps"]:
            continue
        for branch, offset in branches:
            
            for j, p_len in enumerate(result.metadata["prompt_lens"]):
                T = p_len + result.metadata["max_new_tokens"]
                q_list, k_list = [], []
                for block in blocks:
                    q = block.my_captured_q[step_idx][j + offset, :, :T].float()
                    k = block.my_captured_k[step_idx][j + offset, :, :T].float()
                    if q.shape[0] != k.shape[0]:
                        k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
                    q_list.append(q)
                    k_list.append(k)
                    
                    device = q.device
                    
                    if q.shape[0] != k.shape[0]:
                        k = k.repeat_interleave(
                            q.shape[0] // k.shape[0],
                            dim=0,
                        )

                    A = compute_attention_weights(q, k)
                    
                Q, K = torch.stack(q_list), torch.stack(k_list)
                A = compute_attention_weights(Q, K)
                
                ca_metrics = compute_shifted_c_gamma_CA(A)
                positions = torch.arange(T)
                tokens = result.forward_inputs[step][j + offset, :T]
                scopes = {"all": torch.ones(T, dtype=torch.bool),
                          "history": positions < p_len,
                          "forecast": positions >= p_len,
                          "masked": tokens == model.config.mask_token_id}
                if exclude_mask_keys:
                    scopes["is_mask_key"] = tokens == model.config.mask_token_id
                    
                transitions = analyze_anchor_transitions(
                    Q, K, A, ca_metrics["c"], scopes=scopes, mode=config["ca_g_mode"], **anchor_config
                )
                
                common = {"step": step, "batch_id": j, "branch": branch,
                          "attention_source": "qk_softmax_pre_svd"}
                
                for l, layer in enumerate(layer_ids):
                    for h in range(Q.shape[1]):
                        anchors = torch.where(transitions["anchor_mask"][l, h])[0].tolist()
                        row = {**common, "layer": layer, "head": h,
                               "C_A": transitions["C_A"][l, h].item(),
                               "C_O": transitions["C_O"][l, h].item(),
                               "gamma": ca_metrics["gamma"][l, h].item(),
                               "num_anchors": len(anchors), "anchor_positions": anchors,
                               "valid_head": bool(transitions["valid_heads"][l, h]),
                               "q_threshold": anchor_config["q_threshold"],
                               "tail_fraction": anchor_config["tail_fraction"],
                               "tail_margin": anchor_config["tail_margin"]}
                        
                        if config["attention_statistics"]:
                            S = torch.linalg.svdvals(A[l, h])
                            row["rank"] = int((S > 1e-5 * S[0]).sum())
                            
                        attention_rows.append(row)
                        
                if config["ca_g"]:
                    for scope, summary in transitions["scope_summaries"].items():
                        for l in range(len(layer_ids) - 1):
                            if layer_ids[l + 1] != layer_ids[l] + 1:
                                continue
                            gain_rows.append({**common, "scope": scope,
                                              "source_layer": layer_ids[l], "target_layer": layer_ids[l + 1],
                                              "mode": config["ca_g_mode"],
                                              "layer_gain_sum": summary["layer_gain_sum"][l].item(),
                                              "layer_gain_mean": summary["layer_gain_mean"][l].item(),
                                              "layer_gain_sum_scaled": summary["layer_gain_sum_scaled"][l].item(),
                                              "layer_gain_mean_scaled": summary["layer_gain_mean_scaled"][l].item(),
                                              "valid_head_count": int(summary["valid_head_count"][l])})
    return attention_rows, gain_rows


'''

def _compact_layer_result(result):
    """ only put the Gain and CSV needed tensor back to CPU。"""
    return {
        "g": result["g"].detach().cpu(),
        "valid_heads": result["valid_heads"].detach().cpu(),
        "C_A": result["C_A"].detach().cpu(),
        "C_O": result["C_O"].detach().cpu(),
        "gamma": result["gamma"].detach().cpu(),
        "anchor_mask": result["anchor_mask"].detach().cpu(),
        "d_k": result["d_k"],
    }

def attention_statistics(model, result, config):
    blocks = [block for block in model.get_blocks() if block.my_captured_steps
              and (config["layers"] is None or block.layer_id in config["layers"])]
    
    #print(next(blocks[0].parameters()).device)
    
    attention_rows, gain_rows = [], []
    
    if not blocks:
        return attention_rows, gain_rows
    
    layer_ids = [block.layer_id for block in blocks]
    B = len(result.metadata["prompt_lens"])
    branches = [("full", 0)]
    
    if result.metadata["cfg_scale"] > 0 and "sparse" in config["branches"]:
        branches.append(("sparse", B))
        
    anchor_config = dict(config["anchor"])
    exclude_mask_keys = anchor_config.pop("exclude_mask_keys")
    
    for step_idx, step in enumerate(blocks[0].my_captured_steps):
        if config["steps"] is not None and step not in config["steps"]:
            continue

        for branch, offset in branches:
            for j, p_len in enumerate(result.metadata["prompt_lens"]):
                T = p_len + result.metadata["max_new_tokens"]

                positions = torch.arange(T)
                tokens = result.forward_inputs[step][j + offset, :T]

                scopes = {
                    "all": torch.ones(
                        T,
                        dtype=torch.bool,
                    ),
                    "history": positions < p_len,
                    "forecast": positions >= p_len,
                    "masked": (
                        tokens
                        == model.config.mask_token_id
                    ),
                }

                if exclude_mask_keys:
                    scopes["is_mask_key"] = (tokens == model.config.mask_token_id)

                common = {
                    "step": step,
                    "batch_id": j,
                    "branch": branch,
                    "attention_source": "qk_softmax_pre_svd",
                }

                previous_result = None
                previous_layer = None

                for block in blocks:
                    layer = block.layer_id

                    #  q/k is saved at its block GPU。
                    q = block.my_captured_q[step_idx][j + offset, :, :T].float()
                    k = block.my_captured_k[step_idx][j + offset, :, :T].float()

                    device = q.device
                    
                    #print(device)
                    

                    # GQA
                    if q.shape[0] != k.shape[0]:
                        k = k.repeat_interleave(
                            q.shape[0] // k.shape[0],
                            dim=0,
                        )

                    A = compute_attention_weights( q, k,)

                    is_mask_key = None

                    if exclude_mask_keys:
                        is_mask_key = scopes["is_mask_key"].to(device)

                    layer_result_gpu = analyze_single_layer(
                        q,
                        k,
                        A,
                        is_mask_key=is_mask_key,
                        **anchor_config,
                    )
                    
                    # error ratio
                    E_A_frobenius_ratio = layer_result_gpu["E_A_frobenius_ratio"]

                    
                    ranks = None
                    '''
                    if config["attention_statistics"]:
                        singular_values = torch.linalg.svdvals(A)

                        ranks = (
                            singular_values
                            >
                            1e-5 * singular_values[:, :1]
                        ).sum(dim=-1).cpu()
                        
                    '''

                    current_result = _compact_layer_result(
                        layer_result_gpu
                    )

                    num_heads = current_result["C_A"].shape[0]

                    for h in range(num_heads):
                        anchors = torch.where(
                            current_result["anchor_mask"][h]
                        )[0].tolist()

                        row = {
                            **common,
                            "layer": layer,
                            "head": h,
                            "C_A": current_result[
                                "C_A"
                            ][h].item(),
                            "C_O": current_result[
                                "C_O"
                            ][h].item(),
                            "gamma": current_result[
                                "gamma"
                            ][h].item(),
                            "num_anchors": len(anchors),
                            "anchor_positions": anchors,
                            "valid_head": bool(
                                current_result[
                                    "valid_heads"
                                ][h]
                            ),
                            "q_threshold": anchor_config[
                                "q_threshold"
                            ],
                            "tail_fraction": anchor_config[
                                "tail_fraction"
                            ],
                            "tail_margin": anchor_config[
                                "tail_margin"
                            ],
                            "E_A_frobenius_ratio":E_A_frobenius_ratio[h].item(),
                        }

                        if ranks is not None:
                            row["rank"] = int(ranks[h])

                        attention_rows.append(row)

                    if (
                        config["ca_g"]
                        and previous_result is not None
                        and layer == previous_layer + 1
                    ):
                        transition = analyze_adjacent_layers(
                            previous_result,
                            current_result,
                            scopes,
                        )

                        for scope, summary in transition[
                            "scope_summaries"
                        ].items():
                            gain_rows.append({
                                **common,
                                "scope": scope,
                                "source_layer": previous_layer,
                                "target_layer": layer,
                                "mode": config["ca_g_mode"],
                                "layer_gain_sum": summary[
                                    "layer_gain_sum"
                                ].item(),
                                "layer_gain_mean": summary[
                                    "layer_gain_mean"
                                ].item(),
                                "layer_gain_sum_scaled": summary[
                                    "layer_gain_sum_scaled"
                                ].item(),
                                "layer_gain_mean_scaled": summary[
                                    "layer_gain_mean_scaled"
                                ].item(),
                                "valid_head_count": int(
                                    summary[
                                        "valid_head_count"
                                    ]
                                ),
                            })

                    
                    previous_result = current_result
                    previous_layer = layer

                    # delete A/Q/K 
                    del A
                    del layer_result_gpu
                    del q
                    del k

    return attention_rows, gain_rows


def run_analysis(model, result, config, context=None):

    tables, tensors = {}, {}
    
    if config["hidden_state_analysis"] or config["hidden_state_rank"]:
        distance_rows, rank_rows = hidden_state_analysis(result, config)
        tables["hidden_distance"] = distance_rows
        tables["hidden_rank"] = rank_rows
        
    if config["attention_statistics"] or config["ca_g"]:
        attention_rows, gain_rows = attention_statistics(model, result, config)
        tables["attention_statistics"] = attention_rows
        tables["ca_g"] = gain_rows
        
    if config["beta_S_fisher_split"]:
        tables["beta_S_fisher_split"] = analyze_beta_S_with_fisher_split(model, result, config)
    
    # lookahead mode and slot convergence share the same need of token trajectories
    lookahead = config.get("lookahead", {})
    slot_convergence = config.get("slot_convergence", {})
    
    if lookahead.get("enabled", False) or slot_convergence.get("enabled", False):
        lookahead_rows, slot_rows = analyze_lookahead(
            result, context["settings"], context["scaler"], context["target"], lookahead
        )
        if lookahead.get("enabled", False) and lookahead["save_metrics"]:
            tables["lookahead_metrics"] = lookahead_rows
        if slot_convergence.get("enabled", False):
            tables["slot_convergence"] = slot_rows
    
    heatmap_rows, heatmap_tensors = analyze_attention_heatmaps(
        model, result, config.get("attention_heatmap", {"enabled": False})
    )
    
    if heatmap_rows:
        tables["attention_heatmap_index"] = heatmap_rows
        tensors.update(heatmap_tensors)
    if config["save_qkv"]:
        for block in model.get_blocks():
            if config["layers"] is not None and block.layer_id not in config["layers"]:
                continue
            for i, step in enumerate(block.my_captured_steps):
                if config["steps"] is not None and step not in config["steps"]:
                    continue
                prefix = f"step{step:04d}_layer{block.layer_id:02d}"
                tensors[prefix + "_q"] = block.my_captured_q[i]
                tensors[prefix + "_k"] = block.my_captured_k[i]
                tensors[prefix + "_v"] = block.my_captured_v[i]
    return tables, tensors
