import math

import torch


def select_indices(total, explicit, parity="all", stride=1):
    indices = list(range(total)) if explicit is None else [index for index in explicit if index < total]
    if parity == "odd":
        indices = [index for index in indices if index % 2 == 1]
    elif parity == "even":
        indices = [index for index in indices if index % 2 == 0]
    return indices[::stride]


def _position_scope(scope, tokens, prompt_len, mask_id):
    positions = torch.arange(len(tokens))
    if scope == "history":
        return positions < prompt_len
    if scope == "forecast":
        return positions >= prompt_len
    if scope == "masked":
        return tokens == mask_id
    if scope == "unmasked":
        return tokens != mask_id
    return torch.ones(len(tokens), dtype=torch.bool)


def analyze_attention_heatmaps(model, result, config):

    if not config["enabled"] or config["source"] not in {"qk", "both"}:
        return [], {}

    blocks = {block.layer_id: block for block in model.get_blocks() if block.my_captured_steps}
    layer_ids = select_indices(
        len(model.get_blocks()), config["layers"], config["layer_parity"], config["layer_stride"]
    )
    layer_ids = [layer_id for layer_id in layer_ids if layer_id in blocks]
    rows, tensors = [], {}
    B = len(result.metadata["prompt_lens"])
    branches = [("full", 0)]
    if result.metadata["cfg_scale"] > 0 and "sparse" in config["branches"]:
        branches.append(("sparse", B))

    for layer_id in layer_ids:
        block = blocks[layer_id]
        for capture_index, step in enumerate(block.my_captured_steps):
            if config["steps"] is not None and step not in config["steps"]:
                continue
            q_batch = block.my_captured_q[capture_index].float()
            k_batch = block.my_captured_k[capture_index].float()
            if q_batch.shape[1] != k_batch.shape[1]:
                k_batch = k_batch.repeat_interleave(q_batch.shape[1] // k_batch.shape[1], dim=1)
            head_ids = select_indices(
                q_batch.shape[1], config["heads"], config["head_parity"], config["head_stride"]
            )
            for branch, offset in branches:
                for batch_id, prompt_len in enumerate(result.metadata["prompt_lens"]):
                    sample_id = result.metadata.get("sample_start", 0) + batch_id
                    if config["sample_ids"] is not None and sample_id not in config["sample_ids"]:
                        continue
                    T = prompt_len + result.metadata["max_new_tokens"]
                    tokens = result.forward_inputs[step][batch_id + offset, :T]
                    query_mask = _position_scope(
                        config["query_scope"], tokens, prompt_len, model.config.mask_token_id
                    )
                    key_mask = _position_scope(
                        config["key_scope"], tokens, prompt_len, model.config.mask_token_id
                    )
                    query_positions = torch.where(query_mask)[0]
                    key_positions = torch.where(key_mask)[0]
                    for head in head_ids:
                        q = q_batch[batch_id + offset, head, :T][query_mask]
                        k = k_batch[batch_id + offset, head, :T][key_mask]
                        matrix = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
                        key = (f"attention_step{step:04d}_layer{layer_id:02d}_{branch}"
                               f"_batch{batch_id:04d}_head{head:02d}")
                        tensors[key] = matrix.cpu()
                        rows.append({
                            "step": step, "batch_id": batch_id, "branch": branch,
                            "layer": layer_id, "head": head,
                            "attention_source": "qk_softmax_pre_svd",
                            "query_scope": config["query_scope"], "key_scope": config["key_scope"],
                            "prompt_len": prompt_len,
                            "query_positions": query_positions.tolist(),
                            "key_positions": key_positions.tolist(), "matrix_key": key,
                        })
                        
    return rows, tensors


def _normalization(matrix, config):
    import matplotlib.colors as colors

    vmin, vmax = config["vmin"], config["vmax"]
    if config["norm"] == "power":
        return colors.PowerNorm(gamma=config["gamma"], vmin=vmin, vmax=vmax)
    if config["norm"] == "log":
        positive = matrix[matrix > 0]
        actual_vmin = float(positive.min()) if positive.size else 1e-8
        log_vmin = actual_vmin if vmin is None else max(vmin, actual_vmin)
        return colors.LogNorm(vmin=log_vmin, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def build_attention_heatmap(matrix, row, config):
    """ attention heatmap Figure set up"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = matrix.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=tuple(config["figsize"]))
    image = ax.imshow(matrix, aspect="auto", cmap=config["cmap"],
                      norm=_normalization(matrix, config), interpolation="nearest")
    ax.set_title(f"QK attention | step {row['step']} | layer {row['layer']} | head {row['head']} | {row['branch']}")
    ax.set_xlabel(f"Key position ({row['key_scope']})")
    ax.set_ylabel(f"Query position ({row['query_scope']})")
    _set_position_ticks(ax, row["key_positions"], axis="x")
    _set_position_ticks(ax, row["query_positions"], axis="y")
    if row["key_scope"] == "all":
        ax.axvline(row["prompt_len"] - 0.5, color="white", linestyle="--", linewidth=0.8)
    if row["query_scope"] == "all":
        ax.axhline(row["prompt_len"] - 0.5, color="white", linestyle="--", linewidth=0.8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    
    return fig


def _set_position_ticks(ax, positions, axis):
    if not positions:
        return
    count = min(8, len(positions))
    tick_ids = torch.linspace(0, len(positions) - 1, count).round().long().unique().tolist()
    labels = [positions[index] for index in tick_ids]
    if axis == "x":
        ax.set_xticks(tick_ids, labels)
    else:
        ax.set_yticks(tick_ids, labels)


def build_svd_attention_heatmap(before, after, row, config):
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = before.detach().cpu().float().numpy()
    after = after.detach().cpu().float().numpy()
    
    # color set up
    shared_config = dict(config)
    if shared_config["vmin"] is None:
        shared_config["vmin"] = float(min(before.min(), after.min()))
    if shared_config["vmax"] is None:
        shared_config["vmax"] = float(max(before.max(), after.max()))
    shared_norm = _normalization(torch.cat([
        torch.as_tensor(before).flatten(), torch.as_tensor(after).flatten()
    ]).numpy(), shared_config)
    fig, axes = plt.subplots(1, 2, figsize=(2 * config["figsize"][0], config["figsize"][1]),
                             sharex=True, sharey=True)
    for ax, matrix, title in zip(axes, [before, after], ["Before SVD", "After SVD"]):
        image = ax.imshow(matrix, aspect="auto", cmap=config["cmap"],
                          norm=shared_norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Key position")
        if "prompt_len" in row:
            ax.axvline(row["prompt_len"] - 0.5, color="white", linestyle="--", linewidth=0.8)
            ax.axhline(row["prompt_len"] - 0.5, color="white", linestyle="--", linewidth=0.8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Query position")
    fig.suptitle(f"SVD attention | step {row['step']} | layer {row['layer']} | head {row['head']} | {row['branch']}")
    fig.tight_layout()
    return fig
