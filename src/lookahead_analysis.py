from collections import Counter

import numpy as np

from .serialize import SerializerSettings, deserialize_str, serialize_arr


SLOT_NAMES = ("H", "T", "O", "comma")


def _agreement(values):
    if not values:
        return float("nan")
    return Counter(values).most_common(1)[0][1] / len(values)


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _repeat_rate(values):
    """计算所有不同位置有序对上的重复率。"""
    count = len(values)
    if count <= 1:
        return 0.0

    repeated_pairs = sum(
        frequency * (frequency - 1)
        for frequency in Counter(values).values()
    )
    return repeated_pairs / (count * (count - 1))


def _decode_draft(token_string, settings, scaler, horizon):
    if token_string and not token_string.endswith(settings.time_sep):
        token_string += settings.time_sep
    raw_pred = deserialize_str(token_string, settings, ignore_last=False, steps=horizon)
    if raw_pred is None:
        return np.asarray([], dtype=float)
    return np.asarray(scaler.inv_transform(raw_pred[:horizon]), dtype=float)


def _complete_value_tokens(token_string, settings, digits_per_value, horizon):
    values = [value.strip() for value in token_string.split(settings.time_sep)]
    return [value for value in values[:horizon]
            if len(value) == digits_per_value and all(char.isdigit() for char in value)]


def analyze_lookahead(result, settings, scaler, target, config):
    """[LOOKAHEAD] 计算 draft MAE/repeat/confidence 和 [ANALYSIS] slot convergence。"""
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    target = np.asarray(target, dtype=float)
    horizon = len(target)
    group_size = result.metadata["group_size"]
    digits_per_value = group_size - 1
    rows, slot_rows = [], []
    previous_drafts = {}
    previous_slots = {j: [] for j in range(len(result.metadata["prompt_lens"]))}
    # [LOOKAHEAD] 与最终 R_excess 使用同一离散 token 定义，便于逐步和最终结果对照。
    target_tokens = [value.strip() for value in
                     serialize_arr(scaler.transform(target), settings).split(settings.time_sep)
                     if value.strip()]
    target_repeat_rate = _repeat_rate(target_tokens)

    for record in result.lookahead_token_records:
        batch_id = record["batch_id"]
        token_string = record["token_ids"]
        draft = _decode_draft(token_string, settings, scaler, horizon)
        valid_length = min(len(draft), horizon)
        draft = draft[:valid_length]
        draft_mae = float(np.mean(np.abs(draft - target[:valid_length]))) if valid_length else float("nan")
        previous = previous_drafts.get(batch_id)
        common = min(valid_length, 0 if previous is None else len(previous))
        update_mae = (float(np.mean(np.abs(draft[:common] - previous[:common])))
                      if common else float("nan"))
        previous_drafts[batch_id] = draft.copy()
        
        """
        values = _complete_value_tokens(token_string, settings, digits_per_value, horizon)
        agreements = [_agreement([value[slot] for value in values])
                      for slot in range(digits_per_value)]
        repeat_rate = _repeat_rate(values)
        """
        
        values = _complete_value_tokens(
            token_string, settings, digits_per_value, horizon
        )

        # 分别取得百位、十位和个位的完整 slot 序列。
        slot_values = [
            [value[slot] for value in values]
            for slot in range(digits_per_value)
        ]

        # Dominant slot agreement，对应论文中的 Agree_s(t)。
        agreements = [
            _agreement(slot_sequence)
            for slot_sequence in slot_values
        ]

        # Slot-wise pairwise repetition，对应论文中的 R_slot,s。
        slot_repeat_rates = [
            _repeat_rate(slot_sequence)
            for slot_sequence in slot_values
        ]

        # Value-level pairwise repetition，对应论文中的 R_draft(t)。
        repeat_rate = _repeat_rate(values)
        
        

        confidence = np.asarray(record["draft_confidence"], dtype=float)
        masked_before = np.asarray(record["masked_token_mask_before"], dtype=bool)
        masked_after = np.asarray(record["masked_token_mask_after"], dtype=bool)
        digit_mask = np.arange(len(confidence)) % group_size < digits_per_value
        slot_confidence = [_mean(confidence[slot::group_size]) for slot in range(digits_per_value)]
        revealed_values = []
        for value_index in range(min(horizon, len(masked_after) // group_size)):
            start = value_index * group_size
            revealed_values.append(bool((~masked_after[start:start + digits_per_value]).all()))

        padded_draft = draft.tolist()
        if padded_draft and len(padded_draft) < horizon:
            padded_draft.extend([padded_draft[-1]] * (horizon - len(padded_draft)))
        rows.append({
            "step": record["step"],
            "block": record["block"],
            "block_step": record["block_step"],
            "batch_id": batch_id,
            "valid_length": valid_length,
            "coverage": valid_length / horizon,

            "draft_mae": draft_mae,
            "draft_update_mae": update_mae,

            # Appendix B: value-level pairwise repetition.
            "draft_repeat_rate": repeat_rate,
            "target_repeat_rate": target_repeat_rate,
            "draft_repeat_excess": repeat_rate - target_repeat_rate,

            "mean_draft_confidence": _mean(confidence[digit_mask]),
            "mean_masked_confidence": _mean(
                confidence[digit_mask & masked_before]
            ),
            "mean_revealed_confidence": _mean(
                confidence[digit_mask & ~masked_after]
            ),
            "chosen_confidence": _mean(record["chosen_confidence"]),

            # Appendix B: dominant slot agreement.
            **{
                f"slot_{SLOT_NAMES[slot]}_agreement": agreements[slot]
                for slot in range(digits_per_value)
            },

            # Appendix B: slot-wise pairwise repetition.
            **{
                f"slot_{SLOT_NAMES[slot]}_repeat_rate":
                    slot_repeat_rates[slot]
                for slot in range(digits_per_value)
            },

            **{
                f"slot_{SLOT_NAMES[slot]}_confidence":
                    slot_confidence[slot]
                for slot in range(digits_per_value)
            },

            "fully_revealed_values": int(sum(revealed_values)),
            "revealed_value_ratio": (
                float(np.mean(revealed_values))
                if revealed_values else 0.0
            ),

            "draft_values": padded_draft,
            "revealed_value_mask": revealed_values,
        })

        # [ANALYSIS] 复用原 canvas_utils Figure 2b/2c：slot agreement + scheduler persistence。
        relative_positions = record.get("relative_transfer_index", [])
        selected_slots = [pos % group_size for pos in relative_positions
                          if pos % group_size < digits_per_value]
        history = previous_slots[batch_id]
        for slot in selected_slots:
            history.append(slot)
        same_count = sum(left == right for left, right in zip(history, history[1:]))
        slot_rows.append({
            "step": record["step"], "block": record["block"],
            "block_step": record["block_step"], "batch_id": batch_id,
            "complete_values": len(values),
            **{f"slot_{SLOT_NAMES[slot]}_agreement": agreements[slot]
               for slot in range(digits_per_value)},
            "selected_slots": [SLOT_NAMES[slot] for slot in selected_slots],
            "slot_persistence": same_count / (len(history) - 1) if len(history) > 1 else float("nan"),
            "slot_null": 1.0 / digits_per_value,
        })
    return rows, slot_rows


def select_panel_steps(rows, config):
    """[LOOKAHEAD] 显式 key_steps 优先；否则选择首段、等距后段并保证最后一步。"""
    all_steps = sorted({row["step"] for row in rows})
    requested = config["panel_steps"]
    if requested is not None:
        return [step for step in requested if step in all_steps]
    early_steps = config["early_steps"]
    late_stride = config["late_stride"]
    selected = [step for step in all_steps
                if step < early_steps or (step - early_steps) % late_stride == 0]
    if all_steps and all_steps[-1] not in selected:
        selected.append(all_steps[-1])
    max_panels = config["max_panels"]
    if len(selected) > max_panels:
        indices = np.linspace(0, len(selected) - 1, max_panels).round().astype(int)
        selected = [selected[index] for index in indices]
    return sorted(set(selected))


def build_lookahead_panel(rows, target, config):
    """[LOOKAHEAD] 返回原 small-multiples 风格 Figure；保存路径由 ResultWriter 决定。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_steps = select_panel_steps(rows, config)
    row_by_step = {row["step"]: row for row in rows if row["step"] in selected_steps}
    selected_steps = [step for step in selected_steps if row_by_step[step]["draft_values"]]
    if not selected_steps:
        return None

    target = np.asarray(target, dtype=float)
    fig, axes = plt.subplots(1, len(selected_steps), figsize=(4 * len(selected_steps), 3.7),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for panel_index, step in enumerate(selected_steps):
        ax = axes[panel_index]
        for previous_step in selected_steps:
            if previous_step >= step:
                break
            ax.plot(row_by_step[previous_step]["draft_values"], color="#cccccc",
                    linewidth=0.9, alpha=0.45)
        row = row_by_step[step]
        draft = np.asarray(row["draft_values"], dtype=float)
        ax.plot(target, color="#1f77b4", linestyle="--", linewidth=1.8,
                label="Truth" if panel_index == 0 else None)
        ax.plot(draft, color="#d62728", linewidth=2.1, label=f"Step {step:02d}")
        revealed = np.asarray(row["revealed_value_mask"], dtype=bool)
        revealed = np.pad(revealed, (0, max(0, len(draft) - len(revealed))))[:len(draft)]
        indices = np.where(revealed)[0]
        if len(indices):
            ax.scatter(indices, draft[indices], color="#ff7f0e", edgecolor="black",
                       linewidth=0.6, marker="s", s=38, zorder=5,
                       label="Fully revealed" if panel_index == 0 else None)
        ax.set_title(f"Step {step:02d}\nMAE={row['draft_mae']:.4f}")
        ax.set_xlabel("Horizon index")
        ax.grid(linestyle=":", alpha=0.45)
        if panel_index == 0:
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
    fig.suptitle("Look-ahead draft evolution", fontsize=13)
    fig.tight_layout()
    return fig


def build_slot_convergence_plot(rows):
    """[ANALYSIS] 复用原 Figure 2b/2c 的 slot agreement 与 persistence 视图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return None
    rows = sorted(rows, key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    colors = {"H": "#d62728", "T": "#ff7f0e", "O": "#2ca02c"}
    for slot in SLOT_NAMES[:3]:
        axes[0].plot(steps, [row[f"slot_{slot}_agreement"] for row in rows],
                     marker="o", linewidth=1.8, color=colors[slot], label=f"{slot} slot")
    axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Draft slot agreement")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    persistence = [row["slot_persistence"] for row in rows]
    axes[1].plot(steps, persistence, marker="s", color="#9467bd", label="Observed")
    axes[1].axhline(rows[0]["slot_null"], color="gray", linestyle="--", label="Null")
    axes[1].set(xlabel="Diffusion step", ylabel="Scheduler slot persistence", ylim=(-0.05, 1.05))
    axes[1].legend()
    for ax in axes:
        ax.grid(linestyle=":", alpha=0.5)
    fig.suptitle("Slot-wise convergence")
    fig.tight_layout()
    return fig
