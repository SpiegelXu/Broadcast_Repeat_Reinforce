import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .sampler_utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class MDLMSamplerConfig:
    max_new_tokens: int = 128
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] = field(default_factory=list)
    cfg_mask_scope: str = "initial_known"
    suppress_tokens: list[int] = field(default_factory=list)
    begin_suppress_tokens: list[int] = field(default_factory=list)
    good_tokens: list[int] = field(default_factory=list)
    right_shift_logits: bool = False
    prefill_commas: bool = True
    digits_per_value: int = 3
    comma_id: int = 11
    # SAMPLER truncate / switch share token queue parameters；baseline doesn't need such parameter
    queue_repeat_window: int = 3
    avoid_same_slot_on_reroute: bool = True
    use_second_choice_on_reroute: bool = True
    return_dict: bool = True


@dataclass
class SamplerOutput:
    sequences: torch.Tensor
    histories: list = field(default_factory=list)
    lookahead_token_records: list = field(default_factory=list)
    csv_records: list = field(default_factory=list)
    save_data: list = field(default_factory=list)
    diffusion_lens_records: list = field(default_factory=list)
    un_cond_lens_record: list = field(default_factory=list)
    all_step_hidden_state_record: dict = field(default_factory=dict)
    forward_inputs: dict = field(default_factory=dict)
    first_unmask: list = field(default_factory=list)
    token_queue_records: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def build_sparse_history_mask(x, attention_mask, prompt_lens, config, mask_id):
    
    unmasked_index = (x != mask_id) & attention_mask.bool()
    if config.cfg_mask_scope == "history":
        positions = torch.arange(x.shape[1], device=x.device)
        unmasked_index &= positions[None] < torch.tensor(prompt_lens, device=x.device)[:, None]
    keep_mask = torch.isin(x, torch.tensor(config.cfg_keep_tokens, device=x.device, dtype=torch.long))
    return unmasked_index & ~keep_mask, keep_mask


@dataclass
class MDLMSampler:
    model: object
    tokenizer: object
    analysis: dict = field(default_factory=dict)

    # Flag for SAMPLER to determine which class to use 
    queue_mode = None

    @torch.no_grad()
    def sample(self, inputs, config=None):
        if config is None:
            config = MDLMSamplerConfig()

        steps, block_size = config.steps, config.block_size
        max_new_tokens = config.max_new_tokens
        cfg_scale = config.cfg_scale
        mask_id, eos_id = self.tokenizer.mask_token_id, self.tokenizer.eos_token_id
        device = self.model.get_input_embeddings().weight.device
        inputs = [torch.as_tensor(p, dtype=torch.long, device=device) for p in inputs]
        prompt_lens = [len(p) for p in inputs]
        B, T = len(inputs), max(prompt_lens) + max_new_tokens

        x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        group_size = config.digits_per_value + 1
        for j, p in enumerate(inputs):
            start, end = prompt_lens[j], prompt_lens[j] + max_new_tokens
            x[j, :start] = p
            x[j, start:end] = mask_id
            attention_mask[j, :end] = 1
            if config.prefill_commas:
                x[j, start + config.digits_per_value:end:group_size] = config.comma_id

        unmasked_index, keep_mask = build_sparse_history_mask(
            x, attention_mask, prompt_lens, config, mask_id
        )
        result = SamplerOutput(sequences=x, first_unmask=[[] for _ in range(B)])
        result.metadata = {
            "prompt_lens": prompt_lens, "max_new_tokens": max_new_tokens,
            "requested_diffusion_steps": steps, "cfg_scale": cfg_scale,
            "cfg_keep_tokens": config.cfg_keep_tokens,
            "cfg_mask_scope": config.cfg_mask_scope,
            "sparse_mask_positions": [torch.where(row)[0].cpu().tolist() for row in unmasked_index],
            "keep_positions": [torch.where(row)[0].cpu().tolist() for row in keep_mask & attention_mask.bool()],
            "prefill_commas": config.prefill_commas, "group_size": group_size,
            "token_queue_mode": self.queue_mode,
        }

        save_histories = self.analysis.get("save_histories", False)
        output_logit_lens = self.analysis.get("output_logit_lens", False)
        
        need_hidden = output_logit_lens or any(self.analysis.get(key, False) for key in
            ["save_hidden_states", "hidden_state_analysis", "hidden_state_rank"])
        heatmap = self.analysis.get("attention_heatmap", {})
        heatmap_qk = heatmap.get("enabled", False) and heatmap.get("source") in {"qk", "both"}
        regular_forward_inputs = any(self.analysis.get(key, False) for key in
            ["capture_qkv", "save_qkv", "attention_statistics", "ca_g", "beta_S_fisher_split"])
        need_forward_inputs = heatmap_qk or regular_forward_inputs
        if save_histories:
            result.histories.append(x.detach().cpu().clone())

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        global_step = 0
       
        token_queues = [[] for _ in range(B)]
        for b in range(num_blocks):
            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=device)
            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens)
                block_mask_index[j, :end-start] = x[j, start:end] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps, stochastic=config.stochastic_transfer,
            )
            for i in range(num_transfer_tokens.shape[1]):
                chosen_steps = self.analysis.get("steps")
                record_step = chosen_steps is None or global_step in chosen_steps
                
                heatmap_record_step = (heatmap.get("enabled", False)
                                       and (heatmap.get("steps") is None
                                            or global_step in heatmap["steps"]))
                mask_index = x == mask_id
                self.model.set_sampling_step(global_step)

                # CFG mode
                if cfg_scale > 0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    model_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    x_, model_attention_mask = x, attention_mask

                outputs = self.model(
                    x_, attention_mask=model_attention_mask, use_cache=False,
                    output_hidden_states=need_hidden and record_step, return_dict=True,
                )
                if need_forward_inputs and ((regular_forward_inputs and record_step)
                                            or (heatmap_qk and heatmap_record_step)):
                    result.forward_inputs[global_step] = x_.detach().cpu().clone()

                if cfg_scale > 0:
                    full_logits, un_logits = outputs.logits.chunk(2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (full_logits - un_logits)
                else:
                    full_logits = outputs.logits
                    un_logits = None
                    logits = full_logits.clone()

                if self.analysis.get("save_logits", False) and record_step:
                    for j, p_len in enumerate(prompt_lens):
                        region = slice(p_len, p_len + max_new_tokens)
                        row = {"step": global_step, "block": b, "block_step": i, "batch_id": j,
                               "token_ids": config.good_tokens,
                               "logits": full_logits[j, region][:, config.good_tokens].float().cpu().tolist()}
                        if un_logits is not None:
                            row["un_logits"] = un_logits[j, region][:, config.good_tokens].float().cpu().tolist()
                            row["delta"] = (full_logits[j, region] - un_logits[j, region])[:, config.good_tokens].float().cpu().tolist()
                        result.save_data.append(row)

                logits[..., config.suppress_tokens] = -torch.inf
                if config.right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits_with_noise = add_gumbel_noise(logits, config.temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                
                # truncate/switch mode will choose the second highest token
                x0_top2 = None
                if self.queue_mode is not None:
                    x0_top2 = torch.topk(logits_with_noise, k=2, dim=-1).indices[..., 1]
                # 保留原 begin_suppress_tokens 的 confidence-only 行为。
                logits[..., config.begin_suppress_tokens] = -torch.inf
                # [LOOKAHEAD] model_confidence 始终表示模型概率；random remasking 仍只改变调度分数。
                p = F.softmax(logits, dim=-1)
                model_confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                if config.remasking == "low_confidence":
                    x0_p = model_confidence
                elif config.remasking == "random":
                    x0_p = torch.rand(x0.shape, device=device)
                else:
                    raise ValueError(config.remasking)

                confidence = torch.where(mask_index, x0_p, -torch.inf)
                for j, p_len in enumerate(prompt_lens):
                    confidence[j, :p_len + b * block_size] = -torch.inf
                    confidence[j, min(p_len + (b + 1) * block_size, p_len + max_new_tokens):] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                
                

                model_confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                draft_x0 = x0.clone()
                # baseline / truncate / switch select which transfer  mode
                transfer_index = self._select_transfer(
                    result, x0, x0_top2, confidence, num_transfer_tokens[:, i],
                    token_queues, prompt_lens, config, global_step, b, i,
                )
                committed_confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                for j in range(B):
                    select_index = torch.where(transfer_index[j])[0]
                    if not result.first_unmask[j] and select_index.numel():
                        result.first_unmask[j] = select_index.cpu().tolist()

                if record_step:
                    self._record_step(result, draft_x0, x0, transfer_index, confidence,
                                      model_confidence, committed_confidence, mask_index, prompt_lens,
                                      config, global_step, b, i)
                if need_hidden and record_step:
                    if output_logit_lens:
                        result.diffusion_lens_records.extend(self.diffusion_logit_process(
                            outputs.hidden_states, transfer_index, x0, global_step, b, "full", 0))
                        if cfg_scale > 0 and "sparse" in self.analysis.get("branches", ["full"]):
                            result.un_cond_lens_record.extend(self.diffusion_logit_process(
                                outputs.hidden_states, transfer_index, x0, global_step, b, "sparse", B))
                    if any(self.analysis.get(key, False) for key in
                           ["save_hidden_states", "hidden_state_analysis", "hidden_state_rank"]):
                        result.all_step_hidden_state_record[global_step] = [
                            h.detach().cpu() for h in outputs.hidden_states
                        ]

                x[transfer_index] = x0[transfer_index]
                if save_histories:
                    result.histories.append(x.detach().cpu().clone())
                global_step += 1
                del outputs, logits, logits_with_noise, full_logits, un_logits

        result.metadata["actual_diffusion_steps"] = global_step
        result.metadata["num_blocks"] = num_blocks
        return result if config.return_dict else x

    def _select_transfer(self, result, x0, x0_top2, confidence, num_transfer_tokens,
                         token_queues, prompt_lens, config, step, block, block_step):
        # Select the highest-confidence positions for token transfer in each batch.
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j])).indices
            transfer_index[j, select_index] = True
        return transfer_index

    @staticmethod
    def _queue_would_repeat(queue, token_id, repeat_window):
        # Check whether adding the token would create a repeated-token window.
        values = queue[-(repeat_window - 1):] + [token_id]
        return len(values) == repeat_window and len(set(values)) == 1

    @staticmethod
    def _push_queue(queue, token_id, repeat_window, reset=False):
        # Optionally reset the queue, append the new token, and maintain its window size.
        if reset:
            queue.clear()
        queue.append(token_id)
        del queue[:max(0, len(queue) - repeat_window + 1)]

    def _record_queue_choice(self, result, *, step, block, block_step, batch_id,
                             action, primary_pos, selected_pos, proposed_token,
                             committed_token, queue_before, queue_after):
        
        # Record the queue-based sampling decision and its before/after states.
        result.token_queue_records.append({
            "step": step, "block": block, "block_step": block_step,
            "batch_id": batch_id, "sampler": self.queue_mode,
            "action": action, "primary_pos": primary_pos,
            "selected_pos": selected_pos,
            "proposed_token": proposed_token, "committed_token": committed_token,
            "queue_before": queue_before, "queue_after": queue_after,
        })

    def _record_step(self, result, draft_x0, x0, transfer_index, confidence,
                     model_confidence, committed_confidence, mask_index, prompt_lens,
                     config, step, b, i):
        
        # Record information about transferred tokens for the current sampling step.
        group_size = config.digits_per_value + 1
        for j, p_len in enumerate(prompt_lens):
            positions = torch.where(transfer_index[j])[0].tolist()
            if self.analysis.get("save_unmasked_positions", False):
                for abs_pos in positions:
                    rel_pos = abs_pos - p_len
                    slot = rel_pos % group_size
                    digit_type = ("H", "T", "O", "comma")[slot] if group_size == 4 else str(slot)
                    result.csv_records.append({
                        "block": b, "step": step, "block_step": i, "batch_id": j,
                        "canvas_pos": abs_pos, "relative_pos": rel_pos,
                        "digit_type": digit_type, "token_id": x0[j, abs_pos].item(),
                        "confidence": confidence[j, abs_pos].item(),
                        "model_confidence": committed_confidence[j, abs_pos].item(),
                    })
            # Apply optional lookahead analysis settings.
            lookahead = self.analysis.get("lookahead", {})
            
            slot_convergence = self.analysis.get("slot_convergence", {})
            if (self.analysis.get("save_trajectory", False)
                    or lookahead.get("enabled", False)
                    or slot_convergence.get("enabled", False)):
                pred_ids = draft_x0[j, p_len:p_len + config.max_new_tokens]
                commit_ids = x0[j, p_len:p_len + config.max_new_tokens]
                pred_confidence = model_confidence[j, p_len:p_len + config.max_new_tokens]
                masked_before = mask_index[j, p_len:p_len + config.max_new_tokens]
                masked_after = masked_before & ~transfer_index[j, p_len:p_len + config.max_new_tokens]
                
                
                result.lookahead_token_records.append({
                    "step": step, "block": b, "block_step": i, "batch_id": j,
                    "token_ids": self.tokenizer.decode(pred_ids, skip_special_tokens=True),
                    "pred_token_ids": pred_ids.cpu().tolist(),
                    "draft_token_ids": pred_ids.cpu().tolist(),
                    "commit_token_ids": commit_ids.cpu().tolist(),
                    "draft_confidence": pred_confidence.float().cpu().tolist(),
                    "masked_token_mask_before": masked_before.cpu().tolist(),
                    "masked_token_mask_after": masked_after.cpu().tolist(),
                    "transfer_index": [[j, pos] for pos in positions],
                    "relative_transfer_index": [pos - p_len for pos in positions],
                    "chosen_token": x0[j, positions].cpu().tolist(),
                    "chosen_confidence": committed_confidence[j, positions].float().cpu().tolist(),
                })

    def diffusion_logit_process(self, hidden_states, transfer_index, x0, i, block,
                               branch="full", batch_offset=0):
        transformer = self.model.model.transformer
        ln_f = transformer["ln_f"]
        output_head = self.model.get_output_embeddings()
        device = output_head.weight.device
        records = []
        for j, abs_pos in zip(*[idx.tolist() for idx in torch.where(transfer_index)]):
            layer_ids_at_idx, layer_probs_at_idx = [], []
            for l_i, layer_tensor in enumerate(hidden_states):
                v = layer_tensor[j + batch_offset, abs_pos].to(device)
                if l_i < len(hidden_states) - 1:
                    v = ln_f(v.to(next(ln_f.parameters()).device)).to(device)
                layer_logits = F.linear(v, output_head.weight, getattr(output_head, "bias", None))
                if self.model.config.scale_logits:
                    layer_logits = layer_logits / math.sqrt(self.model.config.d_model)
                topk_probs, topk_ids = torch.softmax(layer_logits.float(), -1).topk(5)
                layer_ids_at_idx.append(topk_ids.cpu().tolist())
                layer_probs_at_idx.append(topk_probs.cpu().tolist())
            records.append({
                "step": i, "block": block, "batch_id": j, "branch": branch,
                "abs_pos": abs_pos, "birth_token_id": x0[j, abs_pos].item(),
                "layer_ids": layer_ids_at_idx, "layer_probs": layer_probs_at_idx,
            })
        return records


class MDLMLensSampler(MDLMSampler):
    """Preserve the original class name. Analysis options are controlled by `analysis`,
    while sharing the same numerical sampling implementation."""


class MDLMtruncateLensSampler(MDLMLensSampler):
    """When the same token is about to be committed for `repeat_window`
    consecutive times, use Top-2 at the original position instead."""

    queue_mode = "truncate"

    def _select_transfer(self, result, x0, x0_top2, confidence, num_transfer_tokens,
                         token_queues, prompt_lens, config, step, block, block_step):
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            positions = torch.topk(confidence[j], k=int(num_transfer_tokens[j])).indices.tolist()
            for pos in positions:
                queue = token_queues[j]
                queue_before = list(queue)
                proposed_token = int(x0[j, pos])
                action = "keep_top1"
                committed_token = proposed_token
                if self._queue_would_repeat(queue, proposed_token, config.queue_repeat_window):
                    committed_token = int(x0_top2[j, pos])
                    x0[j, pos] = committed_token
                    action = "replace_with_top2"
                    self._push_queue(queue, committed_token, config.queue_repeat_window, reset=True)
                else:
                    self._push_queue(queue, committed_token, config.queue_repeat_window)
                transfer_index[j, pos] = True
                self._record_queue_choice(
                    result, step=step, block=block, block_step=block_step, batch_id=j,
                    action=action, primary_pos=pos, selected_pos=pos,
                    proposed_token=proposed_token, committed_token=committed_token,
                    queue_before=queue_before, queue_after=list(queue),
                )
        return transfer_index


class MDLMswitchLensSampler(MDLMLensSampler):
    """When repetition occurs, search for a safe position in a different
    slot, following confidence order."""

    queue_mode = "switch"

    def _select_transfer(self, result, x0, x0_top2, confidence, num_transfer_tokens,
                         token_queues, prompt_lens, config, step, block, block_step):
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        group_size = config.digits_per_value + 1
        for j in range(confidence.shape[0]):
            valid_positions = torch.where(torch.isfinite(confidence[j]))[0]
            sorted_positions = valid_positions[torch.argsort(confidence[j, valid_positions], descending=True)].tolist()
            remaining = list(sorted_positions)
            for _ in range(min(int(num_transfer_tokens[j]), len(remaining))):
                primary_pos = remaining[0]
                selected_pos = None
                for pos in remaining:
                    proposed_token = int(x0[j, pos])
                    if self._queue_would_repeat(token_queues[j], proposed_token,
                                                config.queue_repeat_window):
                        continue
                    if (pos != primary_pos and config.avoid_same_slot_on_reroute
                            and abs(pos - primary_pos) % group_size == 0):
                        continue
                    selected_pos = pos
                    break

                queue_before = list(token_queues[j])
                if selected_pos is None:
                    selected_pos = remaining[-1]
                    action = "fallback_lowest_confidence"
                elif selected_pos == primary_pos:
                    action = "keep_primary"
                else:
                    action = "reroute"

                proposed_token = int(x0[j, selected_pos])
                committed_token = proposed_token
                if action == "reroute" and config.use_second_choice_on_reroute:
                    committed_token = int(x0_top2[j, selected_pos])
                    x0[j, selected_pos] = committed_token
                    action = "reroute_with_top2"

                self._push_queue(
                    token_queues[j], committed_token, config.queue_repeat_window,
                    reset=action == "fallback_lowest_confidence",
                )
                transfer_index[j, selected_pos] = True
                remaining.remove(selected_pos)
                self._record_queue_choice(
                    result, step=step, block=block, block_step=block_step, batch_id=j,
                    action=action, primary_pos=primary_pos, selected_pos=selected_pos,
                    proposed_token=proposed_token, committed_token=committed_token,
                    queue_before=queue_before, queue_after=list(token_queues[j]),
                )
        return transfer_index


SAMPLERS = {
    "MDLMSampler": MDLMSampler,
    "MDLMLensSampler": MDLMLensSampler,
    "MDLMtruncateLensSampler": MDLMtruncateLensSampler,
    "MDLMswitchLensSampler": MDLMswitchLensSampler,
}
