from dataclasses import asdict

import torch
from transformers import AutoTokenizer

from .model_llada import LLaDAModelLM
from .mdlm import MDLMSamplerConfig, SAMPLERS
from .svd_hijack import hijack_llada_attention
from .diagnostics import run_analysis
from .heatmap_analysis import select_indices





def _merge_capture_selection(first, second):
    
    """Merge capture selections for normal analysis and heatmap to avoid
    redundant capture when their layer or step selections overlap.
    """
    if first is None or second is None:
        return None
    return sorted(set(first) | set(second))


def load_model(model_config):
    local_path = model_config["local_path"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, revision=model_config["revision"],
        local_files_only=model_config["local_files_only"],
    )
    
    tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    tokenizer.eot_token = "<|eot_id|>"
    
    model = LLaDAModelLM.from_pretrained(
        local_path, revision=model_config["revision"],
        torch_dtype=getattr(torch, model_config["torch_dtype"]),
        device_map=model_config["device_map"],
        local_files_only=model_config["local_files_only"],
    ).eval()
    
    model.config.mask_token_id = tokenizer.mask_token_id
    
    return model, tokenizer


def tokenize_fn_llada(str, tokenizer):
    return tokenizer(str)


def prepare_input(input_str, steps, settings, tokenizer, sampling_config):
    
    input_ids = tokenizer(input_str)["input_ids"]
    avg_tokens_per_step = len(input_ids) / len(input_str.split(settings.time_sep))
    
    if sampling_config["max_new_tokens"] is not None:
        max_new_tokens = sampling_config["max_new_tokens"]
    elif sampling_config["length_mode"] == "fixed_slots":
        max_new_tokens = int(steps * (sampling_config["digits_per_value"] + 1))
    else:
        max_new_tokens = max(1, int(avg_tokens_per_step * steps))
        
    prompt_text = sampling_config["prompt_text"].format(steps=steps)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    bos_id = tokenizer.bos_token_id
    input_ids = [bos_id] + prompt_ids + input_ids
    if sampling_config["cfg_keep_policy"] == "prompt_token_overlap":
        cfg_keep_tokens = [bos_id] + prompt_ids
    else:
        cfg_keep_tokens = sampling_config["cfg_keep_tokens"]
    return input_ids, cfg_keep_tokens, prompt_text, max_new_tokens


def llada_completion_fn(input_str, steps, settings, batch_size=1, num_samples=3, temp=1.0,
                        *, model, tokenizer, sampling_config, analysis_config, svd_config,
                        series_id=0, target=None, scaler=None, on_sample=None):
    input_ids, cfg_keep_tokens, prompt_text, max_new_tokens = prepare_input(
        input_str, steps, settings, tokenizer, sampling_config
    )
    # digital wise seperation
    #good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens_str = list("0123456789")
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    
    assert all(len(tokenizer.encode(token, add_special_tokens=False)) == 1 for token in good_tokens_str)
    bad_tokens = sorted(set(range(model.get_output_embeddings().weight.shape[0])) - set(good_tokens))
    
    # mannually assign the step for one mask per step mode
    if sampling_config["one_step_per_mask"]:
        num_blocks = (max_new_tokens + sampling_config["block_size"] - 1) // sampling_config["block_size"]
        sampler_steps = num_blocks * sampling_config["block_size"]
    else:
        sampler_steps = sampling_config["diffusion_steps"]
        
    token_queue = sampling_config["token_queue"]
    sampler_config = MDLMSamplerConfig(
        max_new_tokens=max_new_tokens, steps=sampler_steps,
        block_size=sampling_config["block_size"], temperature=temp,
        suppress_tokens=bad_tokens, good_tokens=good_tokens,
        remasking=sampling_config["remasking"],
        stochastic_transfer=sampling_config["stochastic_transfer"],
        right_shift_logits=sampling_config["right_shift_logits"],
        begin_suppress_tokens=sampling_config["begin_suppress_tokens"],
        cfg_scale=sampling_config["cfg_scale"], cfg_keep_tokens=cfg_keep_tokens,
        cfg_mask_scope=sampling_config["cfg_mask_scope"],
        prefill_commas=sampling_config["prefill_commas"],
        digits_per_value=sampling_config["digits_per_value"],
        comma_id=tokenizer.convert_tokens_to_ids(settings.time_sep),
        queue_repeat_window=token_queue["repeat_window"],   # add this parameter for truncate function
        avoid_same_slot_on_reroute=token_queue["avoid_same_slot_on_reroute"],
        use_second_choice_on_reroute=token_queue["use_second_choice_on_reroute"],
    )
    
    sampler = SAMPLERS[sampling_config["sampler"]](model, tokenizer, analysis_config)
    # get the layers that the heatmap need
    heatmap = analysis_config["attention_heatmap"]
    heatmap_qk = heatmap["enabled"] and heatmap["source"] in {"qk", "both"}   # If we only wanna to draw the heat map, then extract qk is fine
    regular_qkv = any(analysis_config[key] for key in    
                      ["capture_qkv", "save_qkv", "attention_statistics", "ca_g", "beta_S_fisher_split"]) # for analysis, q,k,v are all needded
    need_qkv = regular_qkv or heatmap_qk
    heatmap_layers = select_indices(
        len(model.get_blocks()), heatmap["layers"], heatmap["layer_parity"], heatmap["layer_stride"]
    )
    
    capture_layers = analysis_config["layers"] if regular_qkv else heatmap_layers
    capture_steps = analysis_config["steps"] if regular_qkv else heatmap["steps"]
    if regular_qkv and heatmap_qk:
        # merge the layers to avoid redundant work.
        capture_layers = _merge_capture_selection(analysis_config["layers"], heatmap_layers)
        capture_steps = _merge_capture_selection(analysis_config["steps"], heatmap["steps"])
        
    gen_strs = []
    for sample_start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - sample_start)
        inputs = [input_ids for _ in range(current_batch_size)]
        
        model.configure_capture(enabled=need_qkv, layers=capture_layers,
                                steps=capture_steps, capture_v=analysis_config["save_qkv"])
        
        try:
            cfg_batch_size = current_batch_size if sampler_config.cfg_scale > 0 else None
            with hijack_llada_attention(
                    model, svd_config, cfg_batch_size, heatmap, sample_start) as svd_records:
                outputs = sampler.sample(inputs=inputs, config=sampler_config)
                
            gen_tokens = outputs.sequences[:, len(input_ids):len(input_ids) + max_new_tokens]
            batch_gen_strs = tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            gen_strs.extend(batch_gen_strs)
            
            
            outputs.metadata["sample_start"] = sample_start
            
            tables, tensors = run_analysis(
                model, outputs, analysis_config,
                {"settings": settings, "scaler": scaler, "target": target},
            )
            print(f"finish epoch:", sample_start)
            tables["svd"] = svd_records["records"]
            
            
            for record in tables["svd"]:
                record["prompt_len"] = outputs.metadata["prompt_lens"][record["batch_id"]]
            tensors.update(svd_records["matrices"])
            resolved_sampler = asdict(sampler_config)
            
            
            
            # suppress token can be recounstructed by good_tokens + vocab_size, so
            resolved_sampler.pop("suppress_tokens")
            
            metadata = {**outputs.metadata, "series_id": series_id,
                        "sample_start": sample_start, "batch_size": current_batch_size,
                        "sample_ids": list(range(sample_start, sample_start + current_batch_size)),
                        "forecast_horizon": steps, "input_str": input_str,
                        "target": target,
                        "scaler": None if scaler is None else {"min_": scaler.min_, "q": scaler.q},
                        "prompt_text": prompt_text, "instruction_length": len(input_ids) - len(tokenizer(input_str)["input_ids"]),
                        "cfg_keep_policy": sampling_config["cfg_keep_policy"],
                        "configured_diffusion_steps": sampling_config["diffusion_steps"],
                        "one_step_per_mask": sampling_config["one_step_per_mask"],
                        "sampler_config": resolved_sampler,
                        "vocab_size": model.get_output_embeddings().weight.shape[0],
                        "captured_steps": {block.layer_id: list(block.my_captured_steps) for block in model.get_blocks()},
                        "branch_layout": "full batch, then sparse batch" if cfg_batch_size else "full batch",
                        "completions": batch_gen_strs}
            
            
            
            if on_sample is not None:
                on_sample(outputs, metadata, tables, tensors)
            del outputs, tables, tensors, svd_records
        finally:
            model.disable_qkv_capture()
            model.clear_qkv_capture()
    return gen_strs
