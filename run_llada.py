import argparse
import random
from copy import deepcopy

import numpy as np
import torch

from time_series.experiment_config import load_config
from time_series.monash import load_experiment_data
from time_series.llmtime import get_llada_predictions_data
from time_series.llada_lens import load_model
from time_series.result_writer import ResultWriter


def set_seed(seed=77):
    """
    remark: we provide a way to set seed. But but the experiments do not
        actually use it, since we want to learn the repetiton in a dynamic way
        instead of depending on fix seeds 
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config, model=None, tokenizer=None):
    """
    The entrance of the code
     Run experiments across configured datasets and 
    generate forecasts, and save predictions and analysis results
    """
    
    datasets = load_experiment_data(config["data"])
    
    if model is None:
        model, tokenizer = load_model(config["model"])
        
    output_dirs = []
    seeds = config["seeds"] if config["seeds"] is not None else [None]
    
    for seed in seeds:
        
        for dsname, (train_all, test_all) in datasets.items():
            if seed is not None:
                set_seed(seed)
                
            data_config = config["data"]
            series_ids = data_config["series_indices"]
            
            if series_ids is None:
                series_ids = list(range(len(train_all)))
            elif len(series_ids) == 2 and series_ids[1] is None:
                series_ids = list(range(series_ids[0], len(train_all)))
                
            max_history_len = data_config["max_history_len"]
            forecast_horizon = data_config["forecast_horizon"]
            train = [train_all[i][-max_history_len:] for i in series_ids]
            
            test = []
            for i in series_ids:
                max_test_len = len(test_all[i])  # if we don't set the maxforecast ratio
                
                if data_config["max_forecast_ratio"] is not None:
                    max_test_len = min(
                        max_test_len,
                        max(1, int(len(train_all[i]) * data_config["max_forecast_ratio"])),
                    )
                    
                if forecast_horizon is not None:
                    max_test_len = min(max_test_len, forecast_horizon)
                test.append(test_all[i][:max_test_len])

            run_config = deepcopy(config)
            run_config["seeds"] = None if seed is None else [seed]
            run_config["data"]["datasets_to_run"] = [dsname]
            run_config["data"]["series_indices"] = list(series_ids)
            
            writer = ResultWriter(run_config, dsname, seed)
            
            llama_hypers = config["forecast"]
            seed_label = seed if seed is not None else "unseeded"
            print(f"Starting {dsname} | {config['experiment_name']} | seed={seed_label}")
            
            
            preds = get_llada_predictions_data(
                train, test, model=model, tokenizer=tokenizer,
                sampling_config=config["sampling"], analysis_config=config["analysis"],
                svd_config=config["svd"], context_length=config["model"]["context_length"],
                series_ids=series_ids, on_sample=writer.save_sampling_result,
                **llama_hypers,
            )
            
            writer.save_predictions(preds, train, test)
            output_dirs.append(writer.output_dir)
            print(f"Finished {dsname} | MAE={preds['mae']:.6f} | R_excess={preds['R_excess']:.6f}")
            print(f"Results: {writer.output_dir}")
            
    return output_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    model, tokenizer, model_config = None, None, None
    for config_path in args.config:
        config = load_config(config_path, args.set)
        if args.print_config:
            import yaml
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            continue
        if config["model"] != model_config:
            # 多个实验使用相同模型时只加载一次。
            model, tokenizer = load_model(config["model"])
            model_config = config["model"]
        run_experiment(config, model, tokenizer)


if __name__ == "__main__":
    main()
