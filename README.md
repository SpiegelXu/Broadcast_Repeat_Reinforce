# Broadcast_Repeat_Reinforce

**Code for mechanistic analysis of repetition collapse in diffusion language models (DLMs).**

This repository accompanies our research on attention dynamics behind repetition collapse in diffusion language models, studied through digit-wise time-series forecasting.

## Overview

We investigate how structured attention patterns and layerwise gain dynamics contribute to repetition collapse. The project provides code for running LLaDA-based forecasting experiments and analyzing the associated attention, gain, and hidden-state dynamics.

## Key Components

- **Time-series forecasting:** LLaDA-based inference and look-ahead experiments.
- **Repetition analysis:** Measurements of repetition behavior across decoding steps and digit positions.
- **Mechanistic analysis:** Attention-anchor mass, layerwise gain, and hidden-state convergence.
- **Inference-time interventions:** SVD-based attention analysis and hidden-state/CFG-related experiments.

## Repository Structure

- `run_llada.py` — Main experiment entry point.
- `configs/` — Experiment configurations.
- `src/` — Forecasting and model-related implementation.
- `analysis/` — Scripts for analyzing experimental results.


## Getting Started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment using a configuration file:

```bash
python run_llada.py --config configs/base.yaml
```

To inspect the available configuration options and experiment settings, see `configs/` and `README.md` documentation as the project evolves. Depending on the experiment, you may need to configure model paths, datasets, and hardware settings before running.

## License

This project is released under the MIT License, subject to the applicable licenses of third-party code and dependencies. See [LICENSE](LICENSE) for details.
