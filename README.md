# MetaCog

Code for the paper LLMs learn different forms of metacognition when trained to predict their own accuracy (https://arxiv.org/abs/2609.33886).

This work trains LoRA adapters and a linear head, attached to the middle (`mid`) or last (`end`) decoder layer. These are trained to predict the base model's accuracy on each question. We then ask what the trained confidence actually tracks: the model's **true accuracy**, or its **output consistency** (how concentrated its answer distribution is), in and out of the training distribution. The results suggest that, on questions far from the training distribution, trained probes linearly track consistency better than true accuracy while on questions close to the training distribution they linearly track true accuracy better.

This paper replicates results across 10 base models (Llama, Qwen, Mistral, Phi), 5 multiple-choice datasets (MATH and MMLU-Pro with 4 and 10 options, MedMCQA) with 5 training runs per configuration.

NOTE: the trained models are not finished uploading yet. They should be available soon1

## Setup

```bash
pip install -r requirements.txt
```
(uncomment the last lines for also installing training libraries)

Download the data (evaluation results, per-question performance, embeddings, training logs) into `data/`:

```bash
hf auth login
hf download nyax/MetaCogData --repo-type dataset --local-dir data
```

Trained probes (only needed to re-run evaluations) go into `trainedmodels/`:

```bash
hf download nyax/MetaCogModels --local-dir trainedmodels
```

## Reproducing the figures

Each notebook loads `data/` through `notebook_helper.py` and writes its figures to `fig/` (PDF + PNG). Set `METACOG_DATA` to use another data folder.

| Notebook | Paper figures (`fig/figNN_*`) |
|---|---|
| `plot_performances.ipynb` | 2, 6: accuracy vs consistency under topic shift · 11, 12: confidence before/after training · 13–16: scatter grids |
| `plot_cross_eval.ipynb` | 3, 7: probes evaluated on every dataset · 17–19: transfer heatmaps |
| `plot_distance.ipynb` | 4, 9: Δe vs embedding distance to the training data |
| `plot_development.ipynb` | 5, 8: Δr along training, every checkpoint |
| `plot_training.ipynb` | 10: training loss, plus extra diagnostics (`extra_*`, end-of-training table) |

Figs 2–5 use the end probe; Figs 6–9 repeat them with the mid probe.

`plot_distance.ipynb` uses DM Sans if its `.ttf` files are in `fonts/` (falls back to DejaVu Sans).

## Training pipeline

The scripts below produced `data/`. Run them from the repository root: they read base models from `basemodels/unsloth/<model>` (the Unsloth releases on the Hub), datasets from `data/datasets/`, and write probes (with their evaluations) to `trainedmodels/models/` and training logs to `data/<project>/outputs/`. Each `*.sh` is a generic Slurm launcher for the matching `*.py` (usage in its header; adapt the `#SBATCH` lines to your cluster).

1. `make_datasets.ipynb`, `make_categoriesbalanced.ipynb`: multiple-choice datasets with train/test splits balanced over topic categories.
2. `compute_performance.py`: per-question accuracy and answer distribution of each base model.
3. `compute_baseline.py`: verbalised-confidence baseline.
4. `compute_embeddings.py`: question embeddings (Qwen3-Embedding-4B), used for the distance figure.
5. `fit_unsloth.py`: trains a probe (`--probe mid|end`), with checkpoints and tensorboard logs.
6. `compute_trained_performance.py`: evaluates a trained run on any dataset (`--eval_dataset all`), at every checkpoint or the final model only (`--last_only`).

## Layout

```
data/            downloaded data (see above)
trainedmodels/   downloaded probes
basemodels/      base LLMs, for training
fig/             generated figures
notebook_helper.py   paths, configuration, loaders and plotting helpers of the notebooks
```
