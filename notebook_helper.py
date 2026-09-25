"""Configuration, data loading, statistics and plotting helpers shared by the
figure notebooks (plot_*.ipynb).

The data is the Hugging Face dataset repo, downloaded into DATA_ROOT
(default: ./data, override with the METACOG_DATA environment variable):
    models/<project>/unsloth_full/<run>/evals/<eval dataset>.pt          final model
    models/<project>/unsloth_full/<run>/checkpoint-*/evals/...            development runs
    datasets/processed_with_performance/<dataset>/<model>_quantized_dataset
    datasets/processed_with_baselineguess/<dataset>/<model>_dataset
    datasets/embedded/<dataset>.pt
    <project>/outputs/<run>/events.out.tfevents.*                        training logs
"""
import gc
import os
import re

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage, stats
from tqdm import tqdm

# ================================================================= configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("METACOG_DATA", os.path.join(REPO_ROOT, "data"))
PROJECT = "default"
RUNS_PATH = os.path.join(DATA_ROOT, "models", PROJECT, "unsloth_full")
DATASET_PATH = os.path.join(DATA_ROOT, "datasets")
LOG_PATH = os.path.join(DATA_ROOT, PROJECT, "outputs")
FIG_PATH = os.path.join(REPO_ROOT, "fig")

# Training epochs per dataset. Runs are selected by their epoch count: the main
# experiment (final models, N_RUNS runs) and the development experiment (every
# checkpoint evaluated, N_RUNS_DEV runs).
EPOCHS = {
    "math4_categoriesbalanced": 1,
    "math10_categoriesbalanced": 1,
    "mmlupro4_categoriesbalanced": 8,
    "mmlupro10_categoriesbalanced": 8,
    "medmcqa4_categoriesbalanced": 1,
}
EPOCHS_DEV = {**EPOCHS, "medmcqa4_categoriesbalanced": 8}
N_RUNS = 5
N_RUNS_DEV = 3

DATASETS = list(EPOCHS)
SHORT_LABELS = {d: d.replace("_categoriesbalanced", "") for d in DATASETS}
N_OPTIONS = {                        # answer options per question
    "math4_categoriesbalanced": 4,
    "math10_categoriesbalanced": 10,
    "mmlupro4_categoriesbalanced": 4,
    "mmlupro10_categoriesbalanced": 10,
    "medmcqa4_categoriesbalanced": 4,
}

MODELS = [
    "Llama-2-7b-chat",
    "Llama-3.2-3B-Instruct",
    "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B-Base",
    "Qwen3.5-9B",
    "phi-4",
    "Mistral-7B-Instruct-v0.3",
    "Ministral-3-3B-Instruct-2512",
    "Ministral-3-8B-Instruct-2512",
]

PROBES = ["mid", "end"]              # probe on the middle / last decoder layer
LORA_RANK = {"mid": 16, "end": 8}

# ==================================================================== statistics
def nan_pearsonr(x, y):
    """Pearson r ignoring positions where either array is NaN (NaN if < 2 points)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    return stats.pearsonr(x[mask], y[mask]).statistic


def sem(a, axis=0):
    """Standard error of the mean ignoring NaN, NaN where undefined."""
    out = np.ma.filled(stats.sem(a, axis=axis, nan_policy="omit"), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def consistency(probs, n_options):
    """1 − normalised entropy of the answer distribution (1 = always the same answer)."""
    p = np.asarray(probs, dtype=float)
    entropy = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    return 1 - entropy / np.log(n_options)

# ======================================================================= loading
def find_runs(dev=False):
    """Finished run folders per (dataset, model, probe), sorted by name.

    Run folder names: <dataset>_<epochs>_<lr>_<wd>_<bs>_<rank>_<warmup>_probe-<p>_<id>.
    """
    epochs = EPOCHS_DEV if dev else EPOCHS
    runs = {d: {m: {p: [] for p in PROBES} for m in MODELS} for d in epochs}
    for name in sorted(os.listdir(RUNS_PATH)):
        if not os.path.exists(os.path.join(RUNS_PATH, name, "probe_meta.json")):
            continue                                    # unfinished training
        dataset, split, model, n_epochs, _, _, _, rank, _, probe, _ = name.split("_")
        dataset, probe = f"{dataset}_{split}", probe.split("-")[1]
        if (dataset in epochs and model in MODELS and probe in PROBES
                and epochs[dataset] == int(n_epochs) and LORA_RANK[probe] == int(rank)):
            runs[dataset][model][probe].append(name)
    return runs


def checkpoint_names(run):
    """checkpoint-* folders of a run, in training order."""
    names = [k for k in os.listdir(os.path.join(RUNS_PATH, run)) if k.startswith("checkpoint-")]
    return sorted(names, key=lambda k: int(k.split("-")[1]))


def load_eval(run, dataset, checkpoint=None, split=None):
    """{split: {"predictions", "accuracies", "consistency"}} of one eval file, or None.

    checkpoint=None is the final model (full splits); intermediate checkpoints
    hold a 1000-question subsample ('train_subset', 'test').
    """
    path = os.path.join(RUNS_PATH, run, *([checkpoint] if checkpoint else []),
                        "evals", dataset + ".pt")
    if not os.path.exists(path):
        return None
    splits = torch.load(path, weights_only=False)["splits"]
    if split is not None:
        if split not in splits:
            return None
        splits = {split: splits[split]}
    keep = ("predictions", "accuracies", "consistency")
    return {name: {k: v for k, v in entry.items() if k in keep} for name, entry in splits.items()}


def load_run_checkpoints(run, dataset):
    """Every checkpoint of a run (None where not evaluated), final model last.

    Empty if no intermediate checkpoint was evaluated (--last_only runs).
    """
    ckpts = [load_eval(run, dataset, c) for c in checkpoint_names(run)]
    if all(c is None for c in ckpts):
        return []
    final = load_eval(run, dataset)
    return ckpts + ([final] if final is not None else [])


def load_trained(split=None, eval_dataset=None, dev=False):
    """Eval results of the trained probes: data[train dataset][model][probe] -> runs.

    A run is the final model's load_eval() output or, with dev=True, the list
    of its checkpoints (load_run_checkpoints). eval_dataset defaults to the
    training dataset. At most N_RUNS (N_RUNS_DEV) evaluated runs are kept.
    """
    runs = find_runs(dev)
    n_max = N_RUNS_DEV if dev else N_RUNS
    data = {d: {m: {p: [] for p in PROBES} for m in MODELS} for d in runs}
    for d in tqdm(runs, desc=f"eval on {SHORT_LABELS[eval_dataset]}" if eval_dataset else "load"):
        for m in MODELS:
            for p in PROBES:
                for run in runs[d][m][p]:
                    if len(data[d][m][p]) == n_max:
                        break
                    e = eval_dataset or d
                    result = load_run_checkpoints(run, e) if dev else load_eval(run, e, split=split)
                    if result:
                        data[d][m][p].append(result)
    n_short = sum(len(data[d][m][p]) < n_max for d in data for m in MODELS for p in PROBES)
    if n_short:
        print(f"{n_short} (dataset, model, probe) configurations with fewer than {n_max} runs")
    return data


def load_trained_cross(split=None):
    """Cross-dataset evals: data[eval dataset][train dataset][model][probe] -> runs."""
    return {e: load_trained(split, eval_dataset=e) for e in DATASETS}


def load_performance(split="test"):
    """Per-question accuracy and consistency of the base models: acc[d][m], cons[d][m]."""
    acc = {d: {m: np.nan for m in MODELS} for d in DATASETS}
    cons = {d: {m: np.nan for m in MODELS} for d in DATASETS}
    for d in DATASETS:
        for m in MODELS:
            path = os.path.join(DATASET_PATH, "processed_with_performance", d,
                                f"{m}_quantized_dataset")
            if not os.path.exists(path):
                print("missing:", path)
                continue
            perf = torch.load(path, weights_only=False)[split]
            acc[d][m] = perf["accuracies"]
            cons[d][m] = consistency(perf["outputs"], N_OPTIONS[d])
    return acc, cons


def load_baseline():
    """Verbalised confidence of the base models (test split): base[d][m], NaN if unparsed."""
    base = {d: {m: None for m in MODELS} for d in DATASETS}
    for d in DATASETS:
        for m in MODELS:
            path = os.path.join(DATASET_PATH, "processed_with_baselineguess", d, f"{m}_dataset")
            if not os.path.exists(path):
                print("missing:", path)
                continue
            preds = torch.load(path, weights_only=True)["predictions"]
            base[d][m] = [np.nan if p is None else float(p) for p in preds]
    return base

# ------------------------------------------ embeddings (distance figures)
N_NEAREST = 100     # distance of a question to a dataset = mean over its N nearest questions


def load_embeddings(dataset, split):
    """L2-normalised float32 question embeddings of one split."""
    path = os.path.join(DATASET_PATH, "embedded", dataset + ".pt")
    try:
        obj = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    except Exception:
        obj = torch.load(path, weights_only=False, map_location="cpu")
    x = obj[split]["embeddings"].to(torch.float32)
    del obj
    gc.collect()
    return torch.nn.functional.normalize(x, dim=1)


def knn_distance(test, train, n_nearest=N_NEAREST, device="cpu",
                 chunk_test=8192, chunk_train=1024):
    """Per test row, mean cosine distance to its n_nearest closest train rows (chunked)."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    k = max(1, min(n_nearest, len(train)))
    out = torch.empty(len(test))
    for i in range(0, len(test), chunk_test):
        t = test[i:i + chunk_test].to(device, dtype)
        best = torch.full((len(t), k), float("inf"), device=device, dtype=torch.float32)
        for j in range(0, len(train), chunk_train):
            d = 1.0 - (t @ train[j:j + chunk_train].to(device, dtype).T).float()
            best = torch.topk(torch.cat([best, d], dim=1), k, dim=1, largest=False,
                              sorted=False).values
        out[i:i + chunk_test] = best.mean(dim=1).cpu()
    return out.numpy()


def load_embedding_distances(split="test"):
    """dist[a][e]: per question of the `split` split of e, distance to the train split of a."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dist = {a: {} for a in DATASETS}
    for a in DATASETS:
        train = load_embeddings(a, "train")
        for e in tqdm(DATASETS, desc=f"distances to {SHORT_LABELS[a]}"):
            dist[a][e] = knn_distance(load_embeddings(e, split), train, device=device)
        del train
        gc.collect()
    return dist


def load_question_errors(emb_dist, probes=("end",), split="test"):
    """Per-question |conf − accuracy| and |conf − consistency| of the final models,
    averaged over every run (all models) trained on a and evaluated on e:
    err_acc[a][e], err_cons[a][e]."""
    runs = find_runs()
    sums = {a: {} for a in DATASETS}
    counts = {a: {} for a in DATASETS}
    n_missing = 0
    for a in tqdm(DATASETS, desc="question errors"):
        for m in MODELS:
            for p in probes:
                for run in runs[a][m][p]:
                    for e in DATASETS:
                        ev = load_eval(run, e, split=split)
                        # only the full split is aligned with the embeddings
                        if ev is None or len(ev[split]["predictions"]) != len(emb_dist[a][e]):
                            n_missing += 1
                            continue
                        t = ev[split]
                        conf = np.asarray(t["predictions"], dtype=float)
                        err = np.stack([np.abs(conf - np.asarray(t["accuracies"], dtype=float)),
                                        np.abs(conf - np.asarray(t["consistency"], dtype=float))])
                        sums[a][e] = sums[a][e] + err if e in sums[a] else err
                        counts[a][e] = counts[a].get(e, 0) + 1
    if n_missing:
        print(f"{n_missing} (run, eval dataset) pairs without a full-split eval")
    err_acc = {a: {e: sums[a][e][0] / counts[a][e] for e in sums[a]} for a in DATASETS}
    err_cons = {a: {e: sums[a][e][1] / counts[a][e] for e in sums[a]} for a in DATASETS}
    return err_acc, err_cons

# ------------------------------------------------- training logs (appendix)
# Scalars logged by the HF trainer of fit_unsloth.py: 'train/loss', and on the
# two eval subsets 'eval/sub_train_*', 'eval/sub_test_*'
TB_TAGS = {
    "train_loss": r"train/loss",
    "eval_train_loss": r"eval/sub_train_loss",
    "eval_test_loss": r"eval/sub_test_loss",
    "test_pass005": r"eval/sub_test_0\.05",
    "test_corr_pred_acc": r"eval/sub_test_corr_pred_acc",
    "test_corr_pred_cons": r"eval/sub_test_corr_pred_cons",
    "test_corr_cons_acc": r"eval/sub_test_corr_cons_acc",
    "train_corr_pred_acc": r"eval/sub_train_corr_pred_acc",
    "train_corr_pred_cons": r"eval/sub_train_corr_pred_cons",
}


def load_tensorboard_log(run):
    """{tag: (steps, values)} of one run, or None if it has no event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = os.path.join(LOG_PATH, run)
    files = [f for f in os.listdir(path) if "tfevents" in f] if os.path.isdir(path) else []
    assert len(files) <= 1, f"more than one event file in {path}"
    if not files:
        return None
    ea = EventAccumulator(path, size_guidance={"scalars": 0})    # 0 = keep everything
    ea.Reload()
    scalars = {}
    for tag in ea.Tags()["scalars"]:
        last = {e.step: e.value for e in ea.Scalars(tag)}          # last value per step (restarts)
        steps = np.array(sorted(last), dtype=float)
        scalars[tag] = (steps, np.array([last[s] for s in sorted(last)], dtype=float))
    return scalars


def get_series(scalars, key):
    """(steps, values) of TB_TAGS[key] in one run's scalars, or None."""
    return next((v for tag, v in scalars.items() if re.fullmatch(TB_TAGS[key], tag)), None)


def load_training_logs():
    """Training logs of the main runs: logs[d][m][p] -> [(run name, scalars), ...]."""
    runs = find_runs()
    logs = {d: {m: {p: [] for p in PROBES} for m in MODELS} for d in DATASETS}
    n_missing = 0
    for d in tqdm(DATASETS, desc="training logs"):
        for m in MODELS:
            for p in PROBES:
                for run in runs[d][m][p][:N_RUNS]:
                    scalars = load_tensorboard_log(run)
                    if scalars:
                        logs[d][m][p].append((run, scalars))
                    else:
                        n_missing += 1
    if n_missing:
        print(f"{n_missing} runs without an event file")
    return logs

# ====================================================================== plotting
COLORS = {
    "navy": "#003049",
    "steel": "#669BBC",
    "red": "#D62828",
    "orange": "#F77F00",
    "amber": "#FCBF49",
    # roles, identical in every figure
    "baseline": "#669BBC",
    "end": "#D62828",
    "mid": "#F77F00",
    "points": "#669BBC",
}
PROBE_LABELS = {"end": "trained (end)", "mid": "trained (mid)"}
GRID_KW = dict(color="0.9", lw=0.6)
ERR_KW = dict(ecolor="black", elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=4)
# Scatter clouds coloured by density: pale blue (sparse) -> navy (dense)
POINT_CMAP = LinearSegmentedColormap.from_list("points", ["#C9DDEA", COLORS["steel"], COLORS["navy"]])
# White halo keeping fit lines readable over dense clouds
FIT_OUTLINE = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]


def set_style(fs, lw=1.0):
    """Global font sizes (dict with base/title/label/tick/legend) and line widths.

    Full-width figures are drawn on a ~17 in canvas and printed ~7 in wide,
    hence the large sizes. TrueType fonts keep the PDF text editable.
    """
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": fs["base"],
        "axes.titlesize": fs["title"], "axes.labelsize": fs["label"],
        "xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"],
        "legend.fontsize": fs["legend"],
        "axes.linewidth": lw, "xtick.major.width": lw, "ytick.major.width": lw,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_ax(ax, grid_axis="y", hide=("top", "right"), **grid_kw):
    """Open frame and a light grid behind the data."""
    ax.spines[list(hide)].set_visible(False)
    ax.grid(axis=grid_axis, **{**GRID_KW, **grid_kw})
    ax.set_axisbelow(True)


def bar_offsets(n_series, group_width):
    """Offsets of n_series bars centred in a group, and the slot width of one bar."""
    slot = group_width / n_series
    return (np.arange(n_series) - (n_series - 1) / 2) * slot, slot


def save_fig(fig, name, dpi=300):
    """fig/<name>.pdf (vector, for the paper) and fig/<name>.png."""
    os.makedirs(FIG_PATH, exist_ok=True)
    fig.savefig(os.path.join(FIG_PATH, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_PATH, name + ".png"), dpi=dpi, bbox_inches="tight")


def point_density(x, y, bins=80, smooth=1.5):
    """Smoothed 2D-histogram density at each point, scaled to [0, 1]."""
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    H = ndimage.gaussian_filter(H, smooth)
    ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, bins - 1)
    d = H[ix, iy]
    return d / d.max()


def density_scatter(ax, x, y, s, alpha, **kwargs):
    """Points coloured by local density, densest drawn last."""
    dens = point_density(x, y)
    order = np.argsort(dens)
    return ax.scatter(x[order], y[order], c=np.sqrt(dens[order]), cmap=POINT_CMAP,
                      vmin=0, vmax=1, s=s, alpha=alpha, edgecolors="none",
                      rasterized=True, zorder=1, **kwargs)


def fit_poly_both(x, y):
    """Quadratic fits y = f(x) and x = f(y); returns (coeffs, transposed, r2) of the best."""
    x, y = np.asarray(x), np.asarray(y)
    fits = []
    for u, v, transposed in ((x, y, False), (y, x, True)):
        c = np.polyfit(u, v, deg=2)
        r2 = 1 - np.sum((v - np.poly1d(c)(u)) ** 2) / np.sum((v - v.mean()) ** 2)
        fits.append((c, transposed, r2))
    return fits[0] if fits[0][2] >= fits[1][2] else fits[1]


def quadratic_fit(ax, x, y, color, lw, **kwargs):
    """Draw the best quadratic fit (possibly x = f(y)) without rescaling the axes; returns R²."""
    coeffs, transposed, r2 = fit_poly_both(x, y)
    f = np.poly1d(coeffs)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    t = np.linspace(np.min(y if transposed else x), np.max(y if transposed else x), 200)
    ax.plot(*((f(t), t) if transposed else (t, f(t))), color=color, lw=lw, zorder=3,
            path_effects=FIT_OUTLINE, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return r2
