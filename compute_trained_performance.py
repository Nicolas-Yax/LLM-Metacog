import os
# Fix ministral crash
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # don't generate/use compiled modules
os.environ["TORCHDYNAMO_DISABLE"] = "1"       # make every torch.compile a no-op (eager)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import argparse
import tqdm

import numpy as np
import torch
import unsloth
import torch.nn as nn

from peft import PeftModel, load_peft_weights
from peft.utils import set_peft_model_state_dict
from transformers import DataCollatorWithPadding




# ----------------------------------------------------------------------------- Paths
CHECKPOINT_PATH = 'trainedmodels/models'
MODELS_PATH = 'basemodels'
DATA_PATH = 'data/datasets'

# ----------------------------------------------------------------------------- Args
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_name', type=str, required=True,
                    help="Name of the run folder (the long filename created by the training script)")
parser.add_argument('--project', type=str, default="default",
                    help="Project sub-folder the run was saved under (models/{project}/unsloth_full/{checkpoint_name})")
parser.add_argument('--eval_dataset', type=str, required=True,
                    help="Dataset C to evaluate on. Backward compatible: a single name works as before. "
                         "You may also pass a COMMA-SEPARATED list (e.g. 'math4_...,mmlupro4_...') or the "
                         "special value 'all' to evaluate every dataset found under "
                         "processed_with_performance/ that has a {model}_quantized_dataset. All datasets are "
                         "evaluated in one process, so the base model is loaded only once.")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_size', type=float, default=1000,
                    help="Size of the train split to evaluate on (fixed seed => same subset for every run)")
parser.add_argument('--test_size', type=float, default=1000,
                    help="Size of the train split to evaluate on (fixed seed => same subset for every run)")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_length', type=int, default=2048)
parser.add_argument('--overwrite', action='store_true',
                    help="Re-evaluate checkpoints even if a result file already exists")
parser.add_argument("--last_only", action="store_true")

# ============================================================================= Utilities shared with training
# (Kept identical to the training script. Consider moving these to a shared
#  module imported by both scripts so they cannot drift apart.)

def last_token_index(attention_mask):
    """Index of the last real (non-pad) token per example (RIGHT padding)."""
    return attention_mask.sum(dim=1) - 1

def scaled_entropy_consistency(out, eps=1e-12):
    p = np.asarray(out, dtype=np.float64).reshape(-1)
    p = np.clip(p / p.sum(), eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(1.0 - H / np.log(p.shape[0]))

def consistency_array(outputs_list):
    return np.array([scaled_entropy_consistency(o) for o in outputs_list], dtype=np.float64)

def get_decoder_layers_path(model_path):
    if any(i in model_path for i in ("Qwen2.5", "Llama-3", "Llama-2", "Mistral-7B")):
        return "model.layers"
    elif any(i in model_path for i in ("gemma-3", "Ministral-3", "Qwen3.5")):
        return "model.language_model.layers"
    elif any(i in model_path for i in ("granite",)):
        return "model.layers"
    elif any(i in model_path for i in ("phi-4",)):
        return "model.layers"
    else:
        print("model:", model_path)
        assert False, "Undefined decoder-layers path for given model"

def _resolve_layers(model, layers_path):
    *parents, attr = layers_path.split(".")
    obj = model
    for p in parents:
        obj = getattr(obj, p)
    layers = getattr(obj, attr)
    assert isinstance(layers, nn.ModuleList), f"'{layers_path}' is not a ModuleList (got {type(layers).__name__})"
    return obj, attr, layers

def truncate_decoder_to_half(model, layers_path):
    parent, attr, layers = _resolve_layers(model, layers_path)
    n_full = len(layers)
    n_used = n_full // 2
    setattr(parent, attr, nn.ModuleList(list(layers[:n_used])))
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = n_used
    print(f"[MID-PROBE] {layers_path}: {n_full} -> {n_used} (kept bottom half)")
    return n_full, n_used

PREPROMPT = ("You are going to be asked to guess your likelihood to be right on a given question. "
             "Give your answer with the format 'Accuracy prediction: X%' but replace X with your "
             "percentage chance to be right on this question.\nQuestion :")
POSTPROMPT = "Accuracy prediction:"

# ============================================================================= Eval helpers

def compute_metrics(preds, true_acc, cons, eps=1e-12):
    """Same metrics as during training: pass@0.05 + correlations."""
    def _corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.std(a) < eps or np.std(b) < eps:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])
    return {
        '0.05': float(np.mean(np.abs(true_acc - preds) < 0.05)),
        'mse': float(np.mean((true_acc - preds) ** 2)),
        'corr_pred_acc': _corr(preds, true_acc),
        'corr_pred_cons': _corr(preds, cons),
        'corr_cons_acc': _corr(cons, true_acc),
    }

def load_lora(peft_model, checkpoint_dir, hidden_size):
    """Hot-swap the LoRA weights of the existing PeftModel in place (constant memory)
    and load the matching regression head. Returns the head."""
    adapter_weights = load_peft_weights(checkpoint_dir, device="cuda")
    load_result = set_peft_model_state_dict(peft_model, adapter_weights, adapter_name="default")
    unexpected = getattr(load_result, "unexpected_keys", [])
    assert not unexpected, f"Unexpected keys when loading adapter from {checkpoint_dir}: {unexpected[:5]}"

    head = nn.Linear(hidden_size, 1, dtype=torch.float32).to('cuda')
    head.load_state_dict(torch.load(os.path.join(checkpoint_dir, "linear_head.pt"), weights_only=True))
    head.eval()
    return head

@torch.inference_mode()
def evaluate(model, head, split, tokenizer, batch_size, max_length):
    """Run the probe on a tokenized split and return predicted accuracies (original order)."""
    input_ids = split['input_ids']
    n = len(input_ids)

    # Sort by length (desc) for efficient padding, keep permutation to restore order
    order = sorted(range(n), key=lambda i: len(input_ids[i]), reverse=True)
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = np.arange(n)

    pad = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt', max_length=max_length)

    preds = []
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        feats = [{'input_ids': input_ids[i], 'attention_mask': split['attention_mask'][i]} for i in idxs]
        batch = pad(feats)
        ids = batch['input_ids'].to('cuda')
        attn = batch['attention_mask'].to('cuda')

        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        hidden = out.hidden_states[-1]                       # (B, T, H)

        last_idx = last_token_index(attn)                    # right padding
        b_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[b_idx, last_idx].float()             # (B, H)

        pred = torch.sigmoid(head(pooled)[:, 0])
        preds.append(pred.float().cpu())

    preds = torch.cat(preds).numpy()
    return preds[inv_order]                                  # back to original order

# ============================================================================= Main

if __name__ == '__main__':
    args = parser.parse_args()
    print('args', args)

    run_dir = os.path.join(CHECKPOINT_PATH, args.project, 'unsloth_full', args.checkpoint_name)
    assert os.path.isdir(run_dir), f"Run folder not found: {run_dir}"

    # ---- List checkpoints (checkpoint-{step} dirs, sorted by step) + final root adapter if present
    ckpt_re = re.compile(r'^checkpoint-(\d+)$')
    checkpoints = []
    for name in os.listdir(run_dir):
        m = ckpt_re.match(name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            checkpoints.append((int(m.group(1)), name))
    checkpoints.sort()
    checkpoints = [name for _, name in checkpoints]
    if any(f.startswith('adapter_model') for f in os.listdir(run_dir)):
        checkpoints.append('.')  # final model saved at the run root
    assert checkpoints, f"No checkpoints found in {run_dir}"
    print(f"Found {len(checkpoints)} checkpoints:", checkpoints)

    # ---- Read probe metadata (root if training finished, else any checkpoint)
    meta_path = os.path.join(run_dir, "probe_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(run_dir, checkpoints[0], "probe_meta.json")
    with open(meta_path, "r") as f:
        probe_meta = json.load(f)
    model_name = probe_meta["model"]
    train_dataset_name = probe_meta["dataset"]
    probe_position = probe_meta["probe"]
    print(f"Probe meta: model={model_name}, trained on={train_dataset_name}, probe={probe_position}")

    # ---- Resolve the list of eval datasets (single name stays backward compatible) ----------
    perf_root = os.path.join(DATA_PATH, 'processed_with_performance')
    if args.eval_dataset.strip().lower() == 'all':
        eval_datasets = sorted(
            name for name in os.listdir(perf_root)
            if os.path.exists(os.path.join(perf_root, name, f'{model_name}_quantized_dataset'))
        )
        assert eval_datasets, f"No datasets with a {model_name}_quantized_dataset found under {perf_root}"
    else:
        eval_datasets = [d.strip() for d in args.eval_dataset.split(',') if d.strip()]
    assert eval_datasets, "No eval datasets provided"
    print(f"Eval datasets ({len(eval_datasets)}):", eval_datasets)

    # ---- Load base model ONCE (same settings as training) -----------------------------------
    # This is the expensive step (~minutes). It is reused across all checkpoints AND all
    # eval datasets below, instead of being reloaded once per dataset.
    model_path = os.path.join(MODELS_PATH, 'unsloth', model_name)
    print("loading model:", model_path)
    base_model, tokenizer = unsloth.FastModel.from_pretrained(model_path,
                                                              load_in_4bit=True,
                                                              weights_only=True,
                                                              trust_remote_code=False)
    if hasattr(tokenizer, "tokenizer"):  # multimodal wrapper
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: the probe was trained with RIGHT padding + last-non-pad-token pooling.
    # A raw forward pass does not left-pad-shift position ids the way generate() does,
    # so we must reproduce the training-time right padding exactly.
    tokenizer.padding_side = 'right'

    # ---- Truncate the stack if it is a mid probe (must happen BEFORE wrapping with PEFT)
    layers_path = get_decoder_layers_path(model_path)
    if probe_position == 'mid':
        truncate_decoder_to_half(base_model, layers_path)

    # ---- Wrap once with the first checkpoint; later checkpoints are hot-swapped in place
    peft_model = PeftModel.from_pretrained(base_model,
                                           os.path.join(run_dir, checkpoints[0]),
                                           is_trainable=False)
    peft_model.eval()

    try:
        hidden_size = peft_model.config.hidden_size
    except AttributeError:
        hidden_size = peft_model.config.text_config.hidden_size
    if "hidden_size" in probe_meta:
        assert hidden_size == probe_meta["hidden_size"], "hidden_size mismatch with probe_meta"

    def prepare_split(raw, size=None):
        """Tokenize a raw split and attach labels [accuracy, consistency]. Returns a plain dict.
        size=None => use the FULL split (no subsampling)."""
        texts = raw['formatted_questions']
        n = len(texts)
        indices = np.arange(n)
        if size is not None:
            rng = np.random.default_rng(args.seed)
            k = max(1, int(round(size)))
            indices = np.sort(rng.choice(n, size=min(k, n), replace=False))

        outputs = raw['outputs']
        def _np(o):
            return o.detach().float().cpu().numpy() if isinstance(o, torch.Tensor) else np.asarray(o, dtype=np.float32)

        sel_texts = [texts[i] for i in indices]
        acc = np.asarray([float(raw['accuracies'][i]) for i in indices], dtype=np.float64)
        cons = consistency_array([_np(outputs[i]) for i in indices])

        tokenized = tokenizer([PREPROMPT + t + POSTPROMPT for t in sel_texts],
                              truncation=True, max_length=args.max_length,
                              padding=False, return_tensors=None)
        return {
            'indices': indices.tolist(),
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'accuracies': acc,
            'consistency': cons,
        }

    def saved_len(entry):
        """Number of examples stored in a saved split entry (robust to missing keys)."""
        preds = entry.get('predictions', None) if isinstance(entry, dict) else None
        return len(preds) if preds is not None else -1

    # ---- Evaluate every requested dataset, reusing the single model load --------------------
    summary = {}  # (tag, eval_dataset) -> metrics_only
    for eval_dataset in eval_datasets:
        print(f"\n########## Eval dataset: {eval_dataset} ##########")

        # Load eval dataset (same format/labels as training so ground-truth accuracy is available)
        ds = torch.load(os.path.join(perf_root, eval_dataset, f'{model_name}_quantized_dataset'),
                        weights_only=False)
        n_train_full = len(ds['train']['formatted_questions'])
        n_test_full = len(ds['test']['formatted_questions'])

        def expected_split_sizes(tag):
            """Split name -> expected number of examples for this checkpoint on this dataset."""
            if tag == 'final':
                return {'train': n_train_full, 'test': n_test_full}
            return {'train_subset': min(max(1, int(round(args.train_size))), n_train_full),
                    'test': min(max(1, int(round(args.test_size))), n_test_full)}

        # Lazy tokenization cache for THIS dataset (reset per dataset).
        _split_cache = {}

        def get_split(variant):
            if variant not in _split_cache:
                if variant == 'train':
                    print("Tokenizing FULL train split...")
                    _split_cache[variant] = prepare_split(ds['train'], size=None)
                elif variant == 'test_full':
                    print("Tokenizing FULL test split...")
                    _split_cache[variant] = prepare_split(ds['test'], size=None)
                elif variant == 'train_subset':
                    print("Tokenizing train subset...")
                    _split_cache[variant] = prepare_split(ds['train'], size=args.train_size)
                elif variant == 'test_subset':
                    print("Tokenizing test subset...")
                    _split_cache[variant] = prepare_split(ds['test'], size=args.test_size)
                else:
                    raise ValueError(f"Unknown split variant: {variant}")
                print(f"  {variant}: {len(_split_cache[variant]['input_ids'])} examples")
            return _split_cache[variant]

        def get_split_for(tag, split_name):
            if split_name == 'test':
                return get_split('test_full' if tag == 'final' else 'test_subset')
            return get_split(split_name)  # 'train' (final) or 'train_subset' (intermediate)

        # ---- Evaluate every checkpoint on this dataset
        for ckpt_name in tqdm.tqdm(checkpoints):
            ckpt_dir = os.path.join(run_dir, ckpt_name)
            tag = 'final' if ckpt_name == '.' else ckpt_name
            if tag != 'final' and args.last_only:
                continue

            eval_dir = os.path.join(ckpt_dir, 'evals')
            out_pt = os.path.join(eval_dir, f'{eval_dataset}.pt')
            out_json = os.path.join(eval_dir, f'{eval_dataset}_metrics.json')

            expected = expected_split_sizes(tag)

            # ---- Compare what is already saved with what should be computed.
            # A split is up to date iff it exists in the saved file AND its stored predictions
            # have exactly the expected size; otherwise it goes in `todo`.
            saved_splits = {}
            if os.path.exists(out_pt) and not args.overwrite:
                try:
                    saved = torch.load(out_pt, weights_only=False)
                    saved_splits = saved.get('splits', {})
                except Exception as e:
                    print(f"[WARN] {tag}/{eval_dataset}: could not load {out_pt} ({e}); recomputing")
                    saved_splits = {}

            todo = {name: size for name, size in expected.items()
                    if saved_len(saved_splits.get(name, {})) != size}
            kept = {name: saved_splits[name] for name in expected if name not in todo}

            if not todo:
                print(f"[SKIP] {tag}/{eval_dataset}: all splits already computed with matching sizes "
                      f"({', '.join(f'{k}={v}' for k, v in expected.items())})")
                summary[(tag, eval_dataset)] = {k: v['metrics'] for k, v in kept.items() if 'metrics' in v}
                continue

            for name in todo:
                prev = saved_len(saved_splits.get(name, {}))
                reason = "missing" if prev < 0 else f"size {prev} != expected {todo[name]}"
                print(f"[EVAL] {tag}/{eval_dataset}: recomputing '{name}' ({reason})"
                      + (f", keeping {sorted(kept)}" if kept else ""))

            head = load_lora(peft_model, ckpt_dir, hidden_size)

            results = {'eval_dataset': eval_dataset,
                       'checkpoint': tag,
                       'probe_meta': probe_meta,
                       'seed': args.seed,
                       'train_size': 'full' if tag == 'final' else args.train_size,
                       'test_size': 'full' if tag == 'final' else args.test_size,
                       'splits': dict(kept)}  # carry over splits that were already valid
            metrics_only = {name: entry['metrics'] for name, entry in kept.items() if 'metrics' in entry}

            for split_name in todo:
                split = get_split_for(tag, split_name)
                preds = evaluate(peft_model, head, split, tokenizer, args.batch_size, args.max_length)
                m = compute_metrics(preds, split['accuracies'], split['consistency'])
                results['splits'][split_name] = {
                    'indices': split['indices'],
                    'predictions': preds.astype(np.float32),
                    'accuracies': split['accuracies'].astype(np.float32),
                    'consistency': split['consistency'].astype(np.float32),
                    'metrics': m,
                }
                metrics_only[split_name] = m
                print(f"    {split_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

            os.makedirs(eval_dir, exist_ok=True)
            torch.save(results, out_pt)
            with open(out_json, 'w') as f:
                json.dump(metrics_only, f, indent=2)
            summary[(tag, eval_dataset)] = metrics_only

            del head
            torch.cuda.empty_cache()

        # Free this dataset's raw tensors before loading the next one
        del ds
        _split_cache.clear()

    print("\n===== Summary (test split, corr_pred_acc) =====")
    for (tag, eval_dataset), m in summary.items():
        if 'test' in m:
            print(f"  {tag} / {eval_dataset}: corr={m['test']['corr_pred_acc']:.4f}, "
                  f"pass@0.05={m['test']['0.05']:.4f}")
    print("Done.")