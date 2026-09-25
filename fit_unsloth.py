import os

os.environ.setdefault("UNSLOTH_DISABLE_GRADIENT_OFFLOADING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import uuid
import json
import argparse

import numpy as np
import torch
import unsloth
import torch.nn as nn

from datasets import Dataset as hDataset
from transformers import TrainingArguments, DataCollatorWithPadding
from trl import SFTTrainer

#Token location during right padding resolution
def last_token_index(attention_mask, padding_side):
    """Index of the last real (non-pad) token per example for right padding"""
    return attention_mask.sum(dim=1) - 1

#Consistency computation
def scaled_entropy_consistency(out, eps=1e-12):
    """Computes consistency = 1 - H(p)/log(K), where p is the base-model per-option probability mass (`outputs` column) RENORMALISED over the K real options, and H is Shannon entropy (nats). In other words, it returns the entropy scaled between its minimal (0) and maximal (log(max_options)) values."""
    p = np.asarray(out, dtype=np.float64).reshape(-1)
    p = np.clip(p / p.sum(), eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(1.0 - H / np.log(p.shape[0]))

def consistency_array(outputs_list):
    return np.array([scaled_entropy_consistency(o) for o in outputs_list], dtype=np.float64)


#Cut transformer stack utilities
def get_decoder_layers_path(model_path):
    if any(i in model_path for i in ("Qwen2.5", "Llama-3", "Llama-2", "Mistral-7B")):
        return "model.layers"                  # *ForCausalLM.model.layers
    elif any(i in model_path for i in ("gemma-3","Ministral-3","Qwen3.5",)):
        return "model.language_model.layers"   # Gemma3ForConditionalGeneration.model.language_model.layers
    elif any(i in model_path for i in ("granite",)):
        return "model.layers"                  # GraniteForCausalLM.model.layers
    elif any(i in model_path for i in ("phi-4",)):
        return "model.layers"                  # Phi3ForCausalLM.model.layers
    else:
        print("model:", model_path)
        assert False, "Undefined decoder-layers path for given model"

def _resolve_layers(model, layers_path):
    """Walk `layers_path` -> (parent_module, attr_name, module_list)."""
    *parents, attr = layers_path.split(".")
    obj = model
    for p in parents:
        obj = getattr(obj, p)
    layers = getattr(obj, attr)
    assert isinstance(layers, nn.ModuleList), f"'{layers_path}' is not a ModuleList (got {type(layers).__name__})"
    return obj, attr, layers

def truncate_decoder_to_half(model, layers_path):
    """Drop the TOP half of the decoder layers in-place. Call BEFORE get_peft_model.
    Returns (n_full, n_used). Kept layers are [0, n_used), so their layer indices are
    already correct and need no reindexing."""
    parent, attr, layers = _resolve_layers(model, layers_path)
    n_full = len(layers)
    n_used = n_full // 2
    setattr(parent, attr, nn.ModuleList(list(layers[:n_used])))  # keep bottom half

    # keep config layer count consistent (covers multimodal text_config too)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = n_used

    print(f"[MID-PROBE] {layers_path}: {n_full} -> {n_used} (kept bottom half)")
    return n_full, n_used

def count_lora_layers(peft_model):
    """How many distinct decoder-layer indices carry trainable LoRA params (sanity check)."""
    idxs = set()
    for name, p in peft_model.named_parameters():
        if "lora_" in name and p.requires_grad:
            m = re.search(r"layers\.(\d+)\.", name)
            print(name)
            if m:
                idxs.add(int(m.group(1)))
    print(idxs)
    return len(idxs), (max(idxs) + 1 if idxs else 0)

#Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--dataset',type=str)
parser.add_argument('--model',type=str)
parser.add_argument('--lr',default=5e-5,type=float)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--nb_epochs',default=1,type=int)
parser.add_argument('--lora_rank',default=8,type=int)
parser.add_argument('--weight_decay',default=2e-2,type=float)
parser.add_argument('--warmup_ratio',type=float,default=0.1)
parser.add_argument('--project',type=str,default="default") #Name of the project to save the model to (and tensorboard logs to)
parser.add_argument('--probe', type=str, default='end', choices=['end', 'mid'])
parser.add_argument('--eval_steps', type=int, default=500)
parser.add_argument('--save_numbers', type=int, default=100) # Number of times to save within the full training
parser.add_argument('--eval_subset_size', type=int, default=1000) #All eval are done on a subset 

if __name__ == '__main__':
    args = parser.parse_args()

    print('args',args)

    #Create filename
    parts = [
        args.dataset,
        args.model,
        args.nb_epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.lora_rank,
        args.warmup_ratio,
        'probe-' + str(args.probe),
        str(uuid.uuid4())[:8],
    ]
    filename = '_'.join(str(p) for p in parts)

    #Define paths
    MODELS_PATH = 'basemodels'
    DATA_PATH = 'data/datasets'
    LOG_PATH = 'data/outputs'
    
    project_name = args.project
    SAVE_PATH = f'trainedmodels/models/{project_name}/unsloth_full'

    os.environ.setdefault("TENSORBOARD_LOGGING_DIR",os.path.join(f"data/{project_name}/outputs",filename))

    MULTIMODAL = False

    #Load Model
    model_path = os.path.join(MODELS_PATH,'unsloth',args.model)
    print("loading model:",model_path)
    base_model,tokenizer = unsloth.FastModel.from_pretrained(model_path,
                                                             load_in_4bit=True,
                                                             weights_only=True,
                                                             trust_remote_code=False)
    
    #Get text tokenizer in multimodal models
    if hasattr(tokenizer, "tokenizer"):
        print("multimodal tokenizer detected")
        MULTIMODAL = True
        tokenizer = tokenizer.tokenizer

    tokenizer.padding_side = 'right' #The index of end of padding will be retrieved in the loss function to not fit on pad tokens
    #tokenizer.truncation_side = "left"   # keep the POSTPROMPT / end of prompt

    #Finds where to truncate the transformer stack if mid training option is given
    layers_path = get_decoder_layers_path(model_path)
    if args.probe == 'mid':
        n_layers_full, n_layers_used = truncate_decoder_to_half(base_model, layers_path)
    else:
        _parent, _attr, _layers = _resolve_layers(base_model, layers_path)
        n_layers_full = len(_layers)
        n_layers_used = n_layers_full

    #Load dataset
    ds = torch.load(os.path.join(f'{DATA_PATH}/processed_with_performance/{args.dataset}/{args.model}_quantized_dataset'),weights_only=False)
    train_dataset,validation_dataset = ds['train'],ds['test']

    train_dataset['texts'] = train_dataset['formatted_questions']
    validation_dataset['texts'] = validation_dataset['formatted_questions']

    #Add consistency in the datasets for tensorboard evaluation
    def _outputs_to_np_list(outputs):
        np_list = []
        for o in outputs:
            if isinstance(o, torch.Tensor):
                np_list.append(o.detach().float().cpu().numpy())
            else:
                np_list.append(np.asarray(o, dtype=np.float32))
        return np_list

    train_outputs_raw = _outputs_to_np_list(train_dataset['outputs'])
    validation_outputs_raw = _outputs_to_np_list(validation_dataset['outputs'])

    train_consistency = consistency_array(train_outputs_raw)
    validation_consistency = consistency_array(validation_outputs_raw) 

    train_dataset['labels'] = [[float(a), float(c)] for a, c in zip(train_dataset['accuracies'], train_consistency)]
    validation_dataset['labels'] = [[float(a), float(c)] for a, c in zip(validation_dataset['accuracies'], validation_consistency)]

    train_dataset = hDataset.from_dict(train_dataset)
    validation_dataset = hDataset.from_dict(validation_dataset)


    #Define LoRA target modules (attention + MLP but names differ depending on the architecture).
    if any(i in model_path for i in ("Qwen2.5","Qwen3.5","Llama-3","Llama-2","Mistral-7B","gemma-3","Ministral-3")):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
    elif any(i in model_path for i in ("granite",)):
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "shared_mlp.input_linear", "shared_mlp.output_linear",
        ]
    elif any(i in model_path for i in ("phi-4",)):
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        print("model:",model_path)
        assert False #Undefined target modules for given model

    if MULTIMODAL: #To avoid adding modules on non-language transformer stack
        _target_modules = [
            name for name, mod in base_model.named_modules()
            if isinstance(mod, nn.Linear)
            and "language_model" in name          # excludes vision_tower entirely
            and name.endswith(tuple(target_modules))
        ]
        assert _target_modules and not any("vision_tower" in n for n in _target_modules)
        target_modules = _target_modules


    peft_model = unsloth.FastLanguageModel.get_peft_model(
        base_model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = target_modules,
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        #finetune_vision_layers   = False,   # <- vision tower stays frozen, no LoRA
        #finetune_language_layers = True,
        #finetune_attention_modules = True,
        #finetune_mlp_modules       = True,
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    peft_model.print_trainable_parameters()

    # === [MID-PROBE] === verify LoRA only landed on the kept (bottom-half) layers
    if args.probe == 'mid':
        n_lora_layers, max_lora_layer = count_lora_layers(peft_model)
        print(f"[MID-PROBE] LoRA present on {n_lora_layers} distinct layers "
              f"(highest idx {max_lora_layer - 1}); kept {n_layers_used} of {n_layers_full}")
        assert max_lora_layer <= n_layers_used, (
            "LoRA found on a layer beyond the kept bottom half - truncation must happen "
            "BEFORE get_peft_model.")

    #Get model hidden size (try/except for multimodal models)
    try:
        model_size = peft_model.config.hidden_size
    except AttributeError:
        model_size = peft_model.config.text_config.hidden_size

    print('model_size',model_size)

    #Add final decoding linear layer to peft model
    peft_model.linear_head = nn.Linear(model_size,1,dtype=torch.float32).to('cuda')

    def compute_metrics(pred,eps=1e-12):
        """ Compute correlations and pass@0.05 (how many samples have prediction error lower than 5%) metrics"""
        label_ids = np.asarray(pred.label_ids, dtype=np.float64)
        if label_ids.ndim == 1:
            label_ids = label_ids.reshape(-1, 1)

        true_acc = label_ids[:, 0].reshape(-1)
        cons = label_ids[:, 1].reshape(-1)
        preds = np.asarray(pred.predictions, dtype=np.float64).reshape(-1)

        def _corr(a, b):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if np.std(a) < eps or np.std(b) < eps:
                return float('nan')
            return float(np.corrcoef(a, b)[0, 1])

        with torch.no_grad():
            score = float(np.mean(np.abs(true_acc - preds) < 0.05))
            return {
                '0.05': score,
                'corr_pred_acc': _corr(preds, true_acc),            # predicted accuracy vs true accuracy
                'corr_pred_cons': _corr(preds, cons),      # predicted accuracy vs consistency
                'corr_cons_acc': _corr(cons, true_acc),    # consistency vs true accuracy
            }

    class MetaCogCollator:
        """Data Collator that handles the 2 dimensions of labels (accuracies, consistencies): pop them before padding and then putting them back."""
        def __init__(self, tokenizer, max_length=2048):
            self._pad = DataCollatorWithPadding(
                tokenizer, padding='longest', return_tensors='pt', max_length=max_length
            )

        def __call__(self, features):
            features = [dict(f) for f in features]
            labels = [f.pop("labels") for f in features]
            # keep ONLY what tokenizer.pad should tensorize
            keep = ("input_ids", "attention_mask")
            features = [{k: f[k] for k in keep if k in f} for f in features]
            batch = self._pad(features)
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)  # (B, 2)
            return batch

    class MetaCogTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            #Load data
            labels = inputs["labels"].float() #(batch,2) -> the second dimension is [accuracies,consistencies], the former for training+evaluation and the latter for evaluation only
            target = labels[:, 0] #Get accuracies for loss computation

            #Compute model output
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1] #shape (B, T, H)

            #Get last token in each batch (due to right padding each input in the batch does not have the same index for last token position)
            last_idx = last_token_index(inputs["attention_mask"], tokenizer.padding_side)
            idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[idx, last_idx].float() #shape (B, H])

            #Head prediction
            log_pred = model.linear_head(pooled)
            pred = torch.sigmoid(log_pred[:, 0])

            #Loss computation
            loss = nn.functional.mse_loss(pred, target)
            return (loss, (None, pred)) if return_outputs else loss

        def get_batch_samples(self, epoch_iterator, num_batches, device=None, *args, **kwargs):
            batch_samples = []
            for _ in range(num_batches):
                try:
                    batch_samples.append(next(epoch_iterator))
                except StopIteration:
                    break
            return batch_samples, None

        def _save_probe_metadata(self,output_dir):
            """ Saves the probe position metadata """
            with open(os.path.join(output_dir, "probe_meta.json"), "w") as f:
                    json.dump(getattr(self, "probe_meta", {}), f, indent=2)

        def save_model(self, output_dir=None, _internal_call=False):
            """Save ONLY the LoRA adapter (+ the regression head + probe metadata)."""

            #Create folder
            if output_dir is None:
                output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            #Save LoRA only
            self.model.save_pretrained(output_dir)

            #Save linear head
            linear_head_path = os.path.join(output_dir, "linear_head.pt")
            torch.save(self.model.linear_head.state_dict(), linear_head_path)

            #Save position of the probe in the transformer stack
            self._save_probe_metadata(output_dir)

            #Print where adapters and head have been saved
            print(f"Adapter saved to: {output_dir}")
            print(f"Linear head saved to: {linear_head_path}")

        def _save_checkpoint(self, model, trial, metrics=None):
            """Override checkpoint saving to include linear_head + probe metadata"""
            #Call parent save checkpoint
            checkpoint_folder = super()._save_checkpoint(model, trial)
            
            if checkpoint_folder is None:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                print(f"[CKPT] adapter saved to (fallback): {ckpt_dir}")
                checkpoint_folder = ckpt_dir
                    
            #Save linear head
            linear_head_path = os.path.join(checkpoint_folder, "linear_head.pt")
            torch.save(self.model.linear_head.state_dict(), linear_head_path)
            
            #Save position of the probe in the transformer stack
            self._save_probe_metadata(checkpoint_folder)

            #Print where checkpoint has been saved
            print(f"Linear head checkpoint saved to: {linear_head_path}")

            return checkpoint_folder

    PREPROMPT = f"You are going to be asked to guess your likelihood to be right on a given question. Give your answer with the format 'Accuracy prediction: X%' but replace X with your percentage chance to be right on this question.\nQuestion :"
    POSTPROMPT = f"Accuracy prediction:"

    def tokenize_function(examples):
        """Tokenize the texts in the dataset"""
        tokenized = tokenizer(
            [PREPROMPT+t+POSTPROMPT for t in examples['texts']],
            truncation=False,
            padding=False,  #Padding is done in the data collator
            #max_length=2048,
            return_tensors=None  # Return lists, not tensors
        )
        # Keep the labels (now 2-col [accuracy, consistency])
        tokenized['labels'] = examples['labels']
        return tokenized

    #Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['texts']  # Remove the original text column
    )

    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['texts']  # Remove the original text column
    )

    print("Tokenization complete!")

    #Subsample train and test sets for fast evaluation (full eval will be done in another script)
    def make_eval_subset(dataset, size, seed=42):
        rng = np.random.default_rng(seed)
        n = min(int(size), len(dataset))
        idx = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())
        return dataset.select(idx)
    
    seed = 42
    eval_datasets = {
        "sub_train": make_eval_subset(train_dataset, args.eval_subset_size, seed),
        "sub_test": make_eval_subset(validation_dataset, args.eval_subset_size, seed),
    }

    print("Eval subsampling done!")
    print(f"Train eval set size: {len(eval_datasets['sub_train'])}")
    print(f"Validation eval set size: {len(eval_datasets['sub_test'])}")

    #Create data collator
    data_collator = MetaCogCollator(tokenizer, max_length=2048) #Questions are not much longer than 1k tokens so should be ok

    gradient_acc_steps = max(1, 16//args.batch_size)

    nb_steps = len(train_dataset)*args.nb_epochs//(args.batch_size*gradient_acc_steps)
    save_steps = nb_steps//args.save_numbers
    print(f"Number of steps: {nb_steps}, saving every {save_steps} steps")

    #Create trainer
    trainer = MetaCogTrainer(
        model = peft_model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_datasets,          # === [SUBSET-EVAL] === dict: {train2k, test}
        data_collator = data_collator,
        compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # === [ANALYSIS] === enabled (fixes the (None,pred) tuple)
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps = gradient_acc_steps, #Fix batch to 16
            warmup_ratio = args.warmup_ratio,
            num_train_epochs =args.nb_epochs,
            #max_steps = 60,
            learning_rate = args.lr,
            #bf16 = False,
            #fp16=False,
            logging_steps = 50,
            optim = "paged_adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = "linear",
            logging_dir= os.path.join(LOG_PATH,filename),
            output_dir = os.path.join(SAVE_PATH,filename),
            report_to = ["tensorboard"],
            eval_strategy="steps",
            eval_on_start=True,
            save_strategy="steps",
            save_only_model=True,
            save_steps=save_steps,
            eval_steps=args.eval_steps,
            remove_unused_columns=True,
            train_sampling_strategy="group_by_length",
        ),
    )

    #Save probe metadata to be saved later by the trainer for later reloading
    trainer.probe_meta = {
        "probe": args.probe,
        "n_layers_full": int(n_layers_full),
        "n_layers_used": int(n_layers_used),
        "hidden_size": int(model_size),
        "model": args.model,
        "dataset": args.dataset,
        "lora_rank": int(args.lora_rank),
    }

    #Save first checkpoint (for training curves)
    trainer.save_model(os.path.join(trainer.args.output_dir, "checkpoint-0"))

    #Start training
    trainer.train()

    #Save the final model
    trainer.save_model()
