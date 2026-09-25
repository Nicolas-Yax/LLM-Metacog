import os
# Fix ministral crash
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"   # don't generate/use compiled modules
os.environ["TORCHDYNAMO_DISABLE"] = "1"       # make every torch.compile a no-op (eager)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, TrainingArguments, Trainer, GPTJForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset,Dataset,DatasetDict,load_from_disk
from peft import LoraConfig, get_peft_model
import numpy as np
import re
import hashlib
import json
from tqdm import tqdm
import argparse
import ast
import random
import unsloth

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,help='model to run')
parser.add_argument('--batch_size',type=int,help='batch size')
parser.add_argument('--dataset',type=str,help='dataset to run')
parser.add_argument('--perf_suffix',type=str,default='quantized',
                    help="which performance file to read the questions from: "
                         "'quantized' or 'fullmodel' (falls back to the other "
                         "one if the requested file is missing)")

args = parser.parse_args()

if __name__ == '__main__':
    NUMBERS = False

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS_PATH = 'basemodels'
    DATA_PATH = 'data/datasets'
    model_path = os.path.join(MODELS_PATH,'unsloth',args.model)

    #Load Model
    model_path = os.path.join(MODELS_PATH,'unsloth',args.model)
    model,tokenizer = unsloth.FastModel.from_pretrained(model_path,
                                                                load_in_4bit=True,#args.quantized,
                                                                weights_only=True,
                                                               trust_remote_code=False)

    #For multimodal models
    if hasattr(tokenizer,"tokenizer"):
        tokenizer = tokenizer.tokenizer

    tokenizer.padding_side = 'left'

    #Load dataset: the performance file of THIS model (torch.save of a
    #DatasetDict/dict whose 'train'/'test' splits are dicts of parallel lists),
    #so the baseline is computed on exactly the same (token-filtered) questions.
    suffixes = [args.perf_suffix] + [s for s in ('quantized','fullmodel') if s != args.perf_suffix]
    perf_path = None
    for suffix in suffixes:
        candidate = os.path.join(DATA_PATH,'processed_with_performance',args.dataset,
                                 f'{args.model}_{suffix}_dataset')
        if os.path.exists(candidate):
            perf_path = candidate
            break
    if perf_path is None:
        raise FileNotFoundError(
            f"No performance file for model '{args.model}' in "
            f"{os.path.join(DATA_PATH,'processed_with_performance',args.dataset)} "
            f"(tried suffixes {suffixes})")
    print('loading performance dataset:',perf_path)
    try:
        ds = torch.load(perf_path,weights_only=False)  #holds tensors + python lists
    except TypeError:  #older torch without weights_only kwarg
        ds = torch.load(perf_path)
    test_dataset = ds['test']

    #Pull out the columns we need (note plural names in performance files),
    #then free the heavy outputs/logits tensors before loading batches on GPU.
    questions  = list(test_dataset['questions'])
    choices_all = list(test_dataset['choices'])
    answers    = list(test_dataset['answers'])
    categories = list(test_dataset['categories']) if 'categories' in test_dataset else None
    accuracies = list(test_dataset['accuracies']) if 'accuracies' in test_dataset else None
    del ds, test_dataset
    gc.collect()

    PREPROMPT = f"You are going to be asked to guess your likelihood to be right on a given question. Give your answer with the format 'Accuracy prediction: X%' but replace X with your percentage chance to be right on this question.\nQuestion :"
    POSTPROMPT = f"Accuracy prediction:"# X% with X="

    print('preprompt',PREPROMPT)
    print('postprompt',POSTPROMPT)

    def format_question(question,choices,PREPROMPT,POSTPROMPT):
        s = PREPROMPT + question
        for i,c in enumerate(choices):
            if NUMBERS:
                s += '\n('+str(i+1)+') '+c
            else:
                s += '\n('+chr(65+i)+') '+c
        return s + '\n' + POSTPROMPT

    #Pre-format all prompts once
    input_texts_all = [format_question(question,choices,PREPROMPT,POSTPROMPT)
                       for question,choices in zip(questions,choices_all)]

    #Sort indices by tokenized length so each batch contains similar-length inputs (minimises padding)
    lengths = [len(ids) for ids in tokenizer(input_texts_all).input_ids]
    sorted_indices = sorted(range(len(input_texts_all)), key=lambda k: lengths[k], reverse=True)  #longest first: OOM would happen on batch 0

    num_samples = len(questions)  #len() of the dict would count keys, not rows
    batch_size = args.batch_size
    predictions = [None] * num_samples  #pre-allocated, filled back in original dataset order

    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            for i in range(0, num_samples, batch_size):
                print('--- i',i)
                batch_idx = sorted_indices[i:i+batch_size]
                input_texts = [input_texts_all[k] for k in batch_idx]
                print(input_texts)

                inputs = tokenizer(input_texts, padding="longest", truncation=False, return_tensors="pt").to(device)
                print(inputs.input_ids.shape)
                prediction_token_ids = model.generate(**inputs,do_sample=False,max_new_tokens=5,min_new_tokens=5)
                prediction_tokens = tokenizer.batch_decode(prediction_token_ids,skip_special_tokens=True)

                for j,k in enumerate(batch_idx):
                    completion = prediction_tokens[j][len(input_texts[j]):]
                    predicted_percent = re.findall("(\d+(?:\.\d+)?)%",completion)
                    print(input_texts[j])
                    print(prediction_tokens[j])
                    print(completion)
                    print(predicted_percent)
                    if len(predicted_percent) == 1:
                        predictions[k] = predicted_percent[0]
                    else:
                        print(prediction_tokens)
                        print(predicted_percent)
                        predictions[k] = None

                pbar.update(len(batch_idx))

    print('len(prediction)',len(predictions))
    texts = input_texts_all  #already computed above, in original dataset order
    meta_train_dataset = {'questions':questions,
            'prompts':texts,
            'answers':answers,
            'predictions':predictions}
    if categories is not None:  #performance files may lack a categories column
        meta_train_dataset['categories'] = categories
    if accuracies is not None:  #carry the measured accuracy over: predicted vs actual comes for free
        meta_train_dataset['accuracies'] = accuracies

    os.makedirs(os.path.join(DATA_PATH,'processed_with_baselineguess',args.dataset),
                exist_ok=True
               )

    #Save dataset
    torch.save(meta_train_dataset,
               os.path.join(DATA_PATH,'processed_with_baselineguess',args.dataset,
                             '{}_dataset'.format(args.model)))
