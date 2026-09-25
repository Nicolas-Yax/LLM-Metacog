#make_performance_dataset.py
import torch
import unsloth
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    GPTJForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    set_seed,
    Gemma3ForCausalLM,
)
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import numpy as np
import re
import os
from tqdm import tqdm
import argparse
import ast
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to run')
parser.add_argument('--dataset', type=str, help='dataset to run')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--quantized', type=int, help='quantized')

args = parser.parse_args()

if __name__ == '__main__':
    NUMBERS = False  # True means options are labelled by numbers. False means they are labelled by characters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODELS_PATH = 'basemodels'
    DATA_PATH = 'data/datasets/'
    model_path = os.path.join(MODELS_PATH, 'unsloth', args.model)

    print(args)

    #auto_model = AutoModelForCausalLM
    #if "gemma-3" in model_path:
    #    auto_model = Gemma3ForCausalLM

    base_model, tokenizer = unsloth.FastModel.from_pretrained(
        model_path, 
        #auto_model = auto_model,
        trust_remote_code=False, 
        load_in_4bit=bool(args.quantized),
    )
    
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        
    base_model = unsloth.FastLanguageModel.for_inference(base_model)
    if not (args.quantized):
        base_model = base_model.to(device)

    # --- FIX (silent correctness bug): causal LMs need LEFT padding so that the real
    #     last token sits at position -1 for every row in a batch. With the default
    #     right padding, outputs[:, -1] reads a pad position for all but the longest
    #     sequence, producing wrong logits for batch_size > 1.
    #     Also: padding="longest" needs a pad token to exist.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- FIX: load_from_disk does not accept a config name. The second positional arg
    #     is keep_in_memory, so passing 'default' was a misuse. Dropped it.
    ds = load_from_disk(os.path.join(f'{DATA_PATH}/processed/{args.dataset}'))

    # Instruction prompt
    PREPROMPT = 'Question :'
    POSTPROMPT = 'Answer : Option ('

    # NOTE: StopOnResultPattern is only meaningful with model.generate(). It is NOT used
    # below (we run a single forward pass), but the typo is fixed in case you wire it up.
    class StopOnResultPattern(StoppingCriteria):
        def __init__(self, stop_token, tokenizer):
            self.stop_token = stop_token
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            # Decode the generated tokens so far
            decoded_output = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            for t in decoded_output:
                if not (self.stop_token in t[len(PREPROMPT):]):
                    return False  # Not finished yet
            return True  # --- FIX: was "Trueet" (NameError if ever called)

    def format_question(question, choices, PREPROMPT, POSTPROMPT):
        s = PREPROMPT + question
        for i, c in enumerate(choices):
            if NUMBERS:
                s += '\n(' + str(i + 1) + ') ' + c
            else:
                s += '\n(' + chr(65 + i) + ') ' + c
        return s + '\n' + POSTPROMPT

    def evaluate(eval_dataset, model, tokenizer, batch_size=32):
        # Computes, per question, the probability mass the model assigns to the correct
        # option letter (and the corresponding summed logits).
        # NOTE: this is a *soft* score (mean probability on the gold letter), NOT 0/1
        # classification accuracy. Renamed the accumulator accordingly for clarity.
        num_samples = len(eval_dataset)
        score_per_question = []

        outputsl = []   # probability mass per option letter, per question
        outputsl2 = []  # summed logits per option letter, per question

        # Token-string variants for each option letter A..J.
        answer_tokens = [
            [L, L + ' ', L + ')', L + ') ', ' ' + L + ' ', ' ' + L + ') ', ' ' + L, ' ' + L + ')']
            for L in 'ABCDEFGHIJ' #max 10 options in datasets
        ]

        answer_token_ids = [[] for _ in range(len(answer_tokens))]
        for li in range(len(answer_tokens)):
            for variant in answer_tokens[li]:
                try:
                    if hasattr(tokenizer, 'get_vocab'):
                        answer_token_ids[li].append(tokenizer.get_vocab()[variant])
                    else:
                        answer_token_ids[li].append(tokenizer.tokenizer.get_vocab()[variant])
                except KeyError:
                    pass
            # --- FIX: original asserted len(answer_token_ids) >= 1 (always 10, useless).
            #     Check the per-letter sublist instead, and warn rather than crash on a
            #     letter that has no matching vocab token.
            if len(answer_token_ids[li]) == 0:
                print(f"Warning: no vocab token found for option letter '{chr(65 + li)}'")

        print(answer_token_ids)

        with torch.no_grad():
            with tqdm(total=num_samples) as pbar:
                # --- FIX: renamed the batch index from `i` to `start` so it is no longer
                #     shadowed by the inner option-letter loop variable.
                for start in range(0, num_samples, batch_size):
                    batch = eval_dataset[start:start + batch_size]

                    input_texts = [
                        format_question(question, choices, PREPROMPT, POSTPROMPT)
                        for question, choices in zip(batch['question'], batch['choices'])
                    ]

                    inputs = tokenizer(
                        input_texts, padding="longest", truncation=False, return_tensors="pt"
                    ).to(device)
                    outputs = model(**inputs,logits_to_keep=1).logits
                    outputs = outputs[:, -1]  # last-token logits (correct now that padding is left-side)

                    probs = outputs.softmax(1).cpu()
                    logits_cpu = outputs.cpu()

                    # --- FIX (hard crash): size the per-letter tensors to the number of
                    #     choices actually present in this batch, and only iterate over that
                    #     many letters. The original hard-coded 4 rows but looped over all 10
                    #     answer letters, raising IndexError as soon as an E..J token existed.
                    n_choices = max(len(c) for c in batch['choices'])
                    assert n_choices <= len(answer_token_ids), \
                        "More choices than supported option letters (A-J)."

                    probs_letters = torch.zeros((n_choices, probs.shape[0]), dtype=torch.float32)
                    logits_letters = torch.zeros((n_choices, probs.shape[0]), dtype=torch.float32)

                    for li in range(n_choices):
                        for tid in answer_token_ids[li]:
                            probs_letters[li, :] += probs[:, tid]
                            logits_letters[li, :] += logits_cpu[:, tid]

                    for j in range(len(outputs)):
                        label = batch['answer'][j]
                        # --- FIX: store a plain float, not a tensor, so np.mean is robust.
                        score_per_question.append(float(probs_letters[label, j]))
                        outputsl.append(probs_letters[:, j])
                        outputsl2.append(logits_letters[:, j])

                    #Checkup in logs
                    print('-------------------')
                    print("inputs",input_texts[0])
                    print("probs",probs_letters[:,0])
                    print("coverage",probs_letters[:,0].sum())
                    print("most likely next token",tokenizer.decode([probs[0].argmax()]))

                    pbar.update(len(batch['question']))

        print('len(score_per_question)', len(score_per_question))
        accuracy = float(np.mean(score_per_question)) if score_per_question else 0.0
        return accuracy, score_per_question, outputsl, outputsl2

    def make_meta_dataset(dataset, model, tokenizer, batch_size=32):
        # Make dataset with questions as input and per-question scores as labels.
        global PREPROMPT, POSTPROMPT
        _, labels, outputs, outputs2 = evaluate(dataset, model, tokenizer, batch_size=batch_size)
        prompts = [format_question(q, c, PREPROMPT, POSTPROMPT)
                   for q, c in zip(dataset['question'], dataset['choices'])]
        formatted_questions = [format_question(q, c, '', '')
                               for q, c in zip(dataset['question'], dataset['choices'])]

        meta = {
            'questions': list(dataset['question']),
            'prompts': prompts,
            'formatted_questions': formatted_questions,
            'choices': list(dataset['choices']),
            'answers': list(dataset['answer']),
            'accuracies': labels,
            'outputs': outputs,
            'logits': outputs2,
        }
        try:
            meta['categories'] = list(dataset['category'])
        except KeyError:  # If no category column in dataset
            pass
        return meta

    os.makedirs(
        os.path.join(DATA_PATH, 'processed_with_performance', args.dataset),
        exist_ok=True
    )

    performance_dataset = DatasetDict({
        'train': make_meta_dataset(ds['train'], base_model, tokenizer, batch_size=args.batch_size),
        'test': make_meta_dataset(ds['test'], base_model, tokenizer, batch_size=args.batch_size),
    })

    # Save dataset (kept as torch.save: the values include tensors, so this is not a
    # true HF Dataset / save_to_disk target).
    torch.save(
        performance_dataset,
        os.path.join(
            DATA_PATH, 'processed_with_performance', args.dataset,
            '{}_{}_dataset'.format(
                model_path.split('/')[-1],
                'quantized' if args.quantized else 'fullmodel'
            )
        )
    )
