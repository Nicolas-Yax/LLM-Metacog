import os
import argparse
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='embedding model to use')
parser.add_argument('--dataset', type=str, help='dataset to embed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_PATH = 'basemodels'
    DATA_PATH = 'data/datasets'
    model_path = os.path.join(MODELS_PATH, 'unsloth', args.model)
    
    # Load embedding model (plain HF, not unsloth: embedding models aren't causal LMs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype="auto").to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def last_token_pool(last_hidden_states, attention_mask):
        # Right-padded rows: last real token sits at (sum(mask) - 1).
        # Left-padded rows: last real token is always at position -1.
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]
    
    def embed_texts(texts, indices, batch_size):
        """
        Embed texts sorted by length to minimize padding overhead.
        Returns embeddings in original order along with original indices.
        
        Args:
            texts: list of text strings
            indices: original indices to restore order
            batch_size: batch size for processing
        
        Returns:
            embeddings: tensor of shape (len(texts), embedding_dim)
            original_indices: indices to restore original order
        """
        embeddings = [None] * len(texts)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i + batch_size]
                batch_indices = indices[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                
                outputs = model(**inputs, output_hidden_states=False)
                pooled = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Store embeddings in correct positions
                for j, orig_idx in enumerate(batch_indices):
                    embeddings[orig_idx] = pooled[j].float().cpu()
        
        return torch.stack(embeddings, dim=0)
    
    # Load a performance-dataset file for this dataset. formatted_questions is the
    # same regardless of which LLM was evaluated, so any file in the directory works.
    perf_dir = os.path.join(DATA_PATH, 'processed_with_performance', args.dataset)
    perf_files = sorted(os.listdir(perf_dir))
    if not perf_files:
        raise FileNotFoundError(f"No performance files found in {perf_dir}")
    perf_path = os.path.join(perf_dir, perf_files[0])
    print('loading performance dataset:', perf_path)
    ds = torch.load(perf_path, weights_only=False)
    
    embedded_dataset = {}
    for split in ds:
        formatted_questions = list(ds[split]['formatted_questions'])
        print(f'embedding {len(formatted_questions)} questions from split "{split}"')
        
        # Sort questions by length to minimize padding
        lengths = np.array([len(q) for q in formatted_questions])
        sorted_indices = np.argsort(lengths)
        sorted_questions = [formatted_questions[i] for i in sorted_indices]
        
        print(f'  sorted by length (min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f})')
        
        # Embed sorted questions
        embeddings = embed_texts(sorted_questions, sorted_indices, args.batch_size)
        
        embedded_dataset[split] = {
            'formatted_questions': formatted_questions,
            'embeddings': embeddings,
        }
    
    os.makedirs(os.path.join(DATA_PATH, 'embedded'), exist_ok=True)
    save_path = os.path.join(DATA_PATH, 'embedded', f'{args.dataset}.pt')
    torch.save(embedded_dataset, save_path)
    print('saved to', save_path)