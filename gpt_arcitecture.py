import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.nn.functional import cross_entropy # type: ignore
from torch.utils.data import Dataset # type: ignore

import math
import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

if torch.cuda.is_available():
    torch.set_default_device("cuda")

GPT_CONFIG_124M = {
    "n_vocab": 50257,    # Vocabulary size
    "n_ctx": 1024, # Context length
    "n_embd": 768,         # Embedding dimension
    "n_head": 12,          # Number of attention heads
    "n_layer": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True       # Query-Key-Value bias
}

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special= {"<|endoftext|>",})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
 
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim = True)

        x = (x-mean)/ torch.sqrt(var + self.eps)
        
        return x * self.scale + self.shift

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),
            GeLU(),
            nn.Linear(4*embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    # num of embeddings are n, n at max can be equal to context length
    def __init__(self, embed_dim, output_dim, num_heads, drop_rate, qkv_bias = False):
        super().__init__()

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # trainable layers for initializing queries, keys and values
        self.w_queries = nn.Linear(embed_dim, output_dim*num_heads, bias=qkv_bias) # embed_dim x output_dim*num_heads
        self.w_keys    = nn.Linear(embed_dim, output_dim*num_heads, bias=qkv_bias)
        self.w_values  = nn.Linear(embed_dim, output_dim*num_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, embeddings):
        words = embeddings.shape[-2]

        embeddings = torch.reshape(embeddings, (-1, words, self.embed_dim))  # batches x context_len x embed_dim
        batches = embeddings.shape[0]

        all_queries = self.w_queries(embeddings)  # batches x context_len x (num_heads*output_dim)
        all_keys    = self.w_keys(embeddings)
        all_values  = self.w_values(embeddings)

        # reshape to (B, T, H, d) then transpose to (B, H, T, d)
        queries = all_queries.view(batches, words, self.num_heads, self.output_dim).transpose(1, 2)
        keys    = all_keys.view(batches, words, self.num_heads, self.output_dim).transpose(1, 2)
        values  = all_values.view(batches, words, self.num_heads, self.output_dim).transpose(1, 2)

        # scaled dot-product attention: (B, H, T, T)
        attention_scores = queries @ keys.transpose(2, 3)
        # causal mask over last two dims (T x T), broadcast to (B, H, T, T)
        causal_mask_bool = torch.triu(
            torch.ones(words, words, device=attention_scores.device, dtype=torch.bool),
            diagonal=1
        )

        attention_scores = attention_scores.masked_fill(causal_mask_bool, float("-inf"))

        # softmax over keys dimension
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)  # batches x numheads x context_len x context_len
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)  # batches , context , numheads x output_dim
        context_vector = context_vector.contiguous().view(batches, words, self.output_dim*self.num_heads)

        context_vector = self.out_proj(context_vector)
        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm_1 = LayerNorm(config["n_embd"])
        self.output_dim = config["n_embd"] // config["n_head"]
        self.attention = MultiHeadAttention(config["n_embd"], self.output_dim, config["n_head"], config["drop_rate"], config["qkv_bias"])
        self.dropout = nn.Dropout(config["drop_rate"])
        
        self.norm_2 = LayerNorm(config["n_embd"])
        self.ffnetwork = FeedForward(config["n_embd"])


    def forward(self, input_vector):
        out = self.norm_1(input_vector)
        out = self.attention(out)
        out = self.dropout(out)

        out += input_vector
        shortcut = out

        out = self.norm_2(out)
        out = self.ffnetwork(out)
        out = self.dropout(out)
        
        out += shortcut

        return out

class GPTModel(nn.Module):
    def __init__(self, config = GPT_CONFIG_124M):
        super().__init__()
        self.tok_emb = nn.Embedding(config["n_vocab"], config["n_embd"])
        self.pos_emb = nn.Embedding(config["n_ctx"], config["n_embd"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layer"])]
        )

        self.final_norm = LayerNorm(config["n_embd"])
        self.out_head = nn.Linear(config["n_embd"], config["n_vocab"], bias=False)

    def forward(self, in_idx):
        in_idx = torch.tensor(in_idx, dtype = int)
        in_idx = in_idx.reshape(-1,in_idx.shape[-1])
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        # x = torch.reshape(x, shape = (-1 , min(self.batch_size, len(in_idx)), GPT_CONFIG_124M["n_embd"]))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits

def assign(left,right):
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"Shape mismatch: left {tuple(left.shape)} vs right {tuple(right.shape)}"
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_params(model, params):
    model.pos_emb.weight = assign(model.pos_emb.weight, params["wpe"])
    model.tok_emb.weight = assign(model.tok_emb.weight, params["wte"])

    # --- transformer blocks ---
    for i in range(len(params["blocks"])):
        # layer norms
        model.trf_blocks[i].norm_1.scale = assign(model.trf_blocks[i].norm_1.scale, params["blocks"][i]["ln_1"]["g"])
        model.trf_blocks[i].norm_1.shift = assign(model.trf_blocks[i].norm_1.shift, params["blocks"][i]["ln_1"]["b"])

        model.trf_blocks[i].norm_2.scale = assign(model.trf_blocks[i].norm_2.scale, params["blocks"][i]["ln_2"]["g"])
        model.trf_blocks[i].norm_2.shift = assign(model.trf_blocks[i].norm_2.shift, params["blocks"][i]["ln_2"]["b"])

        # feed-forward layers
        model.trf_blocks[i].ffnetwork.layers[0].weight = assign(
            model.trf_blocks[i].ffnetwork.layers[0].weight,
            params["blocks"][i]["mlp"]["c_fc"]["w"].T
        )
        model.trf_blocks[i].ffnetwork.layers[0].bias = assign(
            model.trf_blocks[i].ffnetwork.layers[0].bias,
            params["blocks"][i]["mlp"]["c_fc"]["b"]
        )
        # layer 1 is GeLU activation
        model.trf_blocks[i].ffnetwork.layers[2].weight = assign(
            model.trf_blocks[i].ffnetwork.layers[2].weight,
            params["blocks"][i]["mlp"]["c_proj"]["w"].T
        )
        model.trf_blocks[i].ffnetwork.layers[2].bias = assign(
            model.trf_blocks[i].ffnetwork.layers[2].bias,
            params["blocks"][i]["mlp"]["c_proj"]["b"]
        )

        # attention weights
        q_w, k_w, v_w = np.split(params["blocks"][i]["attn"]["c_attn"]["w"], 3, axis=-1)
        model.trf_blocks[i].attention.w_queries.weight = assign(model.trf_blocks[i].attention.w_queries.weight, q_w.T)
        model.trf_blocks[i].attention.w_keys.weight   = assign(model.trf_blocks[i].attention.w_keys.weight, k_w.T)
        model.trf_blocks[i].attention.w_values.weight = assign(model.trf_blocks[i].attention.w_values.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][i]["attn"]["c_attn"]["b"], 3, axis=-1)
        model.trf_blocks[i].attention.w_queries.bias = assign(model.trf_blocks[i].attention.w_queries.bias, q_b)
        model.trf_blocks[i].attention.w_keys.bias   = assign(model.trf_blocks[i].attention.w_keys.bias, k_b)
        model.trf_blocks[i].attention.w_values.bias = assign(model.trf_blocks[i].attention.w_values.bias, v_b)

        model.trf_blocks[i].attention.out_proj.weight = assign(
            model.trf_blocks[i].attention.out_proj.weight, 
            params["blocks"][i]["attn"]["c_proj"]["w"].T
        )
        model.trf_blocks[i].attention.out_proj.bias = assign(
            model.trf_blocks[i].attention.out_proj.bias, 
            params["blocks"][i]["attn"]["c_proj"]["b"]
        )
    # --- output and final normalization ---
    model.out_head.weight = assign(model.out_head.weight, params["wte"])
    model.final_norm.scale = assign(model.final_norm.scale, params["g"])
    model.final_norm.shift = assign(model.final_norm.shift, params["b"])

def generate(model, sentence, max_new_tokens=20, context_size=1024, temperature=1.0, top_k=None, eos_id=50246):
    model.eval()
    tokens = tokenizer.encode(sentence)
    # For-loop is the same as before: Get logits, and only focus on last time step
    
    for _ in range(max_new_tokens):
        tokens_cond = tokens[-context_size:]
        with torch.no_grad():
            logits = model(tokens_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

 
        logits = logits / temperature

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

        # Sample from the distribution
        tokens_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)


        if tokens_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        tokens.append(tokens_next)

    new_sentence = tokenizer.decode(tokens)
    return new_sentence

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    vocab = json.load(open(os.path.join(model_dir, "encoder.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    settings["drop_rate"] = 0.1
    settings["qkv_bias"] = True

    return settings, vocab, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params