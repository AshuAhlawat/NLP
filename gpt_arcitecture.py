import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset

if torch.cuda.is_available():
    torch.set_default_device("cuda")

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
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
        x = 0.5 * x * (1 + torch.tanh( torch.tensor((2/torch.pi)**0.5) ) * ( x + 0.044715*x**3) )
        return x

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

class MultiHeadAttention(torch.nn.Module):
    # num of embeddings are n, n at max can be equal to context length
    def __init__(self, embed_dim, output_dim, num_heads):
        super().__init__()

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # trainable layers for initializing queries, keys and values
        self.w_queries = torch.nn.Linear(embed_dim, output_dim*num_heads, bias=False) # embed_dim x output_dim*num_heads
        self.w_keys = torch.nn.Linear(embed_dim, output_dim*num_heads, bias=False)
        self.w_values = torch.nn.Linear(embed_dim, output_dim*num_heads, bias=False)

    def forward(self, embeddings):
        words = embeddings.shape[-2]
        
        embeddings = torch.reshape(embeddings, (-1, words, self.embed_dim))  # batches x context_len x embed_dim
        batches = len(embeddings)

        all_queries = self.w_queries(embeddings) # batches x context_len x output_dim*num_heads
        all_keys = self.w_keys(embeddings)
        all_values = self.w_values(embeddings)

        # split them in columns and then line them up in a tensor
        queries =  all_queries.view(batches, self.num_heads, words, self.output_dim)
        keys = all_keys.view(batches, self.num_heads, words, self.output_dim)
        values = all_values.view(batches, self.num_heads, words, self.output_dim)


        attention_scores = queries @ keys.transpose(2,3) # numheads x n x n

        # as this is causal attention now we will mask the upper right diagonal
        causal_mask_bool =  torch.triu(torch.ones_like(attention_scores), diagonal=1).bool() #triu stands for triangle up

        attention_scores.masked_fill_(causal_mask_bool, -torch.inf) # now a word only depend on the words before it, as the future dependencies are -ve infinity so after softmax the probabilities will be zero

        attention_weights = torch.softmax(attention_scores / self.output_dim**0.5, dim=2) # batches x numheads x context_len x context_len

        context_vectors = (attention_weights @ values).transpose(1, 2)
        # batches x context x numheads x output_dim <from> batches x numheads x context x output_dim
        context_vector = torch.reshape(context_vectors, shape= (batches, words, self.output_dim*self.num_heads))
        # batches x context x numheads*output_dim
        return context_vector

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm_1 = LayerNorm(config["emb_dim"])
        output_dim = config["emb_dim"] // config["n_heads"]
        self.attention = MultiHeadAttention(config["emb_dim"], output_dim, config["n_heads"])
        self.dropout = nn.Dropout(config["drop_rate"])

        
        self.norm_2 = LayerNorm(config["emb_dim"])
        self.ffnetwork = FeedForward(config["emb_dim"])


    def forward(self, input_vector):
        original = input_vector.detach()

        out = self.norm_1(original)
        out = self.attention(out)
        out = self.dropout(out)

        out += original

        out = self.norm_2(out)
        out = self.ffnetwork(out)
        out = self.dropout(out)
        
        out += original

        return out

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        in_idx = torch.tensor(in_idx, dtype = int)
        in_idx = in_idx.reshape(-1,in_idx.shape[-1])
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        # x = torch.reshape(x, shape = (-1 , min(self.batch_size, len(in_idx)), GPT_CONFIG_124M["emb_dim"]))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits