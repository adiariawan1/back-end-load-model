import torch
import torch.nn as nn
import tiktoken
from app.core.config import settings
from app.models.base_model import BaseModel
from huggingface_hub import hf_hub_download

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * x_norm + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            (2 / torch.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(), 
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        
        assert d_out % num_heads == 0
        
        self.d_out = d_out
        self.num_heads = num_heads
        
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key =nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        
        self.out_proj =nn.Linear(d_in,d_out)
        
        self.dropout = nn.Dropout(dropout)
        
        
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self,x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)
    


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
    
    


class AstraModel(BaseModel):
    def __init__(self):
        self.device =torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer =None
        self.config =None
        
    def load(self,path: str= settings.HF_MODEL_ID):
        print(f"proccess load model ke{self.device}")   
        
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        file_path = hf_hub_download(repo_id=path, filename="astra-zero-v1.pt")
        
        ckpt = torch.load(file_path, map_location=self.device, weights_only=False)
        
        self.config = {
            "vocab_size": 50257,    
            "context_length": 256, 
            "emb_dim": 512,         
            "n_heads": 8,          
            "n_layers": 6,        
            "drop_rate": 0.0,
            "qkv_bias": False
        }
        
    
        
        self.model = GPTModel(self.config)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        
        self.model.to(self.device)
        
        self.model.eval()
        
        print("model successfully loaded")
        
    
    def preprocess(self, raw_input: str):
        system_prompt =( "You are a virtual assistant specialized in supporting medical professionals "
            "in their clinical practice by providing reliable, up-to-date, and evidence-based "
            "medical information."
        )
        
        full_prompt = (
            f"{system_prompt}\n\n"
            f"### Instruction:\nAnswer this question truthfully\n\n"
            f"### Input:\n{raw_input}\n\n"
            f"### Response:\n"
        )
        
        tokens = self.tokenizer.encode(full_prompt)
        
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        return input_tensor
    
    def predict(self, processed_input):
        context_length = self.config["context_length"]
        max_new_tokens = settings.MAX_NEW_TOKENS
        temperature = 1.0
        top_k = 60
        top_p = 1.0
        repetition_penalty = 2.0
        generated_tokens = []
        
        with torch.no_grad():
            
            for _ in range(max_new_tokens):
                
                
                input_cond = processed_input[:, -context_length:]

                logits = self.model(input_cond)

                logits = logits[:, -1, :] / temperature
                
                for token_id in set(generated_tokens):
                    logits[0, token_id] /= repetition_penalty

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = -float("inf")
               
                probs = torch.softmax(logits, dim=-1)

                
                next_token = torch.multinomial(probs, num_samples=1)

                
                processed_input = torch.cat([processed_input, next_token], dim=1)
                
               
                generated_tokens.append(next_token.item())

                
                if next_token.item() == self.tokenizer.eot_token:
                    break

        return processed_input

    def postprocess(self, raw_output) -> str:
       
        token_list = raw_output[0].tolist()
        
        
        output = self.tokenizer.decode(token_list)
        
        
        if "### Response:" in output:
            output = output.split("### Response:")[-1]
            
        
        hasil_akhir = output.replace("<|endoftext|>", "").strip()
        
        return hasil_akhir
    
    
astra_model = AstraModel()
            