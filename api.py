# api.py  (FastAPI backend for IMDB Sentiment Transformer)

import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

#model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # attn_mask: (B,T) True for real tokens, False for padding
        pad_mask = ~attn_mask  # MultiheadAttention expects True for padding positions
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attn_mask):
        x = self.embed(input_ids)
        x = self.pos(x)
        for layer in self.encoder:
            x = layer(x, attn_mask)
        mask_f = attn_mask.float().unsqueeze(-1)  # (B,T,1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(pooled))

#FastAPI App
app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Load artifact
with open("artifact.pkl", "rb") as f:
    artifact = pickle.load(f)
vocab = artifact["vocab"]
MAX_LEN = artifact["max_len"]
PAD_IDX = artifact["pad_idx"]
UNK_IDX = artifact["unk_idx"]
cfg = artifact["model_cfg"]

#build & load model
model = TransformerSentimentClassifier(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    dropout=cfg["dropout"],
    num_classes=cfg["num_classes"],
).to(device)
model.load_state_dict(torch.load("sentiment_transformer.pt", map_location=device))
model.eval()

class ReviewIn(BaseModel):
    review: str

def preprocess(text: str):
    tokens = text.lower().split()
    ids = [vocab.get(w, UNK_IDX) for w in tokens]
    if len(ids) < MAX_LEN:
        ids += [PAD_IDX] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  
    attn_mask = (input_ids != PAD_IDX)                            
    return input_ids, attn_mask
@app.get("/")
def home():
    return {"message": "Sentiment API running"}

@app.post("/predict")
def predict(payload: ReviewIn):
    input_ids, attn_mask = preprocess(payload.review)
    with torch.no_grad():
        logits = model(input_ids, attn_mask)
        pred = torch.argmax(logits, dim=1).item()
    return {"sentiment": "positive" if pred == 1 else "negative"}