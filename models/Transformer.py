import torch
import torch.nn as nn
import math
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, model_dim, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim

        self.Q = nn.Linear(model_dim, num_heads * head_dim)
        self.K = nn.Linear(model_dim, num_heads * head_dim)
        self.V = nn.Linear(model_dim, num_heads * head_dim)
        self.projection = (nn.Linear(num_heads * head_dim, model_dim)
                           if num_heads*head_dim != model_dim else nn.Identity())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None, mask=None):
        context = context if context is not None else x
        B, T_q, _ = x.shape
        _, T_kv, _ = context.shape
        q = self.Q(x).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.K(context).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(context).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim))
        if mask is not None:
            if mask.dim() == 3:
                mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T_q, -1)
        attn_out =self.projection(attn_out)
        return attn_out

class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.attention = Attention(model_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim * 4, model_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.attention(self.norm1(x))
        x = self.dropout(x) + residual
        residual = x
        x = self.ffn(self.norm2(x))
        x = self.dropout(x) + residual
        return x

class DecoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        self.self_attention = Attention(model_dim, num_heads, head_dim)
        self.cross_attention = Attention(model_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim, model_dim * 4, model_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attention(x)
        x = self.dropout(x) + residual
        residual = x
        x = self.cross_attention(self.norm2(x), context, mask)
        x = self.dropout(x) + residual
        residual = x
        x = self.ffn(self.norm3(x))
        x = self.dropout(x) + residual
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask = torch.tril(torch.ones(config['pred_steps'], config['time_steps'])).bool()
        self.enc_input_proj = nn.Linear(config['features'], config['model_dim'])
        self.dec_input_proj = nn.Linear(config['features'], config['model_dim'])
        self.pos_encoder = PositionalEncoding(config['model_dim'], config['dropout'])
        self.output_proj = nn.Linear(config['model_dim'], config['features'])
        self.encoder = nn.ModuleList([EncoderBlock(config['model_dim'], config['num_heads'], config['head_dim'], config['dropout'])
                        for _ in range(config['enc_layers'])])
        self.decoder = nn.ModuleList([DecoderBlock(config['model_dim'], config['num_heads'], config['head_dim'], config['dropout'])
                        for _ in range(config['dec_layers'])])

    def forward(self, x, tgt, mask=None):
        casual_mask = self.mask if mask is None else mask
        x = self.enc_input_proj(x)
        out = self.dec_input_proj(tgt)
        x = self.pos_encoder(x)
        out = self.pos_encoder(out)
        for encoder_block in self.encoder:
            x = encoder_block(x)
        for decoder_block in self.decoder:
            out = decoder_block(out, x, casual_mask)
        out = self.output_proj(out)
        return out

from utils import Config
config = Config("../config/config.yml")

model = Transformer(config)
x = torch.randn(1, 72, 6)
context = torch.randn(1, 24, 6)
y = model(x, context)
print(y)