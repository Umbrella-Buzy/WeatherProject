import torch
import torch.nn as nn
import math
import torch.optim as optim
from models.basic_structures import FFN, Attention
from utils import PositionalEncoding, WeatherDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

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
        x = self.self_attention(x, mask=mask)
        x = self.dropout(x) + residual
        residual = x
        x = self.cross_attention(self.norm2(x), context)
        x = self.dropout(x) + residual
        residual = x
        x = self.ffn(self.norm3(x))
        x = self.dropout(x) + residual
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.enc_input_proj = nn.Linear(config['features'], config['model_dim'])
        self.dec_input_proj = nn.Linear(config['out_features'], config['model_dim'])
        self.pos_encoder = PositionalEncoding(config['model_dim'], config['dropout'])
        self.output_proj = nn.Linear(config['model_dim'], config['out_features'])
        self.encoder = nn.ModuleList([EncoderBlock(config['model_dim'], config['num_heads'], config['head_dim'], config['dropout'])
                        for _ in range(config['enc_layers'])])
        self.decoder = nn.ModuleList([DecoderBlock(config['model_dim'], config['num_heads'], config['head_dim'], config['dropout'])
                        for _ in range(config['dec_layers'])])

    def forward(self, x, tgt):
        tgt_l = tgt.shape[1]
        casual_mask = torch.tril(torch.ones(tgt_l, tgt_l)).bool().to(self.config['device'])
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

    def generate(self, x):
        dec_input = x[:, -1, 6:].unsqueeze(1)
        x = self.enc_input_proj(x)
        x = self.pos_encoder(x)
        for encoder_block in self.encoder:
            x = encoder_block(x)
        for _ in range(self.config['pred_steps']):
            tgt_l = dec_input.shape[1]
            casual_mask = torch.tril(torch.ones(tgt_l, tgt_l)).bool().to(self.config['device'])
            out = self.dec_input_proj(dec_input)
            out = self.pos_encoder(out)
            for decoder_block in self.decoder:
                out = decoder_block(out, x, casual_mask)
            out = self.output_proj(out)
            next_step =out[:, -1, :].unsqueeze(1)
            dec_input = torch.cat((dec_input, next_step), dim=1)
        return dec_input[:, 1:, :]









