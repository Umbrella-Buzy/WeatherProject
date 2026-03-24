import torch
import torch.nn as nn
import math

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
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T_q, -1)
        attn_out =self.projection(attn_out)
        return attn_out