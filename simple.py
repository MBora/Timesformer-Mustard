import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        q = self.query(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        weighted_avg = torch.matmul(attn, v)
        weighted_avg = weighted_avg.transpose(1, 2).contiguous().view(batch_size, num_patches, self.embed_dim)

        return self.fc(weighted_avg)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class TimeSformer(nn.Module):
    def __init__(self, num_classes, num_frames, num_patches, embed_dim, num_heads, num_layers, mlp_dim, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(num_patches * embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames * num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, num_frames, num_patches, embed_dim = x.size()
        x = x.view(batch_size, num_frames * num_patches, embed_dim)
        x = self.patch_embed(x) + self.pos_embed
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)
