# -*- coding: utf-8 -*-
"""
This code is based on timm library https://github.com/rwightman/pytorch-image-models
"""

from torch import nn
import torch
import math

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_conv1=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv1 = use_conv1
        if use_conv1:
            self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, stride=1,padding='same')
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if use_conv1:
            self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=3, stride=1,padding='same')
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_conv1:
            x = x.transpose(1,2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.use_conv1:
            x = x.transpose(1,2)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(in_dim_q, out_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim_k, out_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkmatrix = None

    def forward(self, x, x_q):
        B, Nk, Ck = x.shape
        B, Nq, Cq = x_q.shape
        q = self.q(x_q).reshape(B, Nq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B, Nk, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)
        q = q.squeeze(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
       
        attn = attn.softmax(dim=-1)
        
        self.qkmatrix = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, self.qkmatrix

# Cross attention implementation with residual and dropout
class AttentionBlock(nn.Module):

    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,  use_conv1 = False):
        super().__init__()
        self.norm1_q = norm_layer(in_dim_q)
        self.norm1_k = norm_layer(in_dim_k)
        self.attn = Attention(in_dim_k=in_dim_k,in_dim_q=in_dim_q,
            out_dim=out_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, use_conv1=use_conv1)

    def forward(self, xk,xq):
        
        x, a = self.attn(self.norm1_k(xk), self.norm1_q(xq))
        x = self.drop_path(x)
        x = x +  self.drop_path(self.mlp(self.norm2(x)))
        return x

'''class EEGTransformerEncoder(nn.Module):
    def __init__(self, input_features=14, d_model=128, num_heads=8, num_layers=4):
        super(EEGTransformerEncoder, self).__init__()
        self.d_model = d_model

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

    def forward(self, x, mask):
        """
        Forward pass for the EEG Transformer Encoder model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, d_features)
        """
        positional_encoding = self._generate_positional_encoding(x.size(1), self.d_model)
        positional_encoding = positional_encoding.to("cuda")
        # Add positional encoding
        x = x + positional_encoding[:,:x.size(1),:]

        # Pass through transformer encoder
        
        mask = mask==0
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Return the final embedding
        return x.mean(dim=1)  # Aggregates across time steps to get a fixed-size embedding

    def _generate_positional_encoding(self, length, d_model):
        """
        Generates a positional encoding matrix of shape (length, d_model).
        """
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add a batch dimension'''
