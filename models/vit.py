import math

from timm.models.layers import DropPath
import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super(MultiHeadAttention, self).__init__()
    self.config = config
    self.model_embed_size = self.config.model_config.hidden_size
    self.num_heads = self.config.model_config.transformer.num_heads
    self.head_embed_size = self.embed_size // self.num_heads
    self.transformer_attention_dropout_rate = self.config.model_config.transformer.attention_dropout_rate
    self.transformer_dropout_rate = self.config.model_config.transformer.dropout_rate

    self.W_q = nn.Linear(self.model_embed_size, self.model_embed_size)
    self.W_k = nn.Linear(self.model_embed_size, self.model_embed_size)
    self.W_v = nn.Linear(self.model_embed_size, self.model_embed_size)
    self.W_o = nn.Linear(self.model_embed_size, self.model_embed_size)

  def split_heads(self, x):
    batch_size, token_length, emb_size = x.size()
    return x.view(batch_size, token_length, self.num_heads, self.head_embed_size)

  def merge_heads(self, x):
    pass
    batch_size, _, token_length, embed_head = x.size()
    return x.transpose(1, 2).contiguous.view(batch_size, token_length, self.model_embed_size)

  def scale_dot_product_attention(self, q, k, v, mask=None):
    # ----> To DO: Explain the divison and that it can be other things too.
    attention_scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.head_embed_size)
    
    if mask is not None:
      mask = (mask != 0).unsqueeze(1).unsqueeze(2)
      attention_scores = attention_scores.masked_fill(mask == 0, 1e-9)

    attention_probs = torch.softmax(attention_scores, dim=-1)

    attention_probs_dropout = F.dropout(attention_probs, p=self.transformer_attention_dropout_rate, training=True)

    output = torch.matmul(attention_probs_dropout, v)
    return output, attention_probs

  def forward(self, q, k, v, input_mask, causal_mask):
    q = self.split_heads(self.W_q(q))
    k = self.split_heads(self.W_q(k))
    v = self.split_heads(self.W_q(v))

    attention_output, attention_probs = self.scale_dot_product_attention(q, k, v, input_mask)
    output = self.W_o(self.merge_heads(attention_output))
    output = F.dropout(output, p=self.transformer_dropout_rate, training=True)
    return output, attention_probs


class MLP(nn.Module):
  def __init__(self, config):
    super(MLP, self).__init__()
    self.config = config
    outer_channels = self.config.model_config.hidden_size
    inner_channels = self.config.model_config.transformer.mlp_dim
    self.transformer_dropout_rate = self.config.model_config.transformer.dropout_rate

    self.Linear1 = nn.Linear(outer_channels, inner_channels)
    self.Linear2 = nn.Linear(outer_channels, inner_channels)
    self.activation_layer = nn.GELU()

  def forward(self, x):
    x = self.activation_layer(self.Linear1(x))
    x = F.dropout(x, p=self.transformer_dropout_rate, training=True)
    x = self.Linear2(x)
    x = F.dropout(x, p=self.transformer_dropout_rate, training=True)
    return x


class VIT_Block(nn.Module):
  def __init__(self, config, drop_path_percentage):
    super(VIT_Block, self).__init__()
    self.config = config
    hidden_size = self.config.model_config.hidden_size
    self.transformer_dropout_rate = self.config.model_config.transformer.dropout_rate

    self.LayerNorm1 = nn.LayerNorm(hidden_size)
    self.MultiHeadAttention2 = MultiHeadAttention(self.config)
    self.LayerNorm3 = nn.LayerNorm(hidden_size)
    self.MLP4 = MLP(self.config)

    self.drop_path = DropPath(drop_path_percentage) if self.config.model_config.dropout_path else nn.Identity()

  def forward(self, inputs, input_mask):
    x = self.LayerNorm1(inputs)
    x, attention_probs = self.MultiHeadAttention2(x, x, x, input_mask, causal_mask=None)
    x = F.dropout(x, p=self.transformer_dropout_rate, training=True)

    # introduce drop_path
    x = self.drop_path(x) + inputs

    y = self.LayerNorm3(x)
    y = self.MLP4(y)
    output = self.drop_path(y) + x
    return output, attention_probs



class VIT(nn.Module):
  def __init__(self, config):
    super(VIT, self).__init__()
    self.config = config

    self.positional_embedding = None
    self.vit_blocks = None  # loop over VIT_BLOCK depending on number of layers.
    self.cls = None
    self.final_norm = None
  
  def forward(x):
    pass

