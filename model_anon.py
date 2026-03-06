from pathlib import Path
import os
from datetime import datetime
from matplotlib import pyplot as plt
from utils import plot_spectrogram_to_numpy
from vocos import Vocos
from speechtokenizer import SpeechTokenizer
from torch import expm1, nn
import torchaudio
# from dataset_anon import NS2VCDataset, TextAudioCollate
from dataset_anon import NS2VCDataset, TextAudioCollate

import modules.commons as commons
from accelerate import Accelerator
from parametrizations import weight_norm
from operations import MultiheadAttention
from accelerate import DistributedDataParallelKwargs
from ema_pytorch import EMA
import math
import json
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import utils

from tqdm.auto import tqdm

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)

class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dropout=0):
        super().__init__()
        self.layer_norm = LayerNorm(c_in)
        conv = ConvTBC(c_in, c_out, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c_in))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        self.conv = conv

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        return x
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim=512,
        depth=1,
        num_latents = 32, # m in the paper
        heads = 8,
        ff_mult = 4,
        p_dropout = 0.2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        std = math.sqrt((4 * (1.0 - p_dropout)) / dim)
        nn.init.normal_(self.latents, mean=0, std = std)

        self.layers = nn.ModuleList([])
        self.attn = MultiheadAttention(dim, heads, dropout=p_dropout, bias=False,)

    def forward(self, x, x_mask=None, cross_mask = None):
        batch = x.shape[1]
        # x = rearrange(x, 'b c t -> t b c')
        latents = repeat(self.latents, 'n c -> b n c', b = batch).transpose(0, 1)
        latents = self.attn(latents, x, x, key_padding_mask=x_mask)[0] + latents
        assert torch.isnan(latents).any() == False
        # latents = rearrange(latents, 't b c -> b c t')
        return latents

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)
class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, kernel_size, dropout):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    if dilation==1:
        padding = kernel_size//2
    else:
        padding = dilation
    self.dilated_conv = ConvLayer(residual_channels, 2 * residual_channels, kernel_size)
    self.conditioner_projection = ConvLayer(n_mels, 2 * residual_channels, 1)
    self.output_projection = ConvLayer(residual_channels, 2 * residual_channels, 1)
    self.t_proj = ConvLayer(residual_channels, residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner,x_mask):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)
    #T B C
    y = x + self.t_proj(diffusion_step.unsqueeze(0))
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    conditioner = self.conditioner_projection(conditioner)
    y = self.dilated_conv(y) + conditioner
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    gate, filter_ = torch.chunk(y, 2, dim=-1)
    y = torch.sigmoid(gate) * torch.tanh(filter_)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    y = self.output_projection(y)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    residual, skip = torch.chunk(y, 2, dim=-1)
    return (x + residual) / math.sqrt(2.0), skip

class ResidualBlock_pro_spk(nn.Module):
  def __init__(self, prosody_dim,spk_dim, residual_channels, dilation, kernel_size, dropout,cond_probs):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    if dilation==1:
        padding = kernel_size//2
    else:
        padding = dilation
    self.dilated_conv = ConvLayer(residual_channels, 2 * residual_channels, kernel_size)
    self.non_dilated_conv = ConvLayer(2*residual_channels, 2 * residual_channels, kernel_size)

    self.non_dilated_conv2 = ConvLayer(2*residual_channels, 2 * residual_channels, kernel_size)
    self.conditioner_projection_prosody = ConvLayer(prosody_dim, 2 * residual_channels, 1)
    self.conditioner_projection_spk = ConvLayer(spk_dim, 2 * residual_channels, 1)
    self.cond_probs = cond_probs

    self.output_projection = ConvLayer(residual_channels, 2 * residual_channels, 1)
    self.t_proj = ConvLayer(residual_channels, residual_channels, 1)

  def forward(self, x, diffusion_step,content_condition, prosody_condition,spk_condition ,x_mask,use_cond=[None,None]):
    # assert (conditioner is None and self.conditioner_projection is None) or \
    #        (conditioner is not None and self.conditioner_projection is not None)
    # #T B C
    rand_condition= torch.rand(content_condition.shape[0])
    content_condition = content_condition.permute(2,0,1)
    y = x + self.t_proj(diffusion_step.unsqueeze(0))
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    # conditioner = self.conditioner_projection(conditioner)
    # conditioner = content_condition
    y = self.dilated_conv(y) + content_condition 
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    prosody_condition = prosody_condition.permute(2,0,1)
    prosody_conditioner = self.conditioner_projection_prosody(prosody_condition)
    # mask_prosody =  ((rand_condition > self.cond_probs[0]) & (rand_condition < self.cond_probs[0] + self.cond_probs[1])) | (rand_condition > self.cond_probs[0] + self.cond_probs[1] + self.cond_probs[2] )
    mask_prosody =  ((rand_condition > self.cond_probs[0]) & (rand_condition < self.cond_probs[0] + self.cond_probs[1])) | (rand_condition > self.cond_probs[0] + self.cond_probs[1] + self.cond_probs[2] )
    mask_prosody = mask_prosody.to(prosody_condition.device) 
    prosody_conditioner = prosody_conditioner.permute(1,0,2)
    null_prosody = torch.zeros(prosody_conditioner.shape,device=prosody_condition.device)
 
    if use_cond[0] is None:
        # prosody_conditioner = prosody_conditioner.permute(1,0,2)
        # null_prosody = torch.zeros(prosody_conditioner.shape,device=prosody_condition.device)
        prosody_conditioner_mask = torch.where(mask_prosody.unsqueeze(-1).unsqueeze(-1),null_prosody,prosody_conditioner)
        # prosody_conditioner_mask = prosody_conditioner_mask.permute(1,0,2)
    elif use_cond[0] == 0:
        prosody_conditioner_mask = null_prosody 
    else:
        prosody_conditioner_mask = prosody_conditioner

    prosody_conditioner_mask = prosody_conditioner_mask.permute(1,0,2)

    y = self.non_dilated_conv(y) + prosody_conditioner_mask
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    spk_condition = spk_condition.permute(2,0,1)
    spk_conditioner = self.conditioner_projection_spk(spk_condition)

    mask_spk =  (rand_condition > self.cond_probs[0] + self.cond_probs[1] + self.cond_probs[2]) 
    # mask_spk =  (rand_condition > 100) 
    mask_spk = mask_spk.to(spk_condition.device)    
    spk_conditioner = spk_conditioner.permute(1,0,2)
    null_spk = torch.zeros(spk_conditioner.shape,device=spk_condition.device)
    if use_cond[1] is None:
        spk_conditioner_mask = torch.where(mask_spk.unsqueeze(-1).unsqueeze(-1),null_spk,spk_conditioner)
    elif use_cond[1] == 0:
        spk_conditioner_mask = null_spk
    else:
        spk_conditioner_mask = spk_conditioner
    
    spk_conditioner_mask = spk_conditioner_mask.permute(1,0,2)
    y = self.non_dilated_conv2(y) + spk_conditioner_mask
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    #NOTE(anon) masked_fill masks the output everytime it is processed not to affect anything  
    
    gate, filter_ = torch.chunk(y, 2, dim=-1)
    y = torch.sigmoid(gate) * torch.tanh(filter_)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)

    y = self.output_projection(y)
    y = y.masked_fill(x_mask.t().unsqueeze(-1), 0)
    residual, skip = torch.chunk(y, 2, dim=-1)
    return (x + residual) / math.sqrt(2.0), skip


class Diffusion_Encoder_emb(nn.Module):
  def __init__(self,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      kernel_size=3,
      dilation_rate=2,
      n_layers=40,
      n_heads=8,
      p_dropout=0.2,
      cond_probs=[0.5,0.2,0.3],
      prosody_dim=256,
      spk_dim=256,
      dim_time_mult=None,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_heads=n_heads
    self.pre_conv = ConvLayer(in_channels, hidden_channels, 1)
    # self.resampler = PerceiverResampler(dim=hidden_channels, depth=1, heads=8, ff_mult=4)
    self.layers = nn.ModuleList([])
    # self.m = nn.Parameter(torch.randn(hidden_channels,32), requires_grad=True)
    # time condition
    sinu_pos_emb = SinusoidalPosEmb(hidden_channels)

    self.cond_probs = cond_probs
    self.time_mlp = nn.Sequential(
        sinu_pos_emb,
        nn.Linear(hidden_channels, hidden_channels*4),
        nn.GELU(),
        nn.Linear(hidden_channels*4, hidden_channels)
    )
    # print('time_mlp params:', count_parameters(self.time_mlp))
    self.proj = ConvLayer(hidden_channels, out_channels, 1)

    self.residual_layers = nn.ModuleList([
        ResidualBlock_pro_spk(prosody_dim,spk_dim, hidden_channels, dilation_rate, kernel_size, p_dropout,cond_probs)
        for i in range(n_layers)
    ])
    # print('residual_layers params:', count_parameters(self.residual_layers))
    self.skip_conv = ConvLayer(hidden_channels, hidden_channels, 1)
    # self.cross_attn = nn.ModuleList([
    #     MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, bias=False,)
    #     for _ in range(n_layers//3)
    # ])
    # # print('cross_attn params:', count_parameters(self.cross_attn))
    # self.film = nn.ModuleList([
    #     ConvLayer(hidden_channels, 2*hidden_channels,1)
    #     for _ in range(n_layers//3)
    # ])
    # # print('film params:', count_parameters(self.film))
    # self.prompt_proj = nn.ModuleList([
    #     ConvLayer(hidden_channels, hidden_channels, 1)
    #     for _ in range(n_layers//3)
    # ])
    # # print('prompt_proj params:', count_parameters(self.prompt_proj))
  def forward(self, x, data, t,use_cond):
    assert torch.isnan(x).any() == False
    contentvec, prosody_feat,spk_emb, prompt, contentvec_lengths, prompt_lengths = data
    x = rearrange(x, 'b c t -> t b c')
    # contentvec = rearrange(contentvec, 't b c -> b c t')
    # prompt = rearrange(prompt, 't b c -> b c t')
    _, b, _ = x.shape

    #NOTE contentvec is the preprocessed contentvec + f0, we can make it just speechtokenizer first-level
    t = self.time_mlp(t)

    x_mask = ~commons.sequence_mask(contentvec_lengths, x.size(0)).to(torch.bool)
    prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(0)).to(torch.bool)
    q_prompt_lengths = torch.Tensor([32 for _ in range(b)]).to(torch.long).to(x.device)
    q_prompt_mask = ~commons.sequence_mask(q_prompt_lengths, 32).to(torch.bool)

    # cross_mask = ~einsum('b j, b k -> b j k', ~q_prompt_mask, ~prompt_mask).view(x.shape[0], 1, q_prompt_mask.shape[1], prompt_mask.shape[1]).   \
    #     expand(-1, self.n_heads, -1, -1).reshape(x.shape[0] * self.n_heads, q_prompt_mask.shape[1], prompt_mask.shape[1])
    # prompt = self.resampler(prompt, x_mask = prompt_mask)
    # q_cross_mask = ~einsum('b j, b k -> b j k', ~x_mask, ~q_prompt_mask).view(x.shape[0], 1, x_mask.shape[1], q_prompt_mask.shape[1]).  \
    #     expand(-1, self.n_heads, -1, -1).reshape(x.shape[0] * self.n_heads, x_mask.shape[1], q_prompt_mask.shape[1])
    x = self.pre_conv(x)
    #NOTE(anon) do we need this pre_conv (probably yes it is for noisy target)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    ##last time change to here
    skip=0
    for lid, layer in enumerate(self.residual_layers):
        x, skip_connection = layer(x, diffusion_step=t, content_condition=contentvec,prosody_condition=prosody_feat,spk_condition=spk_emb, x_mask = x_mask,use_cond = use_cond)
        # if lid % 3 == 2:
        #     j = (lid+1)//3-1
        #     prompt_ = self.prompt_proj[j](prompt)
        #     x_t = x
        #     prompt_t = prompt_
        #     #CANCELED(anon) change scale shift to use speaker embeddings instead of prompt
        #     scale_shift = self.cross_attn[j](x_t, prompt_t, prompt_t, key_padding_mask=q_prompt_mask)[0]
        #     assert torch.isnan(scale_shift).any() == False
        #     scale_shift = self.film[j](scale_shift)
        #     scale_shift = scale_shift.masked_fill(x_mask.t().unsqueeze(-1), 0)
        #     scale, shift = scale_shift.chunk(2, dim=-1)
        #     x = x*scale+ shift
        #     x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
        skip = skip + skip_connection
        skip = skip.masked_fill(x_mask.t().unsqueeze(-1), 0)
    x = skip / math.sqrt(len(self.residual_layers))
    x = self.skip_conv(x)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    x = F.relu(x)
    x = self.proj(x)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    assert torch.isnan(x).any() == False
    x = rearrange(x, 't b c -> b c t')
    return x


class Diffusion_Encoder(nn.Module):
  def __init__(self,
      in_channels=128,
      out_channels=128,
      hidden_channels=256,
      kernel_size=3,
      dilation_rate=2,
      n_layers=40,
      n_heads=8,
      p_dropout=0.2,
      dim_time_mult=None,
      ):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_heads=n_heads
    self.pre_conv = ConvLayer(in_channels, hidden_channels, 1)
    self.resampler = PerceiverResampler(dim=hidden_channels, depth=1, heads=8, ff_mult=4)
    self.layers = nn.ModuleList([])
    # self.m = nn.Parameter(torch.randn(hidden_channels,32), requires_grad=True)
    # time condition
    sinu_pos_emb = SinusoidalPosEmb(hidden_channels)

    self.time_mlp = nn.Sequential(
        sinu_pos_emb,
        nn.Linear(hidden_channels, hidden_channels*4),
        nn.GELU(),
        nn.Linear(hidden_channels*4, hidden_channels)
    )
    # print('time_mlp params:', count_parameters(self.time_mlp))
    self.proj = ConvLayer(hidden_channels, out_channels, 1)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(hidden_channels, hidden_channels, dilation_rate, kernel_size, p_dropout)
        for i in range(n_layers)
    ])
    # print('residual_layers params:', count_parameters(self.residual_layers))
    self.skip_conv = ConvLayer(hidden_channels, hidden_channels, 1)
    self.cross_attn = nn.ModuleList([
        MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, bias=False,)
        for _ in range(n_layers//3)
    ])
    # print('cross_attn params:', count_parameters(self.cross_attn))
    self.film = nn.ModuleList([
        ConvLayer(hidden_channels, 2*hidden_channels,1)
        for _ in range(n_layers//3)
    ])
    # print('film params:', count_parameters(self.film))
    self.prompt_proj = nn.ModuleList([
        ConvLayer(hidden_channels, hidden_channels, 1)
        for _ in range(n_layers//3)
    ])
    # print('prompt_proj params:', count_parameters(self.prompt_proj))
  def forward(self, x, data, t):
    assert torch.isnan(x).any() == False
    contentvec, prompt, contentvec_lengths, prompt_lengths = data
    x = rearrange(x, 'b c t -> t b c')
    # contentvec = rearrange(contentvec, 't b c -> b c t')
    # prompt = rearrange(prompt, 't b c -> b c t')
    _, b, _ = x.shape

    #NOTE contentvec is the preprocessed contentvec + f0, we can make it just speechtokenizer first-level
    t = self.time_mlp(t)

    x_mask = ~commons.sequence_mask(contentvec_lengths, x.size(0)).to(torch.bool)
    prompt_mask = ~commons.sequence_mask(prompt_lengths, prompt.size(0)).to(torch.bool)
    q_prompt_lengths = torch.Tensor([32 for _ in range(b)]).to(torch.long).to(x.device)
    q_prompt_mask = ~commons.sequence_mask(q_prompt_lengths, 32).to(torch.bool)

    # cross_mask = ~einsum('b j, b k -> b j k', ~q_prompt_mask, ~prompt_mask).view(x.shape[0], 1, q_prompt_mask.shape[1], prompt_mask.shape[1]).   \
    #     expand(-1, self.n_heads, -1, -1).reshape(x.shape[0] * self.n_heads, q_prompt_mask.shape[1], prompt_mask.shape[1])
    prompt = self.resampler(prompt, x_mask = prompt_mask)
    # q_cross_mask = ~einsum('b j, b k -> b j k', ~x_mask, ~q_prompt_mask).view(x.shape[0], 1, x_mask.shape[1], q_prompt_mask.shape[1]).  \
    #     expand(-1, self.n_heads, -1, -1).reshape(x.shape[0] * self.n_heads, x_mask.shape[1], q_prompt_mask.shape[1])
    x = self.pre_conv(x)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    ##last time change to here
    skip=0
    for lid, layer in enumerate(self.residual_layers):
        x, skip_connection = layer(x, diffusion_step=t, conditioner=contentvec, x_mask = x_mask)
        if lid % 3 == 2:
            j = (lid+1)//3-1
            prompt_ = self.prompt_proj[j](prompt)
            x_t = x
            prompt_t = prompt_
            scale_shift = self.cross_attn[j](x_t, prompt_t, prompt_t, key_padding_mask=q_prompt_mask)[0]
            assert torch.isnan(scale_shift).any() == False
            scale_shift = self.film[j](scale_shift)
            scale_shift = scale_shift.masked_fill(x_mask.t().unsqueeze(-1), 0)
            scale, shift = scale_shift.chunk(2, dim=-1)
            x = x*scale+ shift
            x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
        skip = skip + skip_connection
        skip = skip.masked_fill(x_mask.t().unsqueeze(-1), 0)
    x = skip / math.sqrt(len(self.residual_layers))
    x = self.skip_conv(x)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    x = F.relu(x)
    x = self.proj(x)
    x = x.masked_fill(x_mask.t().unsqueeze(-1), 0)
    assert torch.isnan(x).any() == False
    x = rearrange(x, 't b c -> b c t')
    return x


def encode(x, n_q = 8, codec=None):
    quantized_out = torch.zeros_like(x)
    residual = x

    all_losses = []
    all_indices = []
    quantized_list = []
    layers = codec.model.quantizer.vq.layers
    n_q = n_q or len(layers)

    for layer in layers[:n_q]:
        quantized_list.append(quantized_out)
        quantized, indices, loss = layer(residual)
        residual = residual - quantized
        quantized_out = quantized_out + quantized

        all_indices.append(indices)
        all_losses.append(loss)
    quantized_list = torch.stack(quantized_list)
    out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
    return quantized_out, out_indices, out_losses, quantized_list
def rvq_ce_loss(residual_list, indices, codec, n_q=8):
    # codebook = codec.model.quantizer.vq.layers[0].codebook
    layers = codec.model.quantizer.vq.layers
    loss = 0.0
    for i,layer in enumerate(layers[:n_q]):
        residual = residual_list[i].transpose(2,1)
        embed = layer.codebook.t()
        dis = -(
            residual.pow(2).sum(2, keepdim=True)
            - 2 * residual @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        indice = indices[i, :, :]
        dis = rearrange(dis, 'b n m -> (b n) m')
        indice = rearrange(indice, 'b n -> (b n)')
        loss = loss + F.cross_entropy(dis, indice)
    return loss
# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def normalize(code):
    return code
def denormalize(code):
    return code

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
class DiffAnon(nn.Module):
    def __init__(self,
        cfg,
        rvq_cross_entropy_loss_weight = 0.1,
        diff_loss_weight = 1.0,
        f0_loss_weight = 1.0,
        duration_loss_weight = 1.0,
        ddim_sampling_eta = 0,
        min_snr_loss_weight = False,
        min_snr_gamma = 5
        ):
        super().__init__()
        # self.diff_model = Diffusion_Encoder(**cfg['diffusion_encoder'])
        self.diff_model = Diffusion_Encoder_emb(**cfg['diffusion_encoder'])
        print("diff params: ", count_parameters(self.diff_model))
        self.dim = self.diff_model.in_channels
        timesteps = cfg['train']['timesteps']

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = timesteps

        self.sampling_timesteps = cfg['train']['sampling_timesteps']
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.rvq_cross_entropy_loss_weight = rvq_cross_entropy_loss_weight
        self.diff_loss_weight = diff_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.duration_loss_weight = duration_loss_weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr)
        print("Model initialized !!!")
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, data = None, use_cond=[1,1]):
        model_output = self.diff_model(x,data, t,use_cond)
        x_start = model_output
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def model_predictions_cfg(
        self,
        x,
        t,
        data = None,
        use_cond = [1,1],
        cfg_scale = None,
        cfg_scales = None
    ):
        if cfg_scale is None and cfg_scales is None:
            model_output = self.diff_model(x, data, t, use_cond)
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)
            return ModelPrediction(pred_noise, x_start)

        if cfg_scales is None:
            x_cond = self.diff_model(x, data, t, use_cond=[0, 1])
            x_uncond = self.diff_model(x, data, t, use_cond=[0, 0]) 
            x_start =  (1 + cfg_scale) * x_cond - cfg_scale * x_uncond
        else:
            x_prosody = self.diff_model(x, data, t, use_cond=[1, 1])
            x_uncond = self.diff_model(x, data, t, use_cond=[0, 1])
            # x_spk = self.diff_model(x, data, t, use_cond=[0, 1])
            # x_start = (1+cfg_scales[1])*(x_uncond + cfg_scales[0] * (x_prosody - x_uncond)) - cfg_scales[1] * (x_spk)
            x_start = (x_uncond + cfg_scales[0] * (x_prosody - x_uncond)) 
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, data):
        preds = self.model_predictions(x, t, data)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, data):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, data=data)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, content, refer, lengths, refer_lengths, f0, uv, auto_predict_f0 = True):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths, uv)
        content, refer = self.pre_model.infer(data)
        shape = (content.shape[1], self.dim, content.shape[0])
        batch, device = shape[0], refer.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, (content,refer,lengths,refer_lengths))
            imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample(self, content,spk_emb, refer, lengths, refer_lengths, f0, auto_predict_f0 = True,use_cond=[1,1]):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths)
        # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        shape = (content.shape[0], self.dim, content.shape[2])
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, (content,f0,spk_emb,refer,lengths,refer_lengths),use_cond)

            if time_next < 0:
                img = x_start
                # imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # imgs.append(img)

        ret = img
        return ret

    @torch.no_grad()
    def ddim_sample_cfg(self, content,spk_emb, refer, lengths, refer_lengths, f0, auto_predict_f0 = True,use_cond=[1,1],cfg_scales=[0.8,0]):
        data = (content, refer, f0, 0, 0, lengths, refer_lengths)
        # content, refer = self.pre_model.infer(data,auto_predict_f0=auto_predict_f0)
        shape = (content.shape[0], self.dim, content.shape[2])
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], refer.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            #NOTE (anon) change cfg_scale to None
            pred_noise, x_start, *_ = self.model_predictions_cfg(img, time_cond, (content,f0,spk_emb,refer,lengths,refer_lengths),use_cond,cfg_scale=None,cfg_scales=cfg_scales)

            if time_next < 0:
                img = x_start
                # imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # imgs.append(img)

        ret = img
        return ret


    @torch.no_grad()
    def sample(self,
        c,spk_emb, refer, f0, lengths, refer_lengths, speechtokenizer,
        auto_predict_f0=True, sampling_timesteps=100, sample_method='ddim'
        ,use_cond=[1,1],cfg_scales=None):
        self.sampling_timesteps = sampling_timesteps
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if sample_method == 'ddpm':
            sample_fn = self.p_sample_loop
        elif sample_method == 'ddim':
            sample_fn = self.ddim_sample
        elif sample_method == 'ddim_cfg':
            sample_fn = self.ddim_sample_cfg

        if sample_method == 'ddim_cfg':
            predicted_st = sample_fn(c,spk_emb, refer, lengths, refer_lengths, f0, auto_predict_f0,use_cond,cfg_scales=cfg_scales)
        else:
            predicted_st = sample_fn(c,spk_emb, refer, lengths, refer_lengths, f0, auto_predict_f0,use_cond)

        speechtokenizer.to(predicted_st.device)
        print(f"{predicted_st.shape}")
        audio = speechtokenizer.decode_first(predicted_st).squeeze(0)
        # audio = denormalize(audio)
        # vocos.to(audio.device)
        # audio = vocos.decode(audio)

        if audio.ndim == 3:
            audio = rearrange(audio, 'b 1 n -> b n')

        return audio 

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, data, vocos):
        c_padded, refer_padded, f0_padded, spec_padded, \
        wav_padded, spk_emb, lengths, refer_lengths = data
        
        b, d, n, device = *spec_padded.shape, spec_padded.device

        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, spec_padded.size(2)), 1).to(spec_padded.dtype)
        x_start = normalize(spec_padded)*x_mask

        # get pre model outputs
        # content, refer, lf0, lf0_pred = self.pre_model(data)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # it just samples a discrete time step
        noise = torch.randn_like(x_start)*x_mask
        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        # predict and take gradient step
        model_out = self.diff_model(x,(c_padded,f0_padded, spk_emb, refer_padded,lengths,refer_lengths), t,[None,None])
        target = x_start

        loss = F.mse_loss(model_out, target, reduction = 'none')
        if torch.isnan(loss).any():
            print("Loss is Nan!!!")
        loss_diff = reduce(loss, 'b ... -> b (...)', 'mean')
        loss_diff = loss_diff * extract(self.loss_weight, t, loss.shape)
        loss_diff = loss_diff.mean()

        # loss_f0 = F.l1_loss(lf0_pred, lf0)
        # loss = loss_diff + loss_f0
        loss = loss_diff

        # cross entropy loss to codebooks
        # _, indices, _, quantized_list = encode(codes_padded,8,codec)
        # ce_loss = rvq_ce_loss(denormalize(model_out.unsqueeze(0))-quantized_list, indices, codec)
        # loss = loss + 0.1 * ce_loss

        return loss, loss_diff,model_out, target
def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        if torch.isnan(p.grad.data).any():
            print("Nan in gradients !!!")
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2) 
    return total_norm
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
class Trainer(object):
    def __init__(
        self,
        cfg_path = './config.json',
        resume_dir = None,
        resume_milestone = None,
    ):
        super().__init__()

        st_config_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/config.json"    
        st_ckpt_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/ckpt.dev"   



        self.cfg = json.load(open(cfg_path))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        device = self.accelerator.device

        # model
        print("Loading Vocos...")
        # self.vocos = Vocos.from_pretrained("/home/anon/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21")
        print("Vocos loaded!!!")

        print("Loading SpeechTokenizer...")

        st_config_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/config.json"    
        st_ckpt_path = "diffanon_repo/SpeechTokenizer/pretrained_models/speechtokenizer/ckpt.dev"   

        self.speechtokenizer = SpeechTokenizer.load_from_checkpoint(st_config_path,st_ckpt_path)

        # self.speechtokenizer = self.speechtokenizer.to(device)
        self.speechtokenizer.eval()
        print("SpeechTokenizer is loaded!!!")




        self.model = DiffAnon(cfg=self.cfg).to(device)
        # sampling and training hyperparameters

        self.save_and_sample_every = self.cfg['train']['save_and_sample_every']

        self.batch_size = self.cfg['train']['train_batch_size']
        self.gradient_accumulate_every = self.cfg['train']['gradient_accumulate_every']

        self.train_num_steps = self.cfg['train']['train_num_steps']

        # dataset and dataloader
        collate_fn = TextAudioCollate()
        ds = NS2VCDataset(self.cfg, self.speechtokenizer)
        print("Dataset loaded !!!")
        self.ds = ds
        dl = DataLoader(ds, batch_size = self.cfg['train']['train_batch_size'], shuffle = True, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)

        print("Preparing accelerator...")
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        print("Accelerator initialized!!!")
        self.eval_dl = DataLoader(ds, batch_size = 1, shuffle = False, pin_memory = True, num_workers = self.cfg['train']['num_workers'], collate_fn = collate_fn)
        # print(1)
        # optimizer

        self.opt = AdamW(self.model.parameters(), lr = self.cfg['train']['train_lr'], betas = self.cfg['train']['adam_betas'])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = self.cfg['train']['ema_decay'], update_every = self.cfg['train']['ema_update_every'])
            self.ema.to(self.device)

        if resume_milestone is not None and resume_dir is None:
            raise ValueError("resume_dir must be set when resume_milestone is provided")

        if resume_dir is not None:
            self.logs_folder = Path(resume_dir)
        else:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
        self.logs_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        if resume_milestone is not None:
            resume_path = self.logs_folder / f"model-{resume_milestone}.pt"
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            self.load(resume_path)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(model_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        # print(1)
        accelerator = self.accelerator
        device = accelerator.device
        torch.autograd.set_detect_anomaly(True)
        if accelerator.is_main_process:
            logger = utils.get_logger(self.logs_folder)
            writer = SummaryWriter(log_dir=self.logs_folder)
            writer_eval = SummaryWriter(log_dir=os.path.join(self.logs_folder, "eval"))

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = [d.to(device) for d in data]

                    with self.accelerator.autocast():
                        loss, loss_diff, \
                        pred, target = self.model(data, self.speechtokenizer)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                grad_norm = get_grad_norm(self.model)
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                # print(loss_diff, loss_f0, ce_loss)
############################logging#############################################
                if accelerator.is_main_process and self.step % 100 == 0:
                    logger.info('Train Epoch: {} [{:.0f}%]'.format(
                        self.step//len(self.ds),
                        100. * self.step / self.train_num_steps))
                    logger.info(f"Losses: {[loss_diff]}, step: {self.step}")

                    scalar_dict = {"loss/diff": loss_diff, "loss/all": total_loss,
                                "loss/grad": grad_norm}
                    image_dict = {
                        "all/spec": plot_spectrogram_to_numpy(target[0, :, :].detach().unsqueeze(-1).cpu()),
                        "all/spec_pred": plot_spectrogram_to_numpy(pred[0, :, :].detach().unsqueeze(-1).cpu()),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict
                    )

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        c_padded, refer_padded, f0_padded, spec_padded, wav_padded,spk_embed, lengths, refer_lengths = next(iter(self.eval_dl))
                        c, refer, f0 = c_padded.to(device), refer_padded.to(device), f0_padded.to(device) 
                        lengths, refer_lengths = lengths.to(device), refer_lengths.to(device)
                        spk_embed = spk_embed.to(device)
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            samples = self.ema.ema_model.sample(c,spk_embed, refer, f0, lengths, refer_lengths, self.speechtokenizer).detach().cpu()
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), samples, 16000)
                        audio_dict = {}
                        audio_dict.update({
                                f"gen/audio": samples,
                                f"gt/audio": wav_padded[0]
                            })
                        utils.summarize(
                            writer=writer_eval,
                            global_step=self.step,
                            audios=audio_dict,
                            audio_sampling_rate=24000
                        )
                        utils.clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=5, sort_by_time=True)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
