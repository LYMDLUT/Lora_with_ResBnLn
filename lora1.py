import torch
from torch import nn
import math
import weakref
import re

from typing import Dict, Union

def low_rank_approximate(weight, rank, clamp_quantile=0.99):
    if len(weight.shape) == 4: # conv
        weight = weight.flatten(1)
        out_ch, in_ch, k1, k2 = weight.shape

    U, S, Vh = torch.linalg.svd(weight)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    if len(weight.shape) == 4:
        # U is (out_channels, rank) with 1x1 conv.
        U = U.reshape(U.shape[0], U.shape[1], 1, 1)
        # V is (rank, in_channels * kernel_size1 * kernel_size2)
        Vh = Vh.reshape(Vh.shape[0], in_ch, k1, k2)
    return U, Vh

class LoraBlock(nn.Module):
    def __init__(self, host:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, scale=1.0):
        super().__init__()
        self.host = weakref.ref(host)
        host.lora_block = self

        self.rank = rank
        self.dropout = dropout

        if isinstance(host, nn.Linear):
            self.host_type = 'linear'
            self.lora_down = nn.Linear(host.in_features, rank, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.lora_up = nn.Linear(rank, host.out_features, bias=False)
            self.register_buffer('scale', torch.tensor(scale))
        elif isinstance(host, nn.Conv2d):
            self.host_type = 'conv'
            self.lora_down = nn.Conv2d(host.in_channels, rank, kernel_size=host.kernel_size, stride=host.stride,
                                        padding=host.padding, dilation=host.dilation, groups=host.groups, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.lora_up = nn.Conv2d(rank, host.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.register_buffer('scale', torch.tensor(scale))
        else:
            raise NotImplementedError('lora support only Linear and Conv2d now.')

        self.lora_hook_handle = host.register_forward_hook(self.lora_forward_hook)

    def init_weights(self, sdv_init=False):
        host = self.host()
        if sdv_init:
            U, V = low_rank_approximate(host.weight, self.rank)
            self.lora_up.weight.data = U.to(device=host.weight.device, dtype=host.weight.dtype)
            self.lora_down.weight.data = V.to(device=host.weight.device, dtype=host.weight.dtype)
        else:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def lora_forward_hook(self, host, fea_in, fea_out):
        return fea_out + self.dropout(self.lora_up(self.lora_down(fea_in[0]))) * self.scale

    def remove(self):
        host = self.host()
        self.lora_hook_handle.remove()
        del host.lora_block

    def collapse_to_host(self, alpha=None):
        if alpha is None:
            alpha = self.scale

        host = self.host()
        if self.host_type == 'linear':
            host.weight = nn.Parameter(
                host.weight.data +
                alpha * (self.lora_up.weight.data @ self.lora_down.weight.data)
                .to(host.weight.device, dtype=host.weight.dtype)
            )
        elif self.host_type == 'conv':
            host.weight = nn.Parameter(
                host.weight.data +
                alpha * (self.lora_up.weight.data.flatten(1) @ self.lora_down.weight.data.flatten(1))
                .reshape(host.weight.data.shape).to(host.weight.device, dtype=host.weight.dtype)
            )


def extract_lora_state(model:nn.Module):
    return {k: v for k, v in model.state_dict().items() if 'lora_block.' in k}

def collapse_lora(model: nn.Module, remove_lora=True):
    # default_alpha = cfg['default_alpha']
    # lora_cfg: Dict = {item: group['alpha'] for group in cfg['groups'] for item in group['layers']}
    for name, block in model.named_modules():
        if isinstance(block, LoraBlock):
            block.collapse_to_host()
            if remove_lora:
                block.remove()

def warp_layer_with_lora(layer:Union[nn.Linear, nn.Conv2d], rank, dropout=0.1, scale=1.0, sdv_init=False):
    lora_block = LoraBlock(layer, rank, dropout, scale)
    lora_block.init_weights(sdv_init)
    return lora_block

def warp_with_lora(model: nn.Module, rank, dropout=0.1, scale=1.0, sdv_init=False):
    named_modules = {k: v for k, v in model.named_modules() if 'qkv' not in k}
    all_layers = list(named_modules.keys())
    lora_layers = nn.ModuleList()
    for name in all_layers:
        if isinstance(named_modules[name], nn.Linear) or isinstance(named_modules[name], nn.Conv2d):
            lora_layers.append(warp_layer_with_lora(named_modules[name], rank, dropout, scale, sdv_init))
    return lora_layers


def get_match_layers(layers, all_layers):
    res = []
    for name in layers:
        if name.startswith('re:'):
            pattern = re.compile(name[3:])
            res.extend(filter(lambda x: pattern.match(x) != None, all_layers))
        else:
            res.append(name)
    return set(res)