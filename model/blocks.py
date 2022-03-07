import torch
from torch import nn

def pad_layer(dim, padding_mode='reflect'):
    if padding_mode == 'reflect':
        return nn.ReflectionPad2d if dim == 2 else nn.ReflectionPad3d
    elif padding_mode == 'replicate':
        return nn.ReplicationPad2d if dim == 2 else nn.ReplicationPad3d
    else:
        raise NotImplementedError(f'padding {padding_mode} is not implemented')

def conv_layer(dim):
    if dim == 1:
        return nn.Conv1d
    elif dim == 2:
        return nn.Conv2d
    elif dim == 3:
        return nn.Conv3d
    else:
        raise NotImplementedError(f'{dim} dimensional convolution is not implemented')

def norm_layer(dim):
    if dim == 2:
        return nn.BatchNorm2d
    elif dim == 3:
        return nn.BatchNorm3d
    else:
        raise NotImplementedError(f'{dim} dimensional normalization is not implemented')

def pool_layer(dim, pool_mode='avg'):
    if pool_mode == 'avg':
        return nn.AvgPool2d if dim == 2 else nn.AvgPool3d
    elif pool_mode == 'max':
        return nn.MaxPool2d if dim == 2 else nn.MaxPool3d
    else:
        raise NotImplementedError(f'pool {pool_mode} is not implemented')

def activation_layer(activation='ReLU'):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'Tanh':
        return nn.Tanh()
    elif activation == 'LeakyReLU':
        return nn.LeakyRELU(0.2, inplace=True)
    elif activation == 'GELU':
        return nn.GELU()
