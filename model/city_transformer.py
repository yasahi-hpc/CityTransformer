"""
Version 0: Resnet + Transformer (attention on time), using the ground plume concentration only
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, t, s*c) time series data of measurement, s = stations, c = (z, z, 1, 1)

Output:
    imgs: (b, 2, h, w) concentration + binary-map

Version 1: Resnet + MLP, using the ground plume concentration only
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, 1, s*c) time averaged data of measurement, s = stations, c = (z, z, 1, 1)

Output:
    imgs: (b, 2, h, w) concentration + binary-map
"""

import torch
import math
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from .blocks import pad_layer, conv_layer, norm_layer, pool_layer, activation_layer

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dim=2,
                 padding_mode='reflect', activation='ReLU'):
        super().__init__()
        layers = []

        p = int(np.ceil((kernel_size-1.0)/2))
        if padding_mode in ['reflect', 'replicate']:
            layers += [pad_layer(dim=dim, padding_mode=padding_mode)(p)]
            p = 0
        elif padding_mode == 'zeros':
            pass
        else:
            raise NotImplementedError(f'padding {padding_mode} is not implemented')

        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dim=2, activation='ReLU'):
        super().__init__()
        layers = []

        p = int(np.ceil((kernel_size-1.0)/2))
        layers += [nn.Upsample(scale_factor=stride)]
        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dim=2,
                 padding_mode='reflect', activation='ReLU'):
                        
        super().__init__()
        layers = []

        assert dim == 2 or dim == 3
                                             
        p = int(np.ceil((kernel_size-1.0)/2))
        if padding_mode in ['reflect', 'replicate']:
            layers += [pad_layer(dim=dim, padding_mode=padding_mode)(p)]
            p = 0
        elif padding_mode == 'zeros':
            pass
        else:
            raise NotImplementedError(f'padding {padding_mode} is not implemented')
                                                                                        
        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)
                                                                                                                                                                                  
    def forward(self, x):
        return self.model(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dim=2,
                 activation='ReLU'):
                       
        super().__init__()
        layers = []
                                    
        p = int(np.ceil((kernel_size-1.0)/2))
        layers += [nn.Upsample(scale_factor=stride)]
        layers += [conv_layer(dim=dim)(in_channels, out_channels, kernel_size=kernel_size, padding=p)]
        layers += [norm_layer(dim=dim)(out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)
                                                        
    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dim=2, padding_mode='reflect', dropout=0.):
        super().__init__()
        layers = []
        layers += [ConvBlock(in_channels, in_channels, kernel_size=3, dim=dim, padding_mode=padding_mode)]
        if dropout > 0.:
            layers += [nn.Dropout(dropout)]
        layers += [ConvBlock(in_channels, in_channels, kernel_size=3, dim=dim,  padding_mode=padding_mode)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.model(x) # Skip connection
        return out

class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU', skip_connection=False):
        super().__init__()
        layers = []

        layers += [nn.Linear(in_channels, out_channels)]
        layers += [activation_layer(activation=activation)]

        self.model = nn.Sequential(*layers)
        self.skip_connection = skip_connection 

        if in_channels != out_channels:
            self.skip_connection = False

    def forward(self, x):
        if self.skip_connection:
            return x + self.model(x)
        else:
            return self.model(x)

class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        allowed_kwargs = {
                          'd_model',
                          'd_hidden',
                          'nhead',
                          'dim_feedforward',
                          'dropout',
                          'num_layers',
                          'num_channels',
                          'Nt',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        d_model = kwargs.get('d_model', 256)
        d_hidden = kwargs.get('d_hidden', 128)
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 64)
        dropout = kwargs.get('dropout', 0)
        Nt = kwargs.get('Nt', 16)
        num_layers = kwargs.get('num_layers', 6)
        num_channels = kwargs.get('num_channels')
        if num_channels is None:
            raise ValueError('Argument num_channels must be given for TransformerBlock')
        
        # Define embedding
        # Transpose and apply linear layer
        # 1. (b t s c) -> (b t (s c))
        # 2. (b t (s c)) -> (b t d)
        self.series_embedding = nn.Sequential(
            Rearrange('b t s c -> b t (s c)'),
            nn.Linear(num_channels, d_model),
        )
        self.pos_embedding = nn.Parameter( torch.randn(1, Nt, d_model) )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

        # Define Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Define decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
        )

    def forward(self, series):
        """
        series (b, t, s, c)
        """

        # Embedding data
        out = self.series_embedding(series) * self.scale
        out += self.pos_embedding
        out = self.dropout(out)

        # Transformer on time series data
        out = self.transformer_encoder(out)

        # Decode
        out = self.decoder(out)
        return out

class MergeBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        allowed_kwargs = {
                          'in_channels',
                          'h',
                          'w',
                          'hidden_imgs_out',
                          'hidden_series_in',
                          'hidden_series_out',
                          'n_layers',
                          'activation',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        in_channels  = kwargs.get('in_channels', 1024)
        h = kwargs.get('h', 8)
        w = kwargs.get('w', 8)
        hidden_imgs_in  = in_channels * h * w
        hidden_imgs_out = kwargs.get('hidden_imgs_out', 1024)
        hidden_series_in  = kwargs.get('hidden_series_in', 16*128)
        hidden_series_out = kwargs.get('hidden_imgs_out', 1024)
        n_layers          = kwargs.get('n_layers', 2)
        activation        = kwargs.get('activation', 'ReLU')
        self.flatten_imgs = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
        )
        self.flatten_series = nn.Sequential(
            Rearrange('b t d -> b (t d)'),
        )

        self.merge = nn.Sequential(
            nn.Linear(hidden_imgs_in+hidden_series_in, hidden_imgs_in),
            activation_layer(activation=activation),
            Rearrange('b (c h w) -> b c h w', c=in_channels, h=h, w=w),
        )

    def forward(self, imgs, series):
        """
        """
        imgs = self.flatten_imgs(imgs)
        series = self.flatten_series(series)

        return self.merge(torch.cat([imgs, series], dim=1))

class MLPEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        allowed_kwargs = {
                          'dropout',
                          'num_layers',
                          'in_channels',
                          'hidden_dims',
                          'out_channels',
                          'activation',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        dropout = kwargs.get('dropout', 0)
        num_layers = kwargs.get('num_layers', 1)
        in_channels = kwargs.get('in_channels', 128)
        hidden_dims = kwargs.get('hidden_dims', 128)
        out_channels = kwargs.get('out_channels', 128)
        activation = kwargs.get('activation', 'ReLU')

        layers = []
        layers += [Rearrange('b t s c -> b (t s c)')]
        layers += [MLPBlock(in_channels, hidden_dims, activation)]
        for _ in range(num_layers):
            layers += [MLPBlock(hidden_dims, hidden_dims, activation, True)] # With skip connection
        layers += [MLPBlock(hidden_dims, out_channels, activation)]
        layers += [Rearrange('b (c d) -> b c d', c=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        (b, t, s, c)
        """
        return self.model(x)

class FrontEnd(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('in_channels', 2)
        out_channels = kwargs.get('out_channels', 1)
        hidden_dim = kwargs.get('hidden_dim', 16)
        hidden_dim_transformer = kwargs.get('hidden_dim_transformer', 128)
        dim = kwargs.get('dim', 2)
        padding_mode = kwargs.get('padding_mode', 'zeros')
        activation = kwargs.get('activation', 'ReLU')
        n_layers_cnn = kwargs.get('n_layers_cnn', 6)
        #n_resnet_blocks = kwargs.get('n_resnet_blocks', 9)
        n_layers_transformer = kwargs.get('n_layers_transformer', 6)
        n_layers_merge = kwargs.get('n_layers_merge', 2)
        d_model = kwargs.get('d_model', 256)
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 64)
        dropout = kwargs.get('dropout', 0)
        n_stations = kwargs.get('n_stations', 14)
        Nt = kwargs.get('Nt', 16)
        Nx = kwargs.get('Nx', 256)
        Ny = kwargs.get('Ny', 256)
        Nz = kwargs.get('Nz', 16)
        version = kwargs.get('version', 0)
        UNet = kwargs.get('UNet', False)
        
        self.version = version

        # Imgs encoder block
        layers = []

        ### downsample
        for i in range(n_layers_cnn):
            mult = 2**i
            layers += [ConvBlock(hidden_dim * mult, hidden_dim * mult * 2, kernel_size=3, stride=2, dim=dim, padding_mode=padding_mode)]

        self.imgs_downsample = nn.Sequential(*layers)

        # These sublayers are created for UNet model only
        num_channels = n_stations * (Nz * 2 + 2)
        if UNet:
            # Series encoder block

            if self.version == 0:
                # Attention on time
                # Input: (b, t, c)
                self.series_encoder = TransformerBlock(d_model=d_model, d_hidden=hidden_dim_transformer, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, num_layers=n_layers_transformer, Nt=Nt, num_channels=num_channels)
                hidden_series_in = Nt * hidden_dim_transformer
                hidden_series_out = Nt * hidden_dim_transformer
            elif self.version == 1:
                # Input: (b, 1, c)
                Nt = 1
                self.series_encoder = MLPEncoder(in_channels=num_channels, hidden_dims=hidden_dim_transformer, out_channels=hidden_dim_transformer)
                hidden_series_in = hidden_dim_transformer
                hidden_series_out = hidden_dim_transformer

            # Merge block
            in_channels = hidden_dim * 2**n_layers_cnn
            h = Ny // 2**n_layers_cnn
            w = Nx // 2**n_layers_cnn
            
            hidden_imgs_out = in_channels
            self.merge = MergeBlock(in_channels=in_channels, h=h, w=w, hidden_imgs_out=hidden_imgs_out,
                                    hidden_series_in=hidden_series_in, hidden_series_out=hidden_series_out,
                                    n_layers=n_layers_merge)

    def forward(self, imgs):
        encoded_imgs = self.imgs_downsample(imgs)

        return encoded_imgs

class ConnectionLayers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('in_channels', 2)
        out_channels = kwargs.get('out_channels', 1)
        hidden_dim = kwargs.get('hidden_dim', 16)
        hidden_dim_transformer = kwargs.get('hidden_dim_transformer', 128)
        dim = kwargs.get('dim', 2)
        padding_mode = kwargs.get('padding_mode', 'zeros')
        activation = kwargs.get('activation', 'ReLU')
        n_layers_cnn = kwargs.get('n_layers_cnn', 6)
        n_resnet_blocks = kwargs.get('n_resnet_blocks', 9)
        n_layers_transformer = kwargs.get('n_layers_transformer', 6)
        n_layers_merge = kwargs.get('n_layers_merge', 2)
        d_model = kwargs.get('d_model', 256)
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 64)
        dropout = kwargs.get('dropout', 0)
        n_stations = kwargs.get('n_stations', 14)
        Nt = kwargs.get('Nt', 16)
        Nx = kwargs.get('Nx', 256)
        Ny = kwargs.get('Ny', 256)
        Nz = kwargs.get('Nz', 16)
        version = kwargs.get('version', 0)
        
        self.version = version

        # Imgs encoder block
        layers = []

        ### Resnet blocks
        mult = 2**n_layers_cnn
        for i in range(n_resnet_blocks):
            layers += [ResnetBlock(in_channels=hidden_dim * mult, dim=dim, padding_mode=padding_mode, dropout=0.)]

        self.imgs_encoder = nn.Sequential(*layers)

        ### Upsample, the output of this layer is fed to local enhancer
        layers = []
        for i in range(n_layers_cnn):
            mult = 2**(n_layers_cnn - i)
            layers += [DeconvBlock(hidden_dim * mult, int(hidden_dim * mult / 2), kernel_size=3, stride=2, dim=dim)]

        self.imgs_upsample = nn.Sequential(*layers)

        # Series encoder block
        num_channels = n_stations * (Nz * 2 + 2)
        if self.version == 0:
            # Attention on time
            # Input: (b, t, c)
            self.series_encoder = TransformerBlock(d_model=d_model, d_hidden=hidden_dim_transformer, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, num_layers=n_layers_transformer, Nt=Nt, num_channels=num_channels)
            hidden_series_in = Nt * hidden_dim_transformer
            hidden_series_out = Nt * hidden_dim_transformer
        elif self.version == 1:
            # Input: (b, 1, c)
            Nt = 1

            self.series_encoder = MLPEncoder(in_channels=num_channels, hidden_dims=hidden_dim_transformer, out_channels=hidden_dim_transformer)
            hidden_series_in = hidden_dim_transformer
            hidden_series_out = hidden_dim_transformer

        # Merge block
        in_channels = hidden_dim * 2**n_layers_cnn
        h = Ny // 2**n_layers_cnn
        w = Nx // 2**n_layers_cnn
        
        hidden_imgs_out = in_channels
        self.merge = MergeBlock(in_channels=in_channels, h=h, w=w, hidden_imgs_out=hidden_imgs_out,
                                hidden_series_in=hidden_series_in, hidden_series_out=hidden_series_out,
                                n_layers=n_layers_merge)

    def forward(self, imgs, series):
        encoded_imgs = self.imgs_encoder(imgs)
        encoded_series = self.series_encoder(series)
        encoded_imgs = self.merge(encoded_imgs, encoded_series)
        encoded_imgs = self.imgs_upsample(encoded_imgs)

        return encoded_imgs

class BackEnd(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        out_channels = kwargs.get('out_channels', 2) #map and prediction
        hidden_dim = kwargs.get('hidden_dim', 16)
        dim = kwargs.get('dim', 2)
        padding_mode = kwargs.get('padding_mode', 'zeros')
        layers = []

        ### 1x1 convolution to generate flows
        layers += [ConvBlock(hidden_dim, out_channels, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode, activation='Tanh')]

        self.imgs_decoder = nn.Sequential(*layers)

    def forward(self, imgs):
        decoded_imgs = self.imgs_decoder(imgs)

        return decoded_imgs

class CityTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        allowed_kwargs = {
                          'in_channels',
                          'out_channels',
                          'hidden_dim',
                          'hidden_dim_transformer',
                          'dim',
                          'padding_mode',
                          'activation',
                          'n_layers',
                          'n_resnet_blocks_low',
                          'n_resnet_blocks_high',
                          'n_layers_merge',
                          'n_layers_cnn',
                          'n_layers_transformer',
                          'n_precision_enhancers',
                          'd_model',
                          'version',
                          'nhead',
                          'dim_feedforward',
                          'dropout',
                          'n_stations',
                          'num_series_channels',
                          'Nt',
                          'Nx',
                          'Ny',
                          'Nz',
                          'super_precision',
                          'UNet',
                         }
         
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        kwargs_low_precision = kwargs.copy()
        unused_keys = {'super_precision', 'n_precision_enhancers', 'use_decoder', 'UNet'}
        for key in unused_keys:
            kwargs_low_precision.pop(key, None)

        in_channels = kwargs.get('in_channels', 2)
        out_channels = kwargs.get('out_channels', 2)
        dim = kwargs.get('dim', 2)
        padding_mode = kwargs.get('padding_mode', 'zeros')
        activation = kwargs.get('activation', 'ReLU')
        hidden_dim = kwargs.get('hidden_dim', 16)
        version = kwargs.get('version', 0)
        n_resnet_blocks_high = kwargs.get('n_resnet_blocks_high', 2)

        self.version = version
        self.n_layers = kwargs.get('n_layers', 6)
        self.UNet = kwargs.get('UNet', False)

        # Version 0 to 1 supported
        assert self.version in [0,1]

        ## Construct model
        if self.UNet:
            # U-Net blocks
            self.unet_encoder0 = UNetEncoderBlock(in_channels,  hidden_dim, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode)
            self.unet_decoder0 = UNetEncoderBlock(hidden_dim*2, out_channels, kernel_size=7, stride=1, dim=dim, padding_mode='zeros', activation='Tanh')
            
            for i in range(1, self.n_layers):
                # Encoder layers
                mult = hidden_dim * 2**(i-1)
                setattr(self, f'unet_encoder{i}', UNetEncoderBlock(mult, mult*2, kernel_size=3, stride=2, dim=dim, padding_mode=padding_mode))
            
                # Decoder layers
                final_layer = (i == self.n_layers-1)
                in_channels_ = mult*2 if final_layer else mult*4
                setattr(self, f'unet_decoder{i}', UNetDecoderBlock(in_channels_, mult, kernel_size=3, stride=2, dim=dim))

            # Series encoder block and merge block
            kwargs_low_precision['n_layers_cnn'] = self.n_layers - 1
            kwargs_low_precision['UNet'] = True
            self.front_end = FrontEnd(**kwargs_low_precision)

            self.series_encoder = self.front_end.series_encoder
            self.merge = self.front_end.merge

        else:
            # Res-Net architecture
            layers = []
            layers += [ConvBlock(in_channels, hidden_dim, kernel_size=7, stride=1, dim=dim, padding_mode=padding_mode)]

            self.encode = nn.Sequential(*layers)

            self.downsample = FrontEnd(**kwargs_low_precision)
            self.upsample = ConnectionLayers(**kwargs_low_precision)
            self.final = BackEnd(**kwargs_low_precision)

    def forward(self, imgs, series):
        if self.UNet:
            return self._forward_UNet(imgs, series)
        else:
            return self._forward(imgs, series)

    def _forward(self, imgs, series):
        out = self.encode(imgs)
        out = self.downsample(out)
        out = self.upsample(out, series)
        out = self.final(out)

        return out

    def _forward_UNet(self, imgs, series):
        ### Encoding
        encoded = []
        out = imgs

        for i in range(self.n_layers):
            unet_encoder = getattr(self, f'unet_encoder{i}')
            out = unet_encoder(out)
            encoded.append(out)

        encoded_series = self.series_encoder(series)

        # Merge encoded series and img data
        out = self.merge(out, encoded_series)

        ### Decoding
        for i in reversed(range(self.n_layers)):
            unet_decoder = getattr(self, f'unet_decoder{i}')
            if i == self.n_layers-1:
                out = unet_decoder(out)
            else:
                out = unet_decoder(torch.cat([out, encoded[i]], dim=1))
            
        return out
