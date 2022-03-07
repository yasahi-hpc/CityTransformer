"""
Version 0: Resnet + Transformer (attention on time)
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, c, t, z) time series data of measurements
Output:
    imgs (w/o sp): (b, 2, h, w) concentration + zero_map
    imgs (w/ sp):  (b, p, h, w) concentration (multiple-levels) + zero_map 

Version 1: Resnet + Transformer (attention on z)
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, c, 1, z) time averaged data of measurements
Output:
    imgs (w/o sp): (b, 2, h, w) concentration + zero_map
    imgs (w/ sp):  (b, p, h, w) concentration (multiple-levels) + zero_map 

Version 2: Resnet + Transformer (attention on time, at z = 0)
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, c, t, 1) time series data of measurements
Output:
    imgs (w/o sp): (b, 2, h, w) concentration + zero_map
    imgs (w/ sp):  (b, p, h, w) concentration (multiple-levels) + zero_map 

Version 3: Resnet + MLP (on z)
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, c, 1, z) time averaged data of measurements
Output:
    imgs (w/o sp): (b, 2, h, w) concentration + zero_map
    imgs (w/  sp): (b, p, h, w) concentration (multiple-levels) + zero_map

Version 4: Resnet + MLP (on z==0)
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, c, 1, 1) time averaged data of measurements at z = 0
Output:
    imgs (w/o sp): (b, 2, h, w) concentration + zero_map
    imgs (w/  sp): (b, p, h, w) concentration (multiple-levels) + zero_map
"""

import xarray as xr
import numpy as np
import torch
import pathlib

class CityDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, **kwargs):
        allowed_kwargs = {
                          'device',
                          'version',
                          'inference_mode',
                          'n_digits',
                          'nz',
                          'n_stations',
                          'n_precision_enhancers',
                          'super_precision',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        device = kwargs.get('device', 'cpu')
        version = kwargs.get('version', 0)
        n_digits = kwargs.get('n_digits', 3)
        nz = kwargs.get('nz', 100)
        n_precision_enhancers = kwargs.get('n_precision_enhancers', 2)
        inference_mode = kwargs.get('inference_mode', False)
        super_precision = kwargs.get('super_precision', False)

        self.files = sorted(list(pathlib.Path(path).glob('shot*.nc')))
        self.datanum = len(self.files)
        self.version = version
        self.inference_mode = inference_mode
        self.n_digits = n_digits
        self.nz = nz
        self.n_precision_enhancers = n_precision_enhancers
        self.super_precision = super_precision

        # Version 0 to 4 supported
        assert self.version in [0,1,2,3,4]

        # read datashape
        ds = xr.open_dataset(self.files[0], engine='netcdf4')

        # Interpolate along z direction
        ds = ds.isel(z=slice(0, self.nz))
        self.station_positions = ds['station_position'].values
        num_stations, Nt, Nz = ds['u'].shape
        Ny, Nx = ds['sdf'].shape

        self.resolution_dict = {}
        self.resolution_dict['Nx'] = Nx
        self.resolution_dict['Ny'] = Ny
        self.resolution_dict['Nz'] = Nz
        self.resolution_dict['Nt'] = Nt
        self.resolution_dict['num_stations'] = num_stations * 4

        # Normalization coefficients
        try:
            ds_stats = xr.open_dataset('stats.nc', engine='netcdf4')
        except:
            raise FileNotFoundError('stats.nc storing normalization coefficients not found')

        # Track min, max of output values
        to_tensor = lambda variable: torch.tensor(variable).view(1,-1,1,1).float().to(device)
        self.norm_dict = {}
        for key, value in ds_stats.data_vars.items():
            self.norm_dict[key] = to_tensor(value.values)

        for key, value in ds_stats.attrs.items():
            self.norm_dict[key] = to_tensor(value)

        # Reform series data (4) -> (4, num_stations) -> (1, 4*num_stations, 1, 1)
        self.norm_dict['series_max'] = to_tensor(np.tile(ds_stats['series_max'], (num_stations, 1)).T.flatten())
        self.norm_dict['series_min'] = to_tensor(np.tile(ds_stats['series_min'], (num_stations, 1)).T.flatten())

        clip_digits = -1
        self.norm_dict['concentrations_max'] = to_tensor(ds_stats['log_concentration_multi_digits_max'].isel(num_digits=clip_digits).values)
        self.norm_dict['concentrations_min'] = to_tensor(ds_stats['log_concentration_multi_digits_min'].isel(num_digits=clip_digits).values)

        self.norm_dict['release_sdf_max'] = to_tensor(ds_stats['imgs_max'].values[0])
        self.norm_dict['release_sdf_min'] = to_tensor(ds_stats['imgs_min'].values[0])

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        ds = xr.open_dataset(self.files[idx], engine='netcdf4')
        ds = ds.isel(z=slice(0, self.nz))

        # Input imgs: sdf and release_point
        release_point = ds['release_point'].values
        sdf = ds['sdf'].values
        imgs = torch.tensor(np.stack([release_point, sdf], axis=0)).float()

        # Input series: u, v, scalar, log_scalar (Note: n_channels = num_stations x channels)
        u = ds['u'].values
        v = ds['v'].values
        scalar = ds['scalar'].values
        log_scalar = ds['log_scalar'].values

        time_average = lambda var, keepdims: np.mean(var, axis=1, keepdims=keepdims)
        if self.version == 0:
            # time series data
            # (stations, nt, nz)
            pass

        elif self.version in [1, 3]:
            # targeting time averaged data
            # (stations, 1, nz)
            u = time_average(u, True)
            v = time_average(v, True)
            scalar = time_average(scalar, True)
            log_scalar = time_average(log_scalar, True)

        elif self.version in [2]:
            # time series data at z==0
            # (stations, nt, 1)
            u = np.expand_dims(u[:, :, 0], axis=2)
            v = np.expand_dims(v[:, :, 0], axis=2)
            scalar = np.expand_dims(scalar[:, :, 0], axis=2)
            log_scalar = np.expand_dims(log_scalar[:, :, 0], axis=2)

        elif self.version in [4]:
            # targeting time averaged data at z==0
            # (stations, 1, 1)
            u = np.expand_dims(time_average(u, False)[:, 0], axis=(1, 2))
            v = np.expand_dims(time_average(v, False)[:, 0], axis=(1, 2))
            scalar = np.expand_dims(time_average(scalar, False)[:, 0], axis=(1, 2))
            log_scalar = np.expand_dims(time_average(log_scalar, False)[:, 0], axis=(1, 2))

        series = torch.tensor(np.concatenate([u, v, scalar, log_scalar], axis=0)).float()

        # Output imgs: concentration
        # Release points
        #release_points = [ds.attrs['release_x'], ds.attrs['release_y']]
        #flow_directions = [ds.attrs['v0'], ds.attrs['theta0']]
        release_points = torch.tensor([ds.attrs['release_x'], ds.attrs['release_y']]).float()
        flow_directions = torch.tensor([ds.attrs['v0'], ds.attrs['theta0']]).float()

        zeros_map = np.expand_dims(ds['zeros_map'].values, axis=0)
        if self.super_precision:
            # predict the summation recursively
            out = []
            for i in range(self.n_precision_enhancers+1):
                concentration_multi_digits = ds['log_concentration_multi_digits'].isel(digits=self.n_digits+i).values
                out.append(  np.expand_dims(concentration_multi_digits, axis=0) )

            out.append( zeros_map )
            out_imgs = torch.tensor(np.concatenate(out, axis=0)).float()

        else:
            # This stores the summation from 0 to self.digits
            concentration_multi_digits = ds['log_concentration_multi_digits'].isel(digits=self.n_digits).values
            concentration_multi_digits = np.expand_dims(concentration_multi_digits, axis=0)
            out_imgs = torch.tensor(np.concatenate([concentration_multi_digits, zeros_map], axis=0)).float()

        if self.inference_mode:
            return idx, imgs, out_imgs, series, release_points, flow_directions
        else:
            return imgs, out_imgs, series, release_points
