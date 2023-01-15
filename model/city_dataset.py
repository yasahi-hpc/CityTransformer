"""
Version 0: UNet + Transformer (attention on time), using the ground plume concentration only
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, t, s*c) time series data of measurement, c = (z, z, 1, 1)

Output:
    imgs: (b, 2, h, w) concentration + binary-map
      
Version 1: UNet + MLP, using the ground plume concentration only
Input:
    imgs: (b, c, h, w) sdfs of buildings and release points
    series: (b, 1, s*c) time series data of measurement, c = (z, z, 1, 1)

Output:
    imgs: (b, 2, h, w) concentration + binary-map
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
                          'nb_source_amps',
                          'source_amps_max',
                          'randomize_source_amps',
                          'start_idx',
                          'nz',
                          'n_stations',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        device = kwargs.get('device', 'cpu')
        version = kwargs.get('version', 0)
        nz = kwargs.get('nz', 100)

        # If nb_source_amps > 1, then concentrations are augumented by random numbers
        # This option is for inverse problem
        self.source_amps_max = kwargs.get('source_amps_max', 100)
        self.nb_source_amps = int(kwargs.get('nb_source_amps', 1))
        self.start_idx = kwargs.get('start_idx', 0)
        inference_mode = kwargs.get('inference_mode', False)
        self.randomize_source_amps = kwargs.get('randomize_source_amps', False)
        self.n_stations = kwargs.get('n_stations', 14)

        self.files = sorted(list(pathlib.Path(path).glob('shot*.nc')))
        self.datanum = len(self.files) * self.nb_source_amps
        self.version = version
        self.inference_mode = inference_mode
        self.nz = nz

        # Version 0 and 1 are supported
        assert self.version in [0,1]

        # read datashape
        ds = xr.open_dataset(self.files[0], engine='netcdf4')

        # Control the vertical level of measurements
        ds = ds.isel(z=slice(0, self.nz))
        self.station_positions = ds['station_position'].values
        Nt, _, Nz = ds['u'].shape # (t, s, c)
        Ny, Nx = ds['levelset'].shape

        self.resolution_dict = {}
        self.resolution_dict['Nx'] = Nx
        self.resolution_dict['Ny'] = Ny
        self.resolution_dict['Nz'] = Nz
        self.resolution_dict['Nt'] = Nt
        self.resolution_dict['n_stations'] = self.n_stations

        assert self.n_stations in [5, 10, 14]
        # Control the number of stations for sensitivity analysis
        if self.n_stations == 5:
            self.station_list = [4, 9, 10, 12, 13]
        elif self.n_stations == 10:
            self.station_list = [0, 4, 5, 6, 7, 8, 9, 10, 12, 13]
        else:
            self.station_list = [i for i in range(14)]

        self.station_positions = self.station_positions[self.station_list]

        # Normalization coefficients
        try:
            ds_stats = xr.open_dataset('stats.nc', engine='netcdf4')
        except:
            raise FileNotFoundError('stats.nc storing normalization coefficients not found')

        if self.randomize_source_amps:
            try:
                ds_rands = xr.open_dataset('random_numbers.nc', engine='netcdf4')
            except:
                raise FileNotFoundError('random_numbers.nc storing random numbers to control source amplitude not found')

            self.random_numbers = ds_rands['random_factors'].values.flatten() # uniform numbers between 1 to 100
            self.random_numbers = self.random_numbers[self.start_idx:]
            # check the random numbers are sufficient
            if len(self.random_numbers) < self.datanum:
                small_enough_number = len(self.random_numbers) // len(self.files)
                raise ValueError(f'nb_source_amps is too large. It should be smaller than {small_enough_number}')

        # Track min, max of output values
        to_tensor = lambda variable: torch.tensor(variable).view(1,-1,1,1).float().to(device)
        self.norm_dict = {}

        # Input data shape (b, t, s, c) or (b, 1, s, c)
        # The data shape should be (4) -> (c) = (z, z, 1, 1) -> (1, 1, s, c)
        self.norm_dict['concentrations_max'] = to_tensor(ds_stats['log_concentration_max'].values)
        self.norm_dict['concentrations_min'] = to_tensor(ds_stats['log_concentration_min'].values)

        # Used for inverse problem
        # Amplitude in the range of 1 to 100
        distance_and_source_max = np.array( [ds_stats['imgs_max'].values[0], self.source_amps_max] )
        distance_and_source_min = np.array( [ds_stats['imgs_min'].values[0], 1] )
        self.norm_dict['distance_and_source_max'] = to_tensor( distance_and_source_max )
        self.norm_dict['distance_and_source_min'] = to_tensor( distance_and_source_min )

        imgs_max = ds_stats['imgs_max'].values
        imgs_min = ds_stats['imgs_min'].values
        if self.randomize_source_amps:
            # (release, amps, sdf)
            imgs_max = np.array([imgs_max[0], self.source_amps_max, imgs_max[1]])
            imgs_min = np.array([imgs_min[0], 1, imgs_min[1]])

        self.norm_dict['imgs_max'] = to_tensor( imgs_max )
        self.norm_dict['imgs_min'] = to_tensor( imgs_min )

        def to_series(values, multiplication=False):
            u0, v0, c0, logc0 = values
            u_norm, v_norm = np.ones(Nz) * u0, np.ones(Nz) * v0

            if multiplication:
                # Since mulitplied by random numbers [1, 100], max is also multiplied by 100
                c0 = c0 * 100
                logc0 = logc0 + 2

            c_norm = np.array([c0, logc0])
            variable = np.concatenate([u_norm, v_norm, c_norm], axis=0)

            return torch.tensor(variable).view(1,1,1,-1).float().to(device)

        self.norm_dict['series_max'] = to_series(ds_stats['series_max'].values, self.randomize_source_amps)
        self.norm_dict['series_min'] = to_series(ds_stats['series_min'].values)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        """
        Single dataset

        imgs, out_imgs: (b, c, h, w)
        series: (b, t, s, c)
        """

        source_factor = 1
        # To randomly change the source amplitude
        if self.randomize_source_amps:
            source_factor = self.random_numbers[idx]
            idx = idx % len(self.files)

        ds = xr.open_dataset(self.files[idx], engine='netcdf4')
        ds = ds.isel(z=slice(0, self.nz))

        # Scan with respect to the stations
        station_list = self.station_list
        ds = ds.isel(station=station_list)

        # Input imgs: sdf and release_point (h, w)
        release_point = ds['release_point'].values
        sdf = ds['levelset'].values

        if self.randomize_source_amps:
            # (release, amps, sdf)
            source_amplitude = np.ones_like(sdf) * source_factor
            imgs = torch.tensor(np.stack([release_point, source_amplitude, sdf], axis=0)).float()
        else:
            imgs = torch.tensor(np.stack([release_point, sdf], axis=0)).float()

        # Input series: u, v, concentration, log_concentration
        u = ds['u'].values # (nt, ns, nz)
        v = ds['v'].values

        # On the ground only
        concentration = ds['concentration'].values # (nt, ns)
        log_concentration = ds['log_concentration'].values

        if self.randomize_source_amps:
            concentration *= source_factor
            log_concentration += np.log10(source_factor)

        Nt = 1
        if self.version == 0:
            # Time series data
            # (nt, ns, (nz+nz+1+1))
            conentration = np.expand_dims(concentration, axis=2)
            log_concentration = np.expand_dims(log_concentration, axis=2)
            Nt = self.resolution_dict['Nt']

        elif self.version == 1:
            # Time avareaged data
            # (1, ns, (nz+nz+1+1))
            time_average = lambda var, keepdims: np.mean(var, axis=0, keepdims=keepdims)
            u = time_average(u, True) # (1, ns, nz)
            v = time_average(v, True)
            conentration = np.expand_dims(time_average(concentration, True), axis=2) # (1, ns, 1)
            log_concentration = np.expand_dims(time_average(log_concentration, True), axis=2)

        series = np.concatenate([u, v, conentration, log_concentration], axis=2) # (nt, ns, (nz+nz+1+1))
        series = torch.tensor(series).float()

        # Output imgs: concentration
        # Release points
        release_points = torch.tensor([ds.attrs['release_x'], ds.attrs['release_y']]).float()
        flows_and_sources = torch.tensor([ds.attrs['v0'], ds.attrs['theta0'], source_factor]).float()

        concentration_binary_map = np.expand_dims(ds['concentration_binary_map'].values, axis=0)
        log_concentration_map = np.expand_dims(ds['log_concentration_map'].values, axis=0)

        out_imgs = torch.tensor(np.concatenate([log_concentration_map, concentration_binary_map], axis=0)).float()

        if self.inference_mode:
            return idx, imgs, out_imgs, series, release_points, flows_and_sources
        else:
            return imgs, out_imgs, series, release_points
