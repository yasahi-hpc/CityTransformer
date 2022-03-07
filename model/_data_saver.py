import torch
import numpy as np
import xarray as xr
import pathlib
from ._base_saver import _BaseSaver

def to_numpy(var):
    return var.numpy() if var.device == 'cpu' else var.cpu().numpy()

class _CityTransformerDataSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformer'

        self.clip = kwargs.get('clip')
        self.version = kwargs.get('version')
        self.station_positions = kwargs.get('station_positions')
        self.norm_dict = kwargs.get('norm_dict')
        self.num_vars = 4
        self.num_stations = kwargs.get('num_stations') // self.num_vars

    def _save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        sdf_release = kwargs.get('sdf_release')
        release_points = kwargs.get('release_points')
        flow_directions = kwargs.get('flow_directions')
        ref = kwargs.get('ref')
        pred = kwargs.get('pred')
        indices = kwargs.get('indices')
        series = kwargs.get('series')
        mode = kwargs.get('mode')
        epoch = kwargs.get('epoch')

        # Convert to numpy array
        ref_plume, ref_zeros_map = ref
        pred_plume, pred_zeros_map = pred
        
        series         = to_numpy(series)
        sdf_release    = to_numpy(sdf_release)
        ref_plume      = to_numpy(ref_plume)
        ref_zeros_map  = to_numpy(ref_zeros_map)
        pred_plume     = to_numpy(pred_plume)
        pred_zeros_map = to_numpy(pred_zeros_map)
        
        *_, n_digits, ny, nx = ref_plume.shape

        data_dir = self.out_dir / mode

        for i, idx in enumerate(indices):
            filename = data_dir / f'{mode}{idx:06}_epoch{epoch:04}.nc'

            coords_xy_list = ['y', 'x']
            coords_list = ['digits', 'y', 'x']
            data_vars = {}

            data_vars['sdf_release']    = (coords_xy_list, np.squeeze(sdf_release[i]))
            data_vars['ref_plume']      = (coords_list,    ref_plume[i])
            data_vars['ref_zeros_map']  = (coords_xy_list, np.squeeze(ref_zeros_map[i]))
            data_vars['pred_plume']     = (coords_list,    pred_plume[i])
            data_vars['pred_zeros_map'] = (coords_xy_list, np.squeeze(pred_zeros_map[i]))
            data_vars['station_positions'] = (['stations', 'positions'], self.station_positions)

            series_tmp = series[i]
            coords = {}
            
            *_, nt, nz = series_tmp.shape
            series_tmp = series_tmp.reshape(self.num_vars, -1, nt, nz)
            data_vars['series'] = (['vars', 'stations', 'time', 'z'], series_tmp)
            coords['time'] = np.arange(nt)
            coords['positions'] = np.arange(3)
            
            coords['x'] = np.arange(nx)
            coords['y'] = np.arange(ny)
            coords['z'] = np.arange(nz)
            coords['stations'] = np.arange(self.num_stations)
            coords['vars'] = np.arange(self.num_vars)
            coords['digits'] = np.arange(n_digits)

            attrs = {}
            release_x, release_y = release_points[i]
            v0, theta0 = flow_directions[i]
            attrs['release_x'] = float(release_x)
            attrs['release_y'] = float(release_y)
            attrs['model_name'] = self.model_name
            attrs['clip'] = self.clip
            attrs['v0'] = float(v0)
            attrs['theta0'] = float(theta0)
            attrs['version'] = self.version
            attrs['release_sdf_max'] = float(self.norm_dict['release_sdf_max'])
            attrs['release_sdf_min'] = float(self.norm_dict['release_sdf_min'])
            
            xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs).to_netcdf(filename)

class _CityTransformerInverseDataSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = 'CityTransformerInverse'
        self.clip = kwargs.get('clip')
        self.version = kwargs.get('version')
        self.station_positions = kwargs.get('station_positions')
        self.norm_dict = kwargs.get('norm_dict')
        self.num_vars = 4
        self.num_stations = kwargs.get('num_stations') // self.num_vars

    def _save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        release_points = kwargs.get('release_points')
        flow_directions = kwargs.get('flow_directions')
        ref_release = kwargs.get('ref_release')
        pred_release = kwargs.get('pred_release')
        indices = kwargs.get('indices')
        series = kwargs.get('series')
        mode = kwargs.get('mode')
        epoch = kwargs.get('epoch')

        levelset = to_numpy(levelset)
        ref_release = to_numpy(ref_release)
        pred_release = to_numpy(pred_release)
        series         = to_numpy(series)
        
        *_, ny, nx = ref_release.shape

        data_dir = self.out_dir / mode

        for i, idx in enumerate(indices):
            filename = data_dir / f'{mode}{idx:06}_epoch{epoch:04}.nc'

            coords_xy_list = ['y', 'x']
            data_vars = {}

            data_vars['levelset'] = (coords_xy_list, np.squeeze(levelset[i]))
            data_vars['ref_release'] = (coords_xy_list, np.squeeze(ref_release[i]))
            data_vars['pred_release'] = (coords_xy_list, np.squeeze(pred_release[i]))
            data_vars['station_positions'] = (['stations', 'positions'], self.station_positions)

            series_tmp = series[i]
            coords = {}
            
            *_, nt, nz = series_tmp.shape
            series_tmp = series_tmp.reshape(self.num_vars, -1, nt, nz)
            data_vars['series'] = (['vars', 'stations', 'time', 'z'], series_tmp)
            coords['time'] = np.arange(nt)
            coords['positions'] = np.arange(3)
            
            coords['x'] = np.arange(nx)
            coords['y'] = np.arange(ny)
            coords['z'] = np.arange(nz)
            coords['stations'] = np.arange(self.num_stations)
            coords['vars'] = np.arange(self.num_vars)

            attrs = {}
            release_x, release_y = release_points[i]
            v0, theta0 = flow_directions[i]
            attrs['release_x'] = float(release_x)
            attrs['release_y'] = float(release_y)
            attrs['model_name'] = self.model_name
            attrs['clip'] = self.clip
            attrs['v0'] = float(v0)
            attrs['theta0'] = float(theta0)
            attrs['version'] = self.version
            attrs['epoch'] = epoch
            attrs['release_sdf_max'] = float(self.norm_dict['release_sdf_max'])
            attrs['release_sdf_min'] = float(self.norm_dict['release_sdf_min'])
            
            xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs).to_netcdf(filename)
