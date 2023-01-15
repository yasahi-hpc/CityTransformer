import torch
import numpy as np
import xarray as xr
import pathlib
from ._base_saver import _BaseSaver

def to_numpy(var):
    return var.numpy() if var.device == 'cpu' else var.cpu().numpy()

class CityTransformerDataSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformer'

        self.clip = kwargs.get('clip')
        self.version = kwargs.get('version')
        self.station_positions = kwargs.get('station_positions')
        self.norm_dict = kwargs.get('norm_dict')
        self.n_stations = kwargs.get('n_stations')
        self.nz = kwargs.get('nz')

    def save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        sdf_release = kwargs.get('sdf_release')
        release_points = kwargs.get('release_points')
        flows_and_sources = kwargs.get('flows_and_sources')
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
        
        *_, ny, nx = ref_plume.shape

        data_dir = self.out_dir / mode

        for i, idx in enumerate(indices):
            filename = data_dir / f'{mode}{idx:06}_epoch{epoch:04}.nc'

            coords_list = ['y', 'x']
            data_vars = {}

            data_vars['levelset']       = (coords_list, np.squeeze(levelset[i]))
            data_vars['sdf_release']    = (coords_list, np.squeeze(sdf_release[i]))
            data_vars['ref_plume']      = (coords_list, np.squeeze(ref_plume[i]))
            data_vars['ref_zeros_map']  = (coords_list, np.squeeze(ref_zeros_map[i]))
            data_vars['pred_plume']     = (coords_list, np.squeeze(pred_plume[i]))
            data_vars['pred_zeros_map'] = (coords_list, np.squeeze(pred_zeros_map[i]))
            data_vars['station_positions'] = (['stations', 'positions'], self.station_positions)

            series_tmp = series[i]
            coords = {}
            
            # (nt, ns * (nz+nz+1+1))
            # (u, v, c, logc)
            nt, *_ = series_tmp.shape
            series_tmp = series_tmp.reshape(nt, self.n_stations, -1)

            data_vars['u'] = (['time', 'stations', 'z'], series_tmp[:,:,:self.nz])
            data_vars['v'] = (['time', 'stations', 'z'], series_tmp[:,:,self.nz:self.nz*2])
            data_vars['concentration'] = (['time', 'stations'], series_tmp[:,:,self.nz*2])
            data_vars['log_concentration'] = (['time', 'stations'], series_tmp[:,:,self.nz*2+1])

            coords['time'] = np.arange(nt)
            coords['positions'] = np.arange(3)
            
            coords['x'] = np.arange(nx)
            coords['y'] = np.arange(ny)
            coords['z'] = np.arange(self.nz)
            coords['stations'] = np.arange(self.n_stations)

            attrs = {}
            release_x, release_y = release_points[i]
            v0, theta0, _ = flows_and_sources[i]
            distance_and_source_max = self.norm_dict['distance_and_source_max'].cpu().numpy().squeeze()
            distance_and_source_min = self.norm_dict['distance_and_source_min'].cpu().numpy().squeeze()
            attrs['release_x'] = float(release_x)
            attrs['release_y'] = float(release_y)
            attrs['model_name'] = self.model_name
            attrs['clip'] = self.clip
            attrs['v0'] = float(v0)
            attrs['theta0'] = float(theta0)
            attrs['version'] = self.version
            attrs['distance_max'] = distance_and_source_max[0]
            attrs['distance_min'] = distance_and_source_min[0]
            attrs['source_max'] = distance_and_source_max[1]
            attrs['source_min'] = distance_and_source_min[1]
            
            xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs).to_netcdf(filename)

class CityTransformerInverseDataSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = 'CityTransformerInverse'
        self.clip = kwargs.get('clip')
        self.version = kwargs.get('version')
        self.station_positions = kwargs.get('station_positions')
        self.norm_dict = kwargs.get('norm_dict')
        self.n_stations = kwargs.get('n_stations')
        self.nz = kwargs.get('nz')

    def save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        release_points = kwargs.get('release_points')
        flows_and_sources = kwargs.get('flows_and_sources')
        ref_distance_and_amplitude = kwargs.get('ref_distance_and_amplitude')
        pred_distance_and_amplitude = kwargs.get('pred_distance_and_amplitude')
        indices = kwargs.get('indices')
        series = kwargs.get('series')
        mode = kwargs.get('mode')
        epoch = kwargs.get('epoch')

        levelset = to_numpy(levelset)
        ref_distance_and_amplitude = to_numpy(ref_distance_and_amplitude)
        pred_distance_and_amplitude = to_numpy(pred_distance_and_amplitude)
        series         = to_numpy(series)

        ref_release = ref_distance_and_amplitude[:, 0]
        ref_amplitude = ref_distance_and_amplitude[:, 1]

        pred_release = pred_distance_and_amplitude[:, 0]
        pred_amplitude = pred_distance_and_amplitude[:, 1]
        
        *_, ny, nx = ref_release.shape

        data_dir = self.out_dir / mode

        for i, idx in enumerate(indices):
            filename = data_dir / f'{mode}{idx:06}_epoch{epoch:04}.nc'

            coords_list = ['y', 'x']
            data_vars = {}

            data_vars['levelset']              = (coords_list, np.squeeze(levelset[i]))
            data_vars['ref_release']           = (coords_list, np.squeeze(ref_release[i]))
            data_vars['ref_source_amplitude']  = (coords_list, np.squeeze(ref_amplitude[i]))
            data_vars['pred_release']          = (coords_list, np.squeeze(pred_release[i]))
            data_vars['pred_source_amplitude'] = (coords_list, np.squeeze(pred_amplitude[i]))
            data_vars['station_positions']     = (['stations', 'positions'], self.station_positions)

            series_tmp = series[i]
            coords = {}
            
            # (nt, ns * (nz+nz+1+1))
            # (u, v, c, logc)
            nt, *_ = series_tmp.shape
            series_tmp = series_tmp.reshape(nt, self.n_stations, -1)

            data_vars['u'] = (['time', 'stations', 'z'], series_tmp[:,:,:self.nz])
            data_vars['v'] = (['time', 'stations', 'z'], series_tmp[:,:,self.nz:self.nz*2])
            data_vars['concentration'] = (['time', 'stations'], series_tmp[:,:,self.nz*2])
            data_vars['log_concentration'] = (['time', 'stations'], series_tmp[:,:,self.nz*2+1])

            coords['time'] = np.arange(nt)
            coords['positions'] = np.arange(3)
            
            coords['x'] = np.arange(nx)
            coords['y'] = np.arange(ny)
            coords['z'] = np.arange(self.nz)
            coords['stations'] = np.arange(self.n_stations)

            attrs = {}
            release_x, release_y = release_points[i]
            v0, theta0, source_factor = flows_and_sources[i]
            distance_and_source_max = self.norm_dict['distance_and_source_max'].cpu().numpy().squeeze()
            distance_and_source_min = self.norm_dict['distance_and_source_min'].cpu().numpy().squeeze()
            attrs['release_x'] = float(release_x)
            attrs['release_y'] = float(release_y)
            attrs['model_name'] = self.model_name
            attrs['clip'] = self.clip
            attrs['v0'] = float(v0)
            attrs['theta0'] = float(theta0)
            attrs['source_factor'] = float(source_factor)
            attrs['version'] = self.version
            attrs['epoch'] = epoch
            attrs['distance_max'] = distance_and_source_max[0]
            attrs['distance_min'] = distance_and_source_min[0]
            attrs['source_max'] = distance_and_source_max[1]
            attrs['source_min'] = distance_and_source_min[1]
            
            xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs).to_netcdf(filename)
