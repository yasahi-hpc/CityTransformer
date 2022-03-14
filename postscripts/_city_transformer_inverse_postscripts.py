"""
Convert data and then visualize
 
Data Manupulation
1. Save metrics for validation and test data
 
Save figures
1. Loss curve
2. source location map
3. histogram of distance
4. grid-based historgam 
"""

import pathlib
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ._base_postscript import _BasePostscripts

class CityTransformerInversePostscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformerInverse'
        self.modes = ['val', 'test']
        self.alpha = 0.3
        self.error_min = 0
        self.error_max = 350
        self.vmin = 0
        self.vmax = 100
        self.nb_bins = 100
        
        self.fig_names = ['loss', 'source_detection', 'histogram']
        self.extent = [-1024,1024,-1024,1024]

        # Matplotlib settings
        mpl.style.use('classic')
        fontsize = 28
        self.fontsize = fontsize
        fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        plt.rc('font', family=fontname)
        
        self.title_font = {'fontname':fontname, 'size':fontsize, 'color':'black',
                           'verticalalignment':'bottom'}
        self.axis_font = {'fontname':fontname, 'size':fontsize}

    def _visualize(self, epoch):
        self.data_dir = self.img_dir / 'histogram/data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        super()._visualize_loss()
        self.__preprocess(epoch)
        self.__visualize_source_location(epoch)
        self.__visualize_histogram(epoch)
        self.__visualize_histogram_map(epoch)

    def __preprocess(self, epoch):
        data_vars = {}
        for mode in self.modes:
            all_pred, all_ref, all_flows = [], [], []
            nb_shots = self.nb_shots_dict[mode]

            coords = {'shots': np.arange(nb_shots), 'dim_axis': np.arange(2)}
            for i in range(nb_shots):
                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)
                pred, ref, flows = self.__get_source_location_and_flows(ds=ds)
                all_pred.append(pred)
                all_ref.append(ref)
                all_flows.append(flows)

            all_pred  = np.asarray(all_pred)
            all_ref   = np.asarray(all_ref)
            all_flows = np.asarray(all_flows)
            
            data_vars[f'{mode}_ref']   = (['shots', 'dim_axis'], all_ref)
            data_vars[f'{mode}_pred']  = (['shots', 'dim_axis'], all_pred)
            data_vars[f'{mode}_flows'] = (['shots', 'dim_axis'], all_flows)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.to_netcdf(self.data_dir / f'source_location_{self.arch_name}.nc', engine='netcdf4')

    def __find_release_point(self, distance_function):
        idx_y, idx_x = np.unravel_index(np.argmin(np.abs(distance_function), axis=None), distance_function.shape)
        ny, nx = distance_function.shape
        xmin, xmax, ymin, ymax = self.extent
        gx, gy = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        return gx[idx_x], gy[idx_y]

    def __get_source_location_and_flows(self, ds):
        x, y = ds.attrs['release_x'], ds.attrs['release_y']
        v0, theta0 = ds.attrs['v0'], ds.attrs['theta0']
        u, v = v0 * np.cos(theta0), v0 * np.sin(theta0)
        u, v = self.__vel_power(u), self.__vel_power(v)
        pred = ds['pred_release'].values

        x_, y_ = self.__find_release_point(pred)
        
        return np.array([x_, y_]), np.array([x, y]), np.array([u, v])

    def __vel_power(self, v):
        """
        Return radial profile (hard coded)
        """
        nz = 50
        z_min = 25 # [m]
        dz = 50 # [m]
        z_cutoff = 600 # [m]
        z_ref = z_cutoff
        z_g = 0 # [m]
        alpha = 0.27
        
        z = np.arange(nz) * dz + z_min
        iz_cutoff = np.sum(z <= z_cutoff)
        z_power = ((z - z_g) / z_ref) ** alpha
        z_power = np.where(z <= z_cutoff, z_power, z_power[iz_cutoff])
         
        return v * z_power[0]

    def __visualize_source_location(self, epoch):
    
        for mode in self.modes:
            nb_shots = self.nb_shots_dict[mode]

            for i in range(nb_shots):
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12),
                                         subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.05))
                axes[1, 0].set_visible(False)
                axes[2, 0].set_visible(False)

                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)

                release_sdf_max = ds.attrs['release_sdf_max']
                x, y = ds.attrs['release_x'], ds.attrs['release_y']
                v0, theta0 = ds.attrs['v0'], ds.attrs['theta0']
                u, v = v0 * np.cos(theta0), v0 * np.sin(theta0)
                u, v = self.__vel_power(u), self.__vel_power(v)

                levelset = ds['levelset'].values
                ref = ds['ref_release'].values
                pred = ds['pred_release'].values

                ny, nx = ref.shape
                xmin, xmax, ymin, ymax = self.extent
                gx, gy = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)

                # Ground truth
                axes[0, 0].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im = axes[0, 0].imshow(ref / release_sdf_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)
                axes[0, 0].plot(x, y, 'r*', markersize=10)
                axes[0, 0].set_title('Ground Truth', **self.title_font)

                # prediction
                axes[0, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im = axes[0, 1].imshow(pred / release_sdf_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)

                x_, y_ = self.__find_release_point(pred)
                axes[0, 1].plot(x_, y_, 'b^', markersize=10)
                axes[0, 1].set_title(self.arch_name, **self.title_font)

                # Error
                error = np.abs( pred - ref )
                axes[1, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im2 = axes[1, 1].imshow(error / release_sdf_max, cmap='jet', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=0.1)
                axes[1, 1].plot(x , y , color='none', marker='*', markeredgecolor='r', markeredgewidth=2, markersize=10)
                axes[1, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='b', markeredgewidth=2, markersize=10)
                
                # Boundary map and release points
                im3 = axes[2, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                axes[2, 1].plot(x , y , color='none', marker='*', markeredgecolor='r', markeredgewidth=2, markersize=10)
                axes[2, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='b', markeredgewidth=2, markersize=10)

                # Vector plots
                U = np.where(levelset >= 0., 0., np.ones_like(levelset) * u)
                V = np.where(levelset >= 0., 0., np.ones_like(levelset) * v)
                X, Y = np.meshgrid(gx, gy)
                nsample = 20
                X, Y = X[::nsample, ::nsample], Y[::nsample, ::nsample]
                U, V = U[::nsample, ::nsample], V[::nsample, ::nsample]
                axes[2, 1].quiver(X, Y, U, V, units='xy', scale=1./nsample, color='g')

                cbar  = fig.colorbar(im,  ax=axes[0, :], shrink=0.9)
                cbar2 = fig.colorbar(im2, ax=axes[1, :], shrink=0.9)
                cbar3 = fig.colorbar(im3, ax=axes[2, :])
                
                cbar3.remove()

                figname =  self.img_dir / 'source_detection' / f'log_{mode}{i:06}_epoch{epoch:04}.png'
                plt.savefig(figname, bbox_inches='tight')
                plt.close('all')

    def __visualize_histogram(self, epoch):
        figsize = (10, 10)
        filename = self.data_dir / f'source_location_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')
        bins = np.linspace(self.error_min, self.error_max, self.nb_bins)

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize)

            ref = ds[f'{mode}_ref']
            pred = ds[f'{mode}_pred']

            distance = np.sqrt((ref[:, 0]-pred[:, 0])**2 + (ref[:, 1]-pred[:, 1])**2)
            weights = np.ones_like(distance) / len(distance)
            
            ax.hist(distance, bins=bins, alpha=self.alpha, weights=weights, label=self.arch_name)

            average, std = np.mean( np.abs(distance) ), np.std( np.abs(distance) )
            print(f'model: {self.arch_name}, average: {average}, std: {std}')

            ax.legend(loc='upper right', prop={'size': self.fontsize})
            ax.set_xlabel(r'Error [m]', **self.axis_font)
            ax.grid(ls='dashed', lw=1)
            figname =  self.img_dir / 'histogram' / f'{self.arch_name}_{mode}_hist.png'
            fig.savefig(figname)
            
            plt.close('all')

    def __visualize_histogram_map(self, epoch):
        figsize = (6, 8)
        filename = self.data_dir / f'source_location_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')

        # Create grid to compute histogram
        xmax, ymax = 160, 300
        xmax_fig, ymax_fig = 310, 568
        x = np.linspace(-xmax, xmax, 30)
        y = np.linspace(-ymax, ymax, 60)
        
        X, Y = np.meshgrid(x, y)
        
        dy, dx = y[1] - y[0], x[1] - x[0]

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize)

            # Reading sample data to get levelset and stations
            i = 0
            filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
            ds_sample = xr.open_dataset(filename, engine='netcdf4')
            levelset = ds_sample['levelset'].values

            ref = ds[f'{mode}_ref']
            pred = ds[f'{mode}_pred']

            distance = np.sqrt((ref[:, 0]-pred[:, 0])**2 + (ref[:, 1]-pred[:, 1])**2)
            all_elem = distance.shape[0]
            ave_meter = np.zeros_like(X)

            for iy, y_ in enumerate(y):
                for ix, x_ in enumerate(x):
                    mask_x = np.logical_and(x_ <= ref[:, 0], ref[:, 0] < x_ + dx)
                    mask_y = np.logical_and(y_ <= ref[:, 1], ref[:, 1] < y_ + dy)
                    
                    mask = np.logical_and(mask_x, mask_y)
                    if np.any(mask):
                        ave_meter[iy, ix] = np.mean(distance[mask])
                    else:
                        ave_meter[iy, ix] = 0.

            # Plot
            fig, ax = plt.subplots(figsize = figsize, subplot_kw={'xticks':[], 'yticks':[]})
            
            # Plot buildings
            ax.imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
            im = ax.imshow(ave_meter, cmap='jet', origin='lower', extent=[-xmax,xmax,-ymax,ymax], alpha=self.alpha, vmin=self.vmin, vmax=self.vmax)

            stations = ds_sample['station_positions'].values
            for position in stations: 
                st_x, st_y, _ = position
                ax.plot(st_x, st_y, color='none', marker='*', markeredgecolor='m', markeredgewidth=1, markersize=12)

            cbar = fig.colorbar(im, ax=ax, shrink=0.9)

            ax.set_xlim([-xmax_fig, xmax_fig])
            ax.set_ylim([-ymax_fig, ymax_fig])

            # Add scale bar
            scalebar = AnchoredSizeBar(ax.transData,
                                       100, r'$100 {\rm [m]}$', 'lower right',
                                       pad=0.1,
                                       color='black',
                                       frameon=False,
                                       size_vertical=1,
                                       fontproperties=fm.FontProperties(size=self.fontsize*0.6))
            ax.add_artist(scalebar)

            figname =  self.img_dir / 'histogram' / f'{self.arch_name}_{mode}_hist_map.png'
            fig.savefig(figname)
            
            plt.close('all')
