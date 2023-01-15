"""
Convert data and then visualize
 
Data Manupulation
1. Save metrics for validation and test data
 
Save figures
1. Loss curve
2. source location map (Fig. 11)
3. source emission rates (Fig. 12)
4. histogram of error distance and scatter plot for source emission rates (Fig. 13)
5. grid-based historgam (Fig. 14)
6. Source locations for large relative errors (Fig. 15)
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
        
        self.fig_names = ['loss', 'source_detection', 'source_amplitude', 'histogram']
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
        self.__visualize_source_location(epoch) # Fig.11
        self.__visualize_source_amplitude(epoch) # Fig.12
        self.__visualize_histogram_location(epoch) # Fig.13a
        self.__visualize_scatter_amplitude(epoch) # Fig.13b
        self.__visualize_histogram_map(epoch) # Fig.14
        self.__visualize_flow_direction_map(epoch) # Fig.15

    def __preprocess(self, epoch):
        data_vars = {}
        for mode in self.modes:
            all_pred, all_ref, all_flows = [], [], []
            all_pred_amp, all_ref_amp = [], []
            nb_shots = self.nb_shots_dict[mode]

            coords = {'shots': np.arange(nb_shots), 'dim_axis': np.arange(2)}
            for i in range(nb_shots):
                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)

                # Source locations and flow directions
                pred, ref, flows = self.__get_source_location_and_flows(ds=ds)
                all_pred.append(pred)
                all_ref.append(ref)
                all_flows.append(flows)

                # Source amplitudes
                pred_amp, ref_amp = self.__get_source_amplitudes(ds=ds)
                all_pred_amp.append(pred_amp)
                all_ref_amp.append(ref_amp)

            all_pred  = np.asarray(all_pred)
            all_ref   = np.asarray(all_ref)
            all_flows = np.asarray(all_flows)
            all_pred_amp = np.asarray(all_pred_amp)
            all_ref_amp = np.asarray(all_ref_amp)
            
            data_vars[f'{mode}_ref']   = (['shots', 'dim_axis'], all_ref)
            data_vars[f'{mode}_pred']  = (['shots', 'dim_axis'], all_pred)
            data_vars[f'{mode}_flows'] = (['shots', 'dim_axis'], all_flows)
            data_vars[f'{mode}_ref_amp']   = (['shots'], all_ref_amp)
            data_vars[f'{mode}_pred_amp']  = (['shots'], all_pred_amp)

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.to_netcdf(self.data_dir / f'source_info_{self.arch_name}.nc', engine='netcdf4')

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

    def __get_source_amplitudes(self, ds):
        pred = ds['pred_source_amplitude'].values
        ref  = ds['ref_source_amplitude'].values
        
        amp = lambda var: np.mean(np.abs(var))
        
        return amp(pred), amp(ref)

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
        """
        Fig.11
        First row: Distance function (ref) and (pred)
        Second row: Errors in distance function (ref) and (pred)
        Third row: Locations of observation stations, source locations and principal flow directions
        """
    
        for mode in self.modes:
            nb_shots = self.nb_shots_dict[mode]

            for i in range(nb_shots):
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12),
                                         subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.05))
                axes[1, 0].set_visible(False)
                axes[2, 0].set_visible(False)

                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)

                distance_max = ds.attrs['distance_max']
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
                im = axes[0, 0].imshow(ref / distance_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)
                axes[0, 0].plot(x, y, color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=10)
                axes[0, 0].set_title('Ground Truth', **self.title_font)

                # prediction
                axes[0, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im = axes[0, 1].imshow(pred / distance_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)

                x_, y_ = self.__find_release_point(pred)
                axes[0, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='m', markeredgewidth=2, markersize=10)
                axes[0, 1].set_title(self.arch_name, **self.title_font)

                # Error
                error = np.abs( pred - ref )
                axes[1, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im2 = axes[1, 1].imshow(error / distance_max, cmap='jet', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=0.1)
                axes[1, 1].plot(x , y , color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=10)
                axes[1, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='m', markeredgewidth=2, markersize=10)
                
                # Boundary map and release points
                im3 = axes[2, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                axes[2, 1].plot(x , y , color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=10)
                axes[2, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='m', markeredgewidth=2, markersize=10)

                # Vector plots
                U = np.where(levelset >= 0., 0., np.ones_like(levelset) * u)
                V = np.where(levelset >= 0., 0., np.ones_like(levelset) * v)
                X, Y = np.meshgrid(gx, gy)
                nsample = 10
                X, Y = X[::nsample, ::nsample], Y[::nsample, ::nsample]
                U, V = U[::nsample, ::nsample], V[::nsample, ::nsample]
                axes[2, 1].quiver(X, Y, U, V, units='xy', scale=0.5/nsample, color='b', headwidth=10)

                cbar  = fig.colorbar(im,  ax=axes[0, :], shrink=0.9)
                cbar2 = fig.colorbar(im2, ax=axes[1, :], shrink=0.9)
                cbar3 = fig.colorbar(im3, ax=axes[2, :])
                
                cbar3.remove()

                figname =  self.img_dir / 'source_detection' / f'source_location_{mode}{i:06}_epoch{epoch:04}.png'
                plt.savefig(figname, bbox_inches='tight')
                plt.close('all')

    def __visualize_source_amplitude(self, epoch, amp_max=100):
        """
        Fig.12
        First row: Emission rates (ref) and (pred)
        Second row: Errors in emission rates (ref) and (pred)
        Source amplitude has been scanned from 1 to 100
        """

        amp = lambda var: np.mean(np.abs(var))
    
        for mode in self.modes:
            nb_shots = self.nb_shots_dict[mode]

            for i in range(nb_shots):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                                         subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.05))
                axes[1, 0].set_visible(False)

                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)

                levelset = ds['levelset'].values
                ref = ds['ref_source_amplitude'].values
                pred = ds['pred_source_amplitude'].values
                pred_loc = ds['pred_release'].values

                ny, nx = ref.shape
                xmin, xmax, ymin, ymax = self.extent
                gx, gy = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)

                # release points
                x, y = ds.attrs['release_x'], ds.attrs['release_y']
                x_, y_ = self.__find_release_point(pred_loc)

                # Ground truth
                title_font = self.title_font.copy()
                title_font['size'] = 20
                axes[0, 0].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im = axes[0, 0].imshow(ref / amp_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)
                axes[0, 0].plot(x, y, color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=10)
                axes[0, 0].set_title(f'Ground Truth: {amp(ref):.2f}', **title_font)

                # prediction
                axes[0, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im = axes[0, 1].imshow(pred / amp_max, cmap='seismic', origin='lower', extent=self.extent, alpha=self.alpha, vmin=0, vmax=1)

                axes[0, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='m', markeredgewidth=2, markersize=10)
                axes[0, 1].set_title(f'{self.arch_name}: {amp(pred):.2f}', **title_font)

                # errors
                error = np.abs( pred - ref )
                axes[1, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)
                im2 = axes[1, 1].imshow(error / amp_max, cmap='jet', origin='lower', extent=self.extent, alpha=self.alpha, vmin=-0.1, vmax=0.1)
                axes[1, 1].plot(x , y , color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=10)
                axes[1, 1].plot(x_, y_, color='none', marker='^', markeredgecolor='m', markeredgewidth=2, markersize=10)

                figname =  self.img_dir / 'source_amplitude' / f'source_location_{mode}{i:06}_epoch{epoch:04}.png'
                plt.savefig(figname, bbox_inches='tight')
                plt.close('all')

    def __visualize_histogram_location(self, epoch):
        """
        Fig.13a
        Histogram of errors in source location
        Histograms are normalized by the total counts.
        """

        figsize = (10, 10)
        filename = self.data_dir / f'source_info_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')
        bins = np.linspace(self.error_min, self.error_max, self.nb_bins)

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize)

            ref = ds[f'{mode}_ref'].values
            pred = ds[f'{mode}_pred'].values

            distance = np.sqrt((ref[:, 0]-pred[:, 0])**2 + (ref[:, 1]-pred[:, 1])**2)
            weights = np.ones_like(distance) / len(distance)
            
            ax.hist(distance, bins=bins, alpha=self.alpha, weights=weights, label=self.arch_name)

            average, std = np.mean( np.abs(distance) ), np.std( np.abs(distance) )
            print(f'error_distance, model: {self.arch_name}, mode: {mode}, average: {average:.2f}, std: {std:.2f}')

            ax.legend(loc='upper right', prop={'size': self.fontsize})
            ax.set_xlabel('Error Distance ' + r'$E_D$' + ' [m]', **self.axis_font)
            ax.grid(ls='dashed', lw=1)
            figname =  self.img_dir / 'histogram' / f'{self.arch_name}_{mode}_hist.png'
            fig.savefig(figname, bbox_inches='tight')
            
            plt.close('all')

    def __visualize_scatter_amplitude(self, epoch, amp_max=100, fac=1.5):
        """
        Fig.13b
        Scatter plot of reference and predicted source emission rates
        Perfect fitting and Factor of 1.5 lines are shown
        """

        figsize = (10, 10)
        filename = self.data_dir / f'source_info_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize)

            ref = ds[f'{mode}_ref_amp'].values
            pred = ds[f'{mode}_pred_amp'].values

            error = ref - pred
            relative_error = error / ref * 100 # In percent

            # reference line: factor of 1 and 1.5
            x = np.arange(100+1)
            ax.plot(x, x, '-k', label='Perfect Fitting')
            ax.plot(x, x*fac, '-r', label=r'${\rm FAC}_{1.5}$')
            ax.plot(x, x/fac, '-r')
            
            ax.scatter(ref, pred, c='r', label=self.arch_name, alpha=0.3, edgecolors='none')

            average, std = np.mean( np.abs(relative_error) ), np.std( np.abs(relative_error) )
            print(f'error_amplitude, model: {self.arch_name}, mode: {mode}, average: {average:.2f}, std: {std:.2f}')

            ax.legend(loc='upper right', prop={'size': self.fontsize}, scatterpoints=1)
            ax.set_xlabel('Reference Source Emission rate ' + r'$S_r$', **self.axis_font)
            ax.set_ylabel('Predicted Source Emission rate ' + r'$S_p$', **self.axis_font)
            ax.set_xlim([0, amp_max])
            ax.set_ylim([0, amp_max])
            ax.grid(ls='dashed', lw=1)
            figname =  self.img_dir / 'histogram' / f'{self.arch_name}_{mode}_scatter.png'
            fig.savefig(figname, bbox_inches='tight')
            
            plt.close('all')

    def __visualize_histogram_map(self, epoch):
        """
        Fig.14
        The average error distance in grid based historgrams
        Started points are locations of observation stations
        """

        figsize = (6, 8)
        filename = self.data_dir / f'source_info_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')

        # Create grid to compute histogram
        xmax, ymax = 160, 300
        xmax_fig, ymax_fig = 310, 568
        x = np.linspace(-xmax, xmax, 30)
        y = np.linspace(-ymax, ymax, 60)
        
        X, Y = np.meshgrid(x, y)
        
        dy, dx = y[1] - y[0], x[1] - x[0]

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'xticks':[], 'yticks':[]})

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
            cbar.set_label('Error Distance ' + r'$E_D$' + ' [m]', size=self.fontsize)

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

    def __visualize_flow_direction_map(self, epoch, threshold=10):
        """
        Fig.15
        Distribution of source locations for the large relative errors of source emission rates
        Source locations, observation stations and principal flow directions are shown
        """

        figsize = (6, 8)
        filename = self.data_dir / f'source_info_{self.arch_name}.nc'
        ds = xr.open_dataset(filename, engine='netcdf4')

        # Create grid to compute histogram
        xmax, ymax = 160, 300
        xmax_fig, ymax_fig = 310, 568

        for mode in self.modes:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'xticks':[], 'yticks':[]})

            # Reading sample data to get levelset and stations
            i = 0
            filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
            ds_sample = xr.open_dataset(filename, engine='netcdf4')
            levelset = ds_sample['levelset'].values

            ref_amp = ds[f'{mode}_ref_amp'].values
            pred_amp = ds[f'{mode}_pred_amp'].values
             
            error_amp = ref_amp - pred_amp
            relative_error = error_amp / ref_amp * 100
             
            large_error = np.abs(relative_error) > threshold
            if np.sum(large_error) == 0:
                raise ValueError(f'Threshold for large relative error is too large: {threshold}. Please try a smaller value')

            # Source locations with large relative errors
            ref = ds[f'{mode}_ref'].values[large_error]
            pred = ds[f'{mode}_pred'].values[large_error]
            flows = ds[f'{mode}_flows'].values[large_error]

            distance = np.sqrt((ref[:, 0]-pred[:, 0])**2 + (ref[:, 1]-pred[:, 1])**2)
            average, std = np.mean( np.abs(distance) ), np.std( np.abs(distance) )
            print(f'error_distance with large relative errors, model: {self.arch_name}, mode: {mode}, average: {average}, std: {std}')

            # Plot buildings
            ax.imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent)

            # Plot flow directions and locations (refs and preds)
            for pos_r, pos_p, flow in zip(ref, pred, flows):
                r_x, r_y = pos_r
                p_x, p_y = pos_p
                u, v = flow
                
                ax.plot(r_x, r_y, color='none', marker='*', markeredgecolor='g', markeredgewidth=1, markersize=12)
                
                # Add flow vectors
                ax.annotate(text='', xy=(r_x+u*30, r_y+v*30), xytext=(r_x, r_y), xycoords='data',\
                            arrowprops=dict(facecolor='blue', width=2.0,headwidth=7.0,headlength=7.0,shrink=0,alpha=0.5))

            # Add monitoring stations
            stations = ds_sample['station_positions'].values
            for position in stations: 
                st_x, st_y, _ = position
                ax.plot(st_x, st_y, color='none', marker='*', markeredgecolor='m', markeredgewidth=1, markersize=12)

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
            figname =  self.img_dir / 'histogram' / f'{self.arch_name}_{mode}_flow_map.png'
            fig.savefig(figname)
            plt.close('all')
