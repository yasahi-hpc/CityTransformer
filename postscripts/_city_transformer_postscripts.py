"""
Convert data and then visualize

Data Manupulation
1. Save metrics for validation and test data

Save figures
1. Loss curve
2. plume dispersion and errors
3. metrics
"""

import pathlib
import numpy as np
import xarray as xr
from numpy import ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib.colors import LogNorm
from ._base_postscript import _BasePostscripts
from .metrics import get_metric

class CityTransformerPostscripts(_BasePostscripts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformer'
        self.modes = ['val', 'test']
        self.threshold = 0.5
        self.clip = 1.e-8
        self.alpha = 0.9
        self.vmin = self.clip
        self.vmax = 1.0
        self.nb_bins = 100

        self.fig_names = ['loss', 'contour', 'metrics']
        self.extent = [-1024,1024,-1024,1024]

        self.metrics = {'FAC2',
                        'FAC5',
                        'MG',
                        'VG',
                        'NAD',
                        'FB',
                       }

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

    def __preprocess(self, epoch):
        for mode in self.modes:
            all_metrics = {metric_name: [] for metric_name in self.metrics}
            nb_shots = self.nb_shots_dict[mode]
            for i in range(nb_shots):
                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)

                levelset = ds['levelset'].values

                # Target metrics
                metric_dict = {'FAC2': {'factor': 2, 'levelset': levelset},
                               'FAC5': {'factor': 5, 'levelset': levelset},
                               'MG': {'levelset': levelset},
                               'VG': {'levelset': levelset},
                               'NAD': {'levelset': levelset},
                               'FB': {'levelset': levelset},
                              }

                evaluated_metrics = self.__evaluate_metrics(ds, metric_dict=metric_dict)
                for metric_name in metric_dict.keys():
                    all_metrics[metric_name].append(evaluated_metrics[metric_name])

            # Saving dataset
            data_vars = {}
            for metric_name, evaluated_values in all_metrics.items():
                data_vars[metric_name] = (['shot_idx'], np.asarray(evaluated_values))

            coords = {'shot_idx': np.arange(nb_shots)}
            filename = self.data_dir / f'{mode}_epoch{epoch:04}.nc'
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            ds.to_netcdf(filename)

    def __evaluate_metrics(self, ds, metric_dict):
        evaluated_metrics = {}

        pred, pred_binary = ds['pred_plume'].values.squeeze(), ds['pred_zeros_map'].values
        ref, ref_binary = ds['ref_plume'].values.squeeze(), ds['ref_zeros_map'].values
        levelset = ds['levelset'].values

        pred = self.__mask_img(img=pred, binary=pred_binary, levelset=levelset, threshold=self.threshold, clip=self.clip)
        ref  = self.__mask_img(img=ref,  binary=ref_binary, levelset=levelset, threshold=self.threshold, clip=self.clip)

        for metric_name, kwargs in metric_dict.items():
            metric = get_metric(metric_name)(**kwargs)
            evaluated_metrics[metric_name] = metric.evaluate(pred, ref)
             
        return evaluated_metrics

    def __mask_img(self, img, binary, levelset, threshold, clip, apply_mask=False):
        img, binary = np.squeeze(img), np.squeeze(binary)

        mask = np.logical_or(binary<threshold, levelset >= 0.)
        img = 10**img
        img =  np.where(mask, -1., img) * clip

        if apply_mask:
            return ma.masked_where(img <= 0, img)
        else:
            return img

    def __classification_by_factor(self, pred, ref, levelset, threshold, clip):
        """
        factor2   == 0
        factor5   == 0.5
        factor5++ == 1.0
        """

        if type(pred) is tuple:
            pred, pred_binary = pred
            ref, ref_binary = ref
        
        # Create mask based on zeros map and levelset
        def mask_on_img(img, binary):
            mask = np.logical_or(binary < threshold, levelset >= 0.)
            img = 10**img
            img = np.where(mask, -1, img) * clip
            return img

        pred = mask_on_img(pred, pred_binary)
        ref  = mask_on_img(ref, ref_binary)

        factor = np.ones_like(ref) # Default 1.0

        target_area = np.logical_and(ref > 0., levelset < 0)
        fraction = np.where(target_area, pred/ref, 0)

        fac2_area = np.logical_and( fraction >= 1/2., fraction <= 2. )
        fac5_area = np.logical_and( fraction >= 1/5., fraction <= 5. )
         
        fac2_area = np.logical_and(target_area, fac2_area)
        fac5_area = np.logical_and(target_area, fac5_area)
                
        factor[fac5_area] = np.ones_like(ref)[fac5_area] * 0.5
        factor[fac2_area] = np.zeros_like(ref)[fac2_area]
        
        correct_zeros = np.logical_and(pred_binary < 0.5, ref_binary < 0.5)
        masked_fraction = ma.masked_where(np.logical_or(correct_zeros, levelset >= 0.), factor)
        
        return masked_fraction

    def _visualize(self, epoch):
        self.data_dir = self.img_dir / 'metrics/data'
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
        super()._visualize_loss()
        self.__preprocess(epoch)
        self.__visualize_plume_dispersion(epoch)
        self.__visualize_metrics(epoch)

    def __visualize_plume_dispersion(self, epoch):
        figsize = (8, 8)
        for mode in self.modes:
            nb_shots = self.nb_shots_dict[mode]
            for i in range(nb_shots):
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                         subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.05))

                axes[1, 0].set_visible(False)

                filename = pathlib.Path(self.inference_dir) / mode / f'{mode}{i:06}_epoch{epoch:04}.nc'
                ds = xr.open_dataset(filename)
                levelset = ds['levelset'].values
                x, y = ds.attrs['release_x'], ds.attrs['release_y']

                # apply masks
                pred, pred_binary = ds['pred_plume'].values.squeeze(), ds['pred_zeros_map'].values
                ref, ref_binary = ds['ref_plume'].values.squeeze(), ds['ref_zeros_map'].values
                levelset = ds['levelset'].values
                factor = self.__classification_by_factor((pred, pred_binary), (ref, ref_binary), levelset=levelset, threshold=self.threshold, clip=self.clip)

                masked_pred = self.__mask_img(img=pred, binary=pred_binary, levelset=levelset, threshold=self.threshold, clip=self.clip, apply_mask=True)
                masked_ref  = self.__mask_img(img=ref,  binary=ref_binary, levelset=levelset, threshold=self.threshold, clip=self.clip, apply_mask=True)

                # Plotting the ground truth and prediction
                im = axes[0, 0].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent, interpolation='none')
                im = axes[0, 0].imshow(masked_ref, cmap='coolwarm', origin='lower', extent=self.extent, norm=LogNorm(vmin=self.vmin, vmax=self.vmax), alpha=self.alpha, interpolation='none')
                axes[0, 0].plot(x, y, color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=12)

                im = axes[0, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent, interpolation='none')
                im = axes[0, 1].imshow(masked_pred, cmap='coolwarm', origin='lower', extent=self.extent, norm=LogNorm(vmin=self.vmin, vmax=self.vmax), alpha=self.alpha, interpolation='none')
                axes[0, 1].plot(x, y, color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=12)

                # Plotting the factor map
                im2 = axes[1, 1].imshow(levelset < 0., cmap='gray', origin='lower', extent=self.extent, interpolation='none')
                im2 = axes[1, 1].imshow(factor, cmap='jet', origin='lower', extent=self.extent, vmin=0, vmax=1, alpha=self.alpha, interpolation='none')
                axes[1, 1].plot(x, y, color='none', marker='*', markeredgecolor='g', markeredgewidth=2, markersize=12)

                axes[0, 0].set_title('Ground Truth', **self.title_font)
                axes[0, 1].set_title(f'{self.arch_name}', **self.title_font)

                cbar  = fig.colorbar(im,  ax=axes[0, :])
                cbar2 = fig.colorbar(im2, ax=axes[1, :])

                cbar2.remove()
                
                figname = self.img_dir / 'contour' / f'log_{mode}{i:06}_epoch{epoch:04}.png'
                plt.savefig(figname, bbox_inches='tight')
                plt.close('all')

    def __visualize_metrics(self, epoch):
        figsize = (20, 12)
        plot_dict = {}
        # key: metric_name, value: xmin, xmax, ymin, ymax, label
        # xmin, xmax are also used to make histogram
        plot_dict['FAC2'] = (0, 1, 0, 0.05, 'FAC_2')
        plot_dict['FAC5'] = (0, 1, 0, 0.1, 'FAC_5')
        plot_dict['FB']   = (-2, 2, 0, 0.05, 'FB')
        plot_dict['NAD']  = (0, 0.15, 0, 0.15, 'NAD')
        plot_dict['MG']   = (0, 2, 0, 0.1, 'MG')
        plot_dict['VG']   = (1, 1.15, 0, 0.5, 'VG')

        metric_names = plot_dict.keys()

        for mode in self.modes:
            filename = self.data_dir / f'{mode}_epoch{epoch:04}.nc'
            ds = xr.open_dataset(filename)

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
            for metric_name, ax in zip(metric_names, axes.flatten()):
                xmin, xmax, ymin, ymax, label = plot_dict[metric_name]
                bins = np.linspace(xmin, xmax, self.nb_bins)
                metric = ds[metric_name].values
                weights = np.ones_like(metric) / len(metric)
                _hist, _bins, _patches = ax.hist(metric, bins=bins, alpha=0.5, weights=weights, label=self.arch_name)
                average = np.mean( np.abs(metric) )
                std = np.std( np.abs(metric) )
                print(f'model: {self.arch_name}, metric_name: {metric_name}, average: {average}, std: {std}')

                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_title(metric_name, **self.title_font)
                ax.legend(loc='upper right', prop={'size': self.fontsize*0.6})
                ax.grid(ls='dashed', lw=1)

            figname = self.img_dir / 'metrics' / f'metric_{self.arch_name}.png'
            plt.savefig(figname, bbox_inches='tight')
            plt.close('all')
