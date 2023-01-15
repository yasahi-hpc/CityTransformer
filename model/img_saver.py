import pathlib
import numpy as np
import xarray as xr
from numpy import ma
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib.colors import LogNorm
from ._base_saver import _BaseSaver

def save_loss(loss_data_dir, img_dir, run_number, vmin=0, vmax=0.01):
    mpl.style.use('classic')
    fontsize = 32
    fontname = 'Times New Roman'
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font', family=fontname)
     
    axis_font = {'fontname': fontname, 'size': fontsize}
    title_font = {'fontname': fontname, 'size': fontsize, 'color': 'black',
                  'verticalalignment':'bottom'}
     
    # Load results
    result_files = sorted(list(loss_data_dir.glob(f'checkpoint*.nc')))
    ds = xr.open_mfdataset(result_files, concat_dim='epochs', compat='no_conflicts', combine='nested')

    loss_type = ds.attrs['loss_type']
    epochs = ds['epochs'].values
    modes = {}
    modes['train'] = ('r-', r'${\rm Training}$')
    modes['val'] = ('b-', r'${\rm Validation}$')
    modes['test'] = ('g-', r'${\rm Test}$')
     
    fig, ax = plt.subplots(figsize=(12,12))
    for mode, (ls, name) in modes.items():
        losses = ds[f'{mode}_losses'].values
        ax.plot(epochs, losses, ls, lw=3, label=name)
    
    ax.set_xlabel(r'${\rm epochs}$', **axis_font)
    loss_label = r'${\rm MSE}$ ${\rm loss}$' if loss_type == 'MSE' else r'${\rm MAE}$ ${\rm loss}$'
    ax.set_ylabel(loss_label, **axis_font)
    ax.set_ylim(ymin=vmin, ymax=vmax)
    ax.legend(prop={'size':fontsize*1.2})
    ax.grid(ls='dashed', lw=1)
    fig.tight_layout()
    fig.savefig(img_dir / f'loss_{run_number}.png')
    plt.close('all')

def to_numpy(var):
    return np.squeeze(var.numpy()) if var.device == 'cpu' else  np.squeeze(var.cpu().numpy())

class CityTransformerImageSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip = kwargs.get('clip')
        self.super_precision = kwargs.get('super_precision', False)
        self.n_precision_enhancers = kwargs.get('n_precision_enhancers', 0)
        self.criteria = kwargs.get('criteria', 0.5)
        self.alpha = kwargs.get('alpha', 0.5)
        self.vmin = kwargs.get('vmin')
        self.vmax = kwargs.get('vmax')
        self.cmap = kwargs.get('cmap', 'hot')

        mpl.style.use('classic')

        self.fontsize = 36
        self.fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=self.fontsize)
        plt.rc('ytick', labelsize=self.fontsize)
        plt.rc('font', family=self.fontname)
         
        self.title_font = {'fontname':self.fontname, 'size':self.fontsize, 'color':'black',
                           'verticalalignment':'bottom'}
        self.axis_font = {'fontname':self.fontname, 'size':self.fontsize}
        self.xmax = 1024
        self.ymax = 1024
        self.extent = [-self.xmax, self.xmax, -self.ymax, self.ymax]

    def save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        release_points = kwargs.get('release_points')
        ref = kwargs.get('ref')
        pred = kwargs.get('pred')
        mode = kwargs.get('mode')
        epoch = kwargs.get('epoch')
        n_cols = kwargs.get('n_cols', 4)


        for i_precision in range(self.n_precision_enhancers+1):
            self.__save_images(i_precision, levelset, release_points, ref, mode, n_cols, 'ref', epoch)
            self.__save_images(i_precision, levelset, release_points, pred, mode, n_cols, 'pred', epoch)

    def __save_images(self, i_precision, levelset, release_points, imgs, mode, n_cols, name, epoch):
        # First save images
        if type(imgs) is tuple:
            imgs, zeros_map = imgs

        assert imgs.shape[1] == self.n_precision_enhancers+1

        # Access to the specified precision
        imgs = imgs[:, i_precision]

        imgs, zeros_map = to_numpy(imgs), to_numpy(zeros_map)
        levelset = to_numpy(levelset)
        release_points = to_numpy(release_points)

        ## Creaet mask based on binary_map and levelset
        mask = np.logical_or(zeros_map < self.criteria, levelset >= 0.)
        imgs = 10**imgs
        imgs = np.where(mask, -1, imgs) * self.clip

        n_samples = len(imgs)
        n_rows = n_samples // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24,24), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))

        for i, ax in np.ndenumerate(axes.ravel()):
            masked_imgs = ma.masked_where(imgs[i] <= 0, imgs[i])
            im = ax.imshow(levelset[i] < 0., cmap='gray', origin='lower', extent=self.extent, interpolation='none') 
            im = ax.imshow(masked_imgs, cmap=self.cmap, origin='lower', extent=self.extent, alpha=self.alpha, norm=LogNorm())

            # Add source locations
            x_, y_ = release_points[i]
            ax.plot(x_, y_, '*', markersize=10)

        # Set title and filename
        title = f'{name} (epoch = {epoch:03})'
        filename = f'{mode}_{name}_refine{i_precision}_epoch{epoch:03}.png'

        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title, **self.title_font, y=0.9)
        fig_dir = self.out_dir / mode
        fig.savefig(fig_dir / filename)
        plt.close('all')

class CityTransformerInverseImageSaver(_BaseSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmin = kwargs.get('vmin', 0)
        self.vmax = kwargs.get('vmax', 2000)
        self.cmap = kwargs.get('cmap', 'seismic')

        mpl.style.use('classic')

        self.fontsize = 36
        self.fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=self.fontsize)
        plt.rc('ytick', labelsize=self.fontsize)
        plt.rc('font', family=self.fontname)
         
        self.title_font = {'fontname':self.fontname, 'size':self.fontsize, 'color':'black',
                           'verticalalignment':'bottom'}
        self.axis_font = {'fontname':self.fontname, 'size':self.fontsize}

        self.cmap = 'seismic'

        self.xmax = 1024
        self.ymax = 1024
        self.alpha = 0.5
        self.extent = [-self.xmax, self.xmax, -self.ymax, self.ymax]

    def save(self, *args, **kwargs):
        levelset = kwargs.get('levelset')
        release_points = kwargs.get('release_points')
        ref = kwargs.get('ref')
        pred = kwargs.get('pred')
        mode = kwargs.get('mode')
        epoch = kwargs.get('epoch')
        n_cols = kwargs.get('n_cols', 4)

        data_dict = {'ref': ref,
                     'pred': pred,}

        print('ref.shape', ref.shape)
        print('pred.shape', pred.shape)
         
        for name, data in data_dict.items():
            self.__save_images(levelset, release_points, data, mode, n_cols, name, epoch)

    def __save_images(self, levelset, release_points, imgs, mode, n_cols, name, epoch):
        imgs = to_numpy(imgs)
        levelset = to_numpy(levelset)
        release_points = to_numpy(release_points)
        
        n_samples = len(imgs)
        n_rows = n_samples // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24,24), subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
        
        for i, ax in np.ndenumerate(axes.ravel()):
            im = ax.imshow(levelset[i] < 0., cmap='gray', origin='lower', extent=self.extent, interpolation='none')
            im = ax.imshow(imgs[i], cmap=self.cmap, origin='lower', extent=self.extent, alpha=self.alpha, vmin=self.vmin, vmax=self.vmax)
            
            # Add source locations
            x_, y_ = release_points[i]
            
            # Estimated
            x_pred, y_pred = self.__pred_source_location(imgs[i])
            
            ax.plot(x_, y_, color='none', marker='*', markeredgecolor='r', markeredgewidth=2, markersize=12)
            ax.plot(x_pred, y_pred, color='none', marker='^', markeredgecolor='b', markeredgewidth=2, markersize=12)
            
        title = f'{name} (epoch = {epoch:03})'
        filename = f'{mode}_{name}_epoch{epoch}.png'
        
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(title, **self.title_font, y=0.9)
        
        fig_dir = self.out_dir / mode
        fig.savefig(fig_dir / filename)
        
        plt.close('all')

    def __pred_source_location(self, distance_function):
        # Get the index where the distance function takes the minimum value
        ny, nx = distance_function.shape
        x, y = np.linspace(-self.xmax, self.xmax, nx), np.linspace(-self.ymax, self.ymax, ny)
        idx_y, idx_x = np.unravel_index(np.argmin(distance_function, axis=None), (ny, nx))
        return x[idx_x], y[idx_y]
