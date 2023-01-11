import abc
import pathlib
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style

class _BasePostscripts(abc.ABC):
    """
    Base class for postscript
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.out_dir = kwargs.get('out_dir', './')
        self.checkpoint_idx = kwargs.get('checkpoint_idx', -1)

    def initialize(self, *args, **kwargs):
        self.__prepare_dirs()

        # Find checkpoint and set the epoch
        self.checkpoint = self.__get_checkpoint(checkpoint_idx=self.checkpoint_idx)

        # Set meta data based on checkpoint file
        self.__set_metadata(self.checkpoint)
    
    def run(self, *args, **kwargs):
        print(f'Post-process for epoch {self.epoch}')
        self._visualize(self.epoch)

    def finalize(self, *args, **kwargs):
        seconds = kwargs.get('seconds')

        log_filename = pathlib.Path(self.out_dir) / f'log_postscript.txt'
        message = f'It took {seconds} [s] to postprocess'

        with open(log_filename, 'w') as f:
            print(message, file=f)
            ds = xr.open_dataset(self.checkpoint, engine='netcdf4')
            print(ds, file=f)

    @abc.abstractmethod
    def _visualize(self, epoch):
        raise NotImplementedError()

    def __prepare_dirs(self):
        out_dir = pathlib.Path(self.out_dir)

        # Check the existence of inference and checkpoint dir
        self.inference_dir = out_dir / 'inference'
        if not self.inference_dir.exists():
            raise IOError('Inference dir not found')

        self.checkpoint_dir = out_dir / 'checkpoint/rank0'
        if not self.checkpoint_dir.exists():
            raise IOError('Checkpoint dir not found')

        # Prepare directories to store final images
        self.img_dir = out_dir / 'postprocessed_imgs'

        for fig_name in self.fig_names:
            sub_img_dir = self.img_dir / fig_name

            if not sub_img_dir.exists():
                sub_img_dir.mkdir(parents=True)

    def __get_checkpoint(self, checkpoint_idx=-1):
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint*.nc'))

        if checkpoint_idx==-1:
            if not checkpoint_files:
                raise ValueError(f'checkpoint not found')
        else:
            # Then inference mode with specified checkpoint file
            if not (checkpoint_idx < len(checkpoint_files)):
                raise ValueError(f'specified checkpoint idx {checkpoint_idx} is out of range')

        checkpoint_files = sorted(checkpoint_files)

        return checkpoint_files[checkpoint_idx]

    def __set_metadata(self, checkpoint):
        ds = xr.open_dataset(checkpoint, engine='netcdf4')

        if ds.attrs['model_name'] != self.model_name:
            raise IOError('Error in model load. Loading different type of model')

        version = ds.attrs['version']
        self.arch_name = 'Transformer' if version == 0 else 'MLP'

        self.epoch = int(ds.attrs['epoch_end'])

        self.nb_shots_dict = {'val': ds.attrs['nb_val'],
                              'test': ds.attrs['nb_test'],}

    def _visualize_loss(self, vmin=0, vmax=0.01):
        mpl.style.use('classic')
        fontsize = 32
        fontname = 'Times New Roman'
        plt.rc('xtick', labelsize=fontsize)
        plt.rc('ytick', labelsize=fontsize)
        plt.rc('font', family=fontname)
        
        axis_font = {'fontname': fontname, 'size': fontsize}
        title_font = {'fontname': fontname, 'size': fontsize, 'color': 'black',
                      'verticalalignment':'bottom'}

        # Loaded results
        result_files = sorted(list(self.checkpoint_dir.glob('checkpoint*.nc')))
        ds = xr.open_mfdataset(result_files, concat_dim='epochs', compat='no_conflicts', combine='nested')

        loss_type = ds.attrs['loss_type']
        epochs = ds['epochs'].values
        modes = {}
        modes['train'] = ('r-', 'Training')
        modes['val'] = ('b-', 'Validation')
        modes['test'] = ('g-', 'Test')
        
        fig, ax = plt.subplots(figsize=(12,12))
        for mode, (ls, name) in modes.items():
            losses = ds[f'{mode}_losses'].values
            ax.plot(epochs, losses, ls, lw=3, label=name)

        ax.set_xlabel('epochs', **axis_font)
        loss_label = f'{loss_type} loss'
        ax.set_ylabel(loss_label, **axis_font)
        ax.set_ylim(ymin=vmin, ymax=vmax)
        ax.legend(prop={'size':fontsize*1.2})
        ax.grid(ls='dashed', lw=1)
        fig.tight_layout()

        img_dir = self.img_dir / 'loss'
        fig.savefig(img_dir / f'loss.png')
        plt.close('all')
