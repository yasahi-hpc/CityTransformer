import abc
import pathlib

class _BaseSaver(abc.ABC):
    """
    Base class to save figures or data
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        allowed_kwargs = {
                          'out_dir',
                          'modes',
                          'clip',
                          'alpha',
                          'version',
                          'vmin',
                          'vmax',
                          'cmap',
                          'super_precision',
                          'n_precision_enhancers',
                          'norm_dict',
                          'station_positions',
                          'n_stations',
                          'nz',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        self.out_dir = kwargs.get('out_dir')
        self.out_dir = pathlib.Path(self.out_dir)
        self.modes = kwargs.get('modes', ['train', 'val', 'test'])

        # Make required directories
        sub_out_dirs = [self.out_dir / mode for mode in self.modes]
        for sub_out_dir in sub_out_dirs:
            if not sub_out_dir.exists():
                sub_out_dir.mkdir(parents=True)

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError()
