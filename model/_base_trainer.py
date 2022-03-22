import torch
import torch.multiprocessing as mp
import numpy as np
import xarray as xr
from torch import nn
import pathlib
import horovod.torch as hvd
from torch.utils.data import DataLoader
from collections import defaultdict
from .utils import Timer, MeasureMemory
from .city_dataset import CityDataset
from .city_transformer import CityTransformer
from ._img_saver import _CityTransformerImageSaver, _CityTransformerInverseImageSaver, save_loss
from ._data_saver import _CityTransformerDataSaver, _CityTransformerInverseDataSaver

class _BaseTrainer:
    """
    Base classs for training
    """

    def __init__(self, *args, **kwargs):
        self.losses = defaultdict(list)
        self.elapsed_times = defaultdict(list)
        self.memory_consumption = {}

        allowed_kwargs = {
                          'device',
                          'model_name',
                          'n_epochs',
                          'checkpoint_idx',
                          'seed',
                          'data_dir',
                          'out_dir',
                          'batch_size',
                          'lr',
                          'momentum',
                          'beta_1',
                          'beta_2',
                          'clip',
                          'version',
                          'loss_type',
                          'opt_type',
                          'activation',
                          'n_digits',
                          'n_precision_enhancers',
                          'n_freq_checkpoint',
                          'super_precision',
                          'use_adasum',
                          'gradient_predivide_factor',
                          'fp16_allreduce',
                          'UNet',
                          'inference_mode',
                         }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood: ', kwarg)

        data_dir = kwargs.get('data_dir')
        if not data_dir:
            raise ValueError('Argument data_dir must be given')
        self.data_dir = data_dir

        self.out_dir = kwargs.get('out_dir', './')
        self.device = kwargs.get('device', 'cuda')
        self.n_epochs = kwargs.get('n_epochs', 1)
        self.n_freq_checkpoint = kwargs.get('n_freq_checkpoint', 10)
        self.n_digits = kwargs.get('n_digits', 8)
        self.seed   = kwargs.get('seed', 0)
        self.batch_size = kwargs.get('batch_size', 16)
        self.version = kwargs.get('version', 0)
        self.lr = kwargs.get('lr', 0.0001)
        self.momentum = kwargs.get('momentum', 0.9)
        self.beta_1 = kwargs.get('beta_1', 0.9)
        self.beta_2 = kwargs.get('beta_2', 0.999)
        self.activation = kwargs.get('activation', 'ReLU')
        self.loss_type = kwargs.get('loss_type', 'MSE')
        self.opt_type = kwargs.get('opt_type', 'Adam')
        self.clip = kwargs.get('clip', '100')
        self.gradient_predivide_factor = kwargs.get('gradient_predivide_factor', 1.0)
        self.fp16_allreduce = kwargs.get('fp16_allreduce', False)
        self.use_adasum = kwargs.get('use_adasum', False)
        self.super_precision = kwargs.get('super_precision', False)
        self.n_precision_enhancers = kwargs.get('n_precision_enhancers', 0)
        self.UNet = kwargs.get('UNet', False)
        self.inference_mode = kwargs.get('inference_mode', False)
        self.checkpoint_idx = kwargs.get('checkpoint_idx', -1) # use last checkpoint

        self.timer = Timer(device=self.device)
        self.load_model = False
        self.mode_name = 'inference' if self.inference_mode else 'train'

        self.min_value = 10**(-(self.n_digits))
        self.epoch_start = 0
        self.run_number = 0

        if self.super_precision:
            raise ValueError('Please do not add --super_precision in your command line arguments. \
                              This functionality is suppressed which needs the multi-digits dataset. \
                              We do not publish multi-digits dataset due to the limitiatio of storage.')

    def initialize(self, *args, **kwargs):
        if self.inference_mode:
            self.initialize_inference(*args, **kwargs)
        else:
            self.initialize_train(*args, **kwargs)

    def initialize_train(self, *args, **kwargs):
        # Horovod: Initialize library
        hvd.init()
        torch.manual_seed(self.seed)

        if self.device == 'cuda':
            # Horovod: Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(self.seed)

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

        self.rank, self.size = hvd.rank(), hvd.size()
        self.master = self.rank == 0

        self.__prepare_dirs()

        # Normalization coefficents are also set
        self.train_loader, self.val_loader, self.test_loader = self.__get_dataloaders()
        self.criterion = self.__get_loss(self.loss_type)()

        # Get image saver
        self.img_saver = self._get_image_saver()

        self.model = self._get_model()
        self.model = self.model.to(self.device)

        # Optimizers
        lr_scaler = self.size if not self.use_adasum else 1
        if self.device == 'cuda' and self.use_adasum and hvd.nccl_build():
            lr_scaler = hvd.local_size()
        self.lr = self.lr * lr_scaler


        if self.opt_type == 'Adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        else:
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Horovod: braodcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.opt, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.fp16_allreduce else hvd.Compression.none
        
        # Horovod: wrap optimizer with DistributedOptimizer.
        self.opt = hvd.DistributedOptimizer(self.opt,
                                            named_parameters=self.model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Adasum if self.use_adasum else hvd.Average,
                                            gradient_predivide_factor=self.gradient_predivide_factor)

        device_name = 'cpu'
        if self.device == 'cuda':
            local_rank = hvd.local_rank()
            device_name = f'{self.device}:{local_rank}'
        self.memory = MeasureMemory(device=device_name)

        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def initialize_inference(self, *args, **kwargs):
        self.rank, self.size = 0, 1
        self.master = True

        self.__prepare_dirs()

        self.train_loader, self.val_loader, self.test_loader = self.__get_dataloaders()
        self.criterion = self.__get_loss(self.loss_type)()

        # Get data saver
        self.data_saver = self._get_data_saver()

        self.model = self._get_model(checkpoint_idx=self.checkpoint_idx)
        self.model = self.model.to(self.device)

        # Synchronize
        if self.device == 'cuda':
            torch.cuda.synchronize() # Waits for everything to finish running

    def run(self, *args, **kwargs):
        if self.inference_mode:
            self._run_inference(*args, **kwargs)
        else:
            self._run(*args, **kwargs)

    def _run(self, *args, **kwargs):
        self.timer.start()
        for epoch in range(1, self.n_epochs+1):
            total_epoch = int(epoch + self.epoch_start)
            
            if self.master:
                print(f'epoch = {total_epoch}')
            
            self.train_sampler.set_epoch(total_epoch)
            self.val_sampler.set_epoch(total_epoch)
            self.test_sampler.set_epoch(total_epoch)
            
            # Training
            with torch.enable_grad():
                self._train(data_loader=self.train_loader, epoch=total_epoch-1)
                            
            # Validation
            with torch.no_grad():
                self._test(data_loader=self.val_loader, epoch=total_epoch-1, mode='val')
            
            # Test
            with torch.no_grad():
                self._test(data_loader=self.test_loader, epoch=total_epoch-1, mode='test')
            
            # Checkpoint save state dict and meta data
            if epoch % self.n_freq_checkpoint == 0 or epoch == self.n_epochs:
                self._check_point(epoch=total_epoch-1)

    def _run_inference(self, *args, **kwargs):
        epoch = self.epoch_start - 1
        print('epoch', epoch)
        # Validation
        with torch.no_grad():
            self._infer(data_loader=self.val_loader, epoch=epoch, mode='val')
            self._infer(data_loader=self.test_loader, epoch=epoch, mode='test')
            
    def _train(self, data_loader, epoch):
        raise NotImplementedError()

    def _test(self, data_loader, epoch, mode):
        raise NotImplementedError()

    def _infer(self, data_loader, epoch, mode):
        raise NotImplementedError()

    def finalize(self, *args, **kwargs):
        self._finalize(*args, **kwargs)

    def _finalize(self, *args, **kwargs):
        seconds = kwargs.get('seconds')

        log_filename = pathlib.Path(self.out_dir) / f'log_{self.mode_name}_{self.run_number:03}.txt'

        if self.master:
            if self.mode_name == 'train':
                save_loss(loss_data_dir=self.sub_checkpoint_dir, 
                          img_dir=self.fig_dir, run_number = self.run_number, vmin=0, vmax=0.01)
                message = f'It took {seconds} [s] to train for {self.n_epochs} epochs'
            else:
                message = f'It took {seconds} [s] to infer'

            with open(log_filename, 'w') as f:
                print(message, file=f)
                checkpoint_found = self.__find_checkpoint(self.checkpoint_idx)
                if checkpoint_found: 
                    ds = xr.open_dataset(self.checkpoint, engine='netcdf4')
                    print(ds, file=f)

    def _check_point(self, epoch):
        self.timer.stop()
        elapsed_seconds = self.timer.elapsed_seconds()
        checkpoint_files = list(self.sub_checkpoint_dir.glob('checkpoint*.nc'))

        idx = len(checkpoint_files)

        # Set file names
        next_checkpoint_file_name = self.sub_checkpoint_dir / f'checkpoint{idx:03}.nc'
        current_state_file_name = self.sub_checkpoint_dir / f'model_checkpoint{idx:03}.pt'

        attrs = {}
        if idx > 0:
            previous_checkpoint_file_name  = self.sub_checkpoint_dir / f'checkpoint{idx-1:03}.nc'
            if not previous_checkpoint_file_name.is_file():
                raise FileNotFoundError(f'{prev_result_filename} does not exist')

            ds = xr.open_dataset(previous_checkpoint_file_name, engine='netcdf4')
            attrs = ds.attrs.copy()
            epoch_end = attrs['epoch_end']
            attrs['last_state_file'] = str(current_state_file_name)
            attrs['epoch_start'] = epoch_end + 1
            attrs['epoch_end'] = epoch
            attrs['elapsed_time'] = elapsed_seconds
            attrs['run_number'] = self.run_number

        else:
            # Then first checkpoint
            attrs = {}
            attrs['model_name'] = self.model_name
            attrs['last_state_file'] = str(current_state_file_name)
            attrs['nb_train'] = len(self.train_loader.dataset)
            attrs['nb_val']   = len(self.val_loader.dataset)
            attrs['nb_test']  = len(self.test_loader.dataset)
            attrs['device'] = self.device
            attrs['learning_rate'] = self.lr
            attrs['clip'] = self.clip
            attrs['batch_size'] = self.batch_size
            attrs['version'] = self.version
            attrs['nb_procs'] = self.size
            attrs['reserved'] = self.memory_consumption['reserved']
            attrs['alloc'] = self.memory_consumption['alloc']
            attrs['epoch_start'] = 0
            attrs['epoch_end'] = epoch
            attrs['super_precision'] = int(self.super_precision)
            attrs['UNet'] = int(self.UNet)
            attrs['n_precision_enhancers'] = self.n_precision_enhancers
            attrs['opt_type'] = self.opt_type
            attrs['loss_type'] = self.loss_type
            attrs['elapsed_time'] = elapsed_seconds
            attrs['run_number'] = self.run_number

        data_vars = {}
        data_vars['train_losses'] = (['epochs'], self.losses['train'])
        data_vars['val_losses']   = (['epochs'], self.losses['val'])
        data_vars['test_losses']  = (['epochs'], self.losses['test'])

        n_epochs = attrs['epoch_end'] - attrs['epoch_start'] + 1
        coords = {}
        coords['epochs'] = np.arange(attrs['epoch_start'], attrs['epoch_end']+1)

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Save meta data
        ds.to_netcdf(next_checkpoint_file_name, engine='netcdf4')

        # Save model
        torch.save(self.model.state_dict(), current_state_file_name)

        # Initialize loss dict after saving
        self.losses = defaultdict(list)

        # Start timer again
        self.timer.start()

    def __prepare_dirs(self):
        out_dir = pathlib.Path(self.out_dir)
        self.checkpoint_dir = out_dir / 'checkpoint'

        if self.inference_mode:
            self.inference_dir = pathlib.Path('inference')
            if not self.inference_dir.exists():
                self.inference_dir.mkdir(parents=True)

            self.sub_checkpoint_dir = self.checkpoint_dir / f'rank{self.rank}'
            if not self.sub_checkpoint_dir.exists():
                raise IOError('checkpoint directory not found')

        else:
            self.fig_dir = out_dir / 'imgs'

            if self.master:
                if not self.checkpoint_dir.exists():
                    self.checkpoint_dir.mkdir(parents=True)

                if not self.fig_dir.exists():
                    self.fig_dir.mkdir(parents=True)

            # Barrier
            hvd.allreduce(torch.tensor(1), name='Barrier')
            self.sub_checkpoint_dir = self.checkpoint_dir / f'rank{self.rank}'
            if not self.sub_checkpoint_dir.exists():
                self.sub_checkpoint_dir.mkdir(parents=True)

            self.sub_fig_dir = self.fig_dir / f'rank{self.rank}'
            if not self.sub_fig_dir.exists():
                self.sub_fig_dir.mkdir(parents=True)

    def __get_dataloaders(self):
        modes = ['train', 'val', 'test']
        train_dir, val_dir, test_dir = [pathlib.Path(self.data_dir) / mode for mode in modes]
        
        dataset_dict = {
                        'device': self.device,
                        'version': self.version,
                        'super_precision': self.super_precision,
                        'n_digits': self.n_digits,
                        'n_precision_enhancers': self.n_precision_enhancers,
                        'inference_mode': self.inference_mode,
                       }

        if self.inference_mode:
            # Do not use train dataset in inference
            val_dataset   = CityDataset(path=val_dir, **dataset_dict)
            test_dataset  = CityDataset(path=test_dir, **dataset_dict)

            train_loader = None
            val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        else:
            train_dataset = CityDataset(path=train_dir, **dataset_dict)
            val_dataset   = CityDataset(path=val_dir, **dataset_dict)
            test_dataset  = CityDataset(path=test_dir, **dataset_dict)

            # Horovod: use Distributed Sampler to partition the training data
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.size, rank=self.rank)
            self.val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=self.size, rank=self.rank, shuffle=False)
            self.test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=self.size, rank=self.rank, shuffle=False)

            kwargs = {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {}
            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork'
            # issues with Infiniband implementations that are not fork-safe
            if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                kwargs['multiprocessing_context'] = 'forkserver'
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, **kwargs)
            val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, sampler=self.val_sampler, **kwargs)
            test_loader  = DataLoader(test_dataset, batch_size=self.batch_size, sampler=self.test_sampler, **kwargs)

        norm_dict = test_dataset.norm_dict.copy()
        self.norm_dict = norm_dict
        self.resolution_dict = test_dataset.resolution_dict.copy()
        self.station_positions = test_dataset.station_positions

        self.imgs_max, self.imgs_min = norm_dict['imgs_max'], norm_dict['imgs_min']
        self.concentrations_max, self.concentrations_min = norm_dict['concentrations_max'], norm_dict['concentrations_min']
        self.series_max, self.series_min = norm_dict['series_max'], norm_dict['series_min']
        self.release_sdf_max, self.release_sdf_min = norm_dict['release_sdf_max'], norm_dict['release_sdf_min']

        return train_loader, val_loader, test_loader 

    def _get_model(self, checkpoint_idx=-1):
        model_dict = {
                      'in_channels': self.in_channels,
                      'out_channels': self.out_channels,
                      'version': self.version,
                      'activation': self.activation,
                      'super_precision': self.super_precision,
                      'n_precision_enhancers': self.n_precision_enhancers,
                      'UNet': self.UNet,
                     }
          
        model = CityTransformer(**self.resolution_dict, **model_dict)
         
        # Restart or inference
        self.load_model = self.__find_checkpoint(checkpoint_idx)
        if self.load_model:
            ds = xr.open_dataset(self.checkpoint, engine='netcdf4')
            if ds.attrs['model_name'] != self.model_name:
                raise IOError('Error in model load. Loading different type of model')

            last_run_number = ds.attrs['run_number']
            last_state_file = ds.attrs['last_state_file']
            print(f'Loading {last_state_file}\n')
            model.load_state_dict( torch.load(last_state_file) )
            self.epoch_start = int(ds.attrs['epoch_end']) + 1
            self.run_number = last_run_number + 1
             
        return model

    def _get_image_saver(self):
        if self.model_name == 'CityTransformer':
            image_saver = _CityTransformerImageSaver(out_dir=self.sub_fig_dir, 
                                                     clip=self.min_value,
                                                     vmin=self.min_value,
                                                     vmax=1.0,
                                                     cmap='hot',
                                                     super_precision=self.super_precision,
                                                     n_precision_enhancers=self.n_precision_enhancers)

        elif self.model_name == 'CityTransformerInverse':
            image_saver = _CityTransformerInverseImageSaver(out_dir=self.sub_fig_dir,
                                                            vmin=0,
                                                            vmax=2000,
                                                            cmap='seismic')

        else:
            raise ValueError('model should be either CityTransformer or CityTransformerInverse')

        return image_saver

    def _get_data_saver(self):
        if self.model_name == 'CityTransformer':
            data_saver = _CityTransformerDataSaver(out_dir=self.inference_dir,
                                                   clip=self.min_value,
                                                   version=self.version,
                                                   station_positions=self.station_positions,
                                                   norm_dict=self.norm_dict,
                                                   num_stations=self.resolution_dict['num_stations']
                                                   )

        elif self.model_name == 'CityTransformerInverse':
            data_saver = _CityTransformerInverseDataSaver(out_dir=self.inference_dir,
                                                          clip=self.min_value,
                                                          version=self.version,
                                                          station_positions=self.station_positions,
                                                          norm_dict=self.norm_dict,
                                                          num_stations=self.resolution_dict['num_stations']
                                                         )

        else:
            raise ValueError('model should be either CityTransformer or CityTransformerInverse')

        return data_saver

    def __get_loss(self, loss_type):
        loss_type_lower = loss_type.lower()
        if loss_type_lower in ['l1', 'mae']:
            return nn.L1Loss
        elif loss_type_lower in ['l2', 'mse']:
            return nn.MSELoss
        elif loss_type_lower in ['bce']:
            return nn.BCELoss
        else:
            raise NotImplementedError(f'loss_type {loss_type} is not implemented')

    def __find_checkpoint(self, checkpoint_idx=-1):
        checkpoint_files = list(self.sub_checkpoint_dir.glob('checkpoint*.nc'))

        if checkpoint_idx==-1:
            if not checkpoint_files:
                self.checkpoint = None
                return False
        else:
            # Then inference mode with specified checkpoint file
            if not (checkpoint_idx < len(checkpoint_files)):
                raise ValueError(f'specified checkpoint idx {checkpoint_idx} is out of range')

        checkpoint_files = sorted(checkpoint_files)
        self.checkpoint = checkpoint_files[checkpoint_idx]
        return True

    def __normalize(self, x, *args):
        """
        Set data range to be [0, 1] or [-1, 1]
        """

        xmax, xmin, scale = args

        if not scale in [1, 2]:
            raise ValueError('scale should be either 1 or 2')

        x -= xmin
        x /= (xmax - xmin)

        if scale == 2:
            x = scale * (x - 0.5)

        return x

    def __denormalize(self, x, *args):
        """
        Revert data range from [0, 1] or [-1, 1] to original range
        """

        xmax, xmin, scale = args
        if not scale in [1, 2]:
            raise ValueError('scale should be either 1 or 2')

        if scale == 2:
            x = x / scale + 0.5

        x = x * (xmax - xmin) + xmin

        return x

    def _preprocess(self, x, *args):
        return self.__normalize(x, *args, 2)
        
    def _postprocess(self, x, *args):
        return self.__denormalize(x, *args, 2)

    def _metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()
