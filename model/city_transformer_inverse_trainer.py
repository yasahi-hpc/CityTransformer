import torch
from ._base_trainer import _BaseTrainer

class CityTransformerInverseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformerInverse'
        self.in_channels = 1
        self.out_channels = 2 # (distance function and source amplitude)
        self.nb_source_amps = 10
        self.randomize_source_amps = True

    def _train(self, data_loader, epoch):
        mode = 'train'
        self.model.train()
        total_loss = 0.
        for i, (imgs, _, series, release_points) in enumerate(data_loader):
            imgs, series = imgs.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,2].to('cpu')
            
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            # (distance function and source amplitude)
            ref_distance_and_amplitude = imgs[:,:2]
            levelset = torch.unsqueeze(imgs[:,2], 1)
            
            self.model.zero_grad()
            
            # (distance function and source amplitude)
            pred_distance_and_amplitude = self.model(levelset, series)
            loss = self.criterion(pred_distance_and_amplitude, ref_distance_and_amplitude)
            
            self.model.zero_grad()

            if 'reserved' not in self.memory_consumption:
                self.memory.measure()
                self.memory_consumption['reserved'] = self.memory.reserved
                self.memory_consumption['alloc']    = self.memory.allocated

            loss.backward()
            # cilp_grad_norm helps prevent the exploding gradient problem in Transformer
            if self.version in [0, 1, 2]:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.opt.step()
             
            total_loss += loss.item() / len(data_loader.sampler)

            if i == 0:
                ref_distance_and_amplitude  = super()._postprocess(ref_distance_and_amplitude.detach(),  self.distance_and_source_max, self.distance_and_source_min)
                pred_distance_and_amplitude = super()._postprocess(pred_distance_and_amplitude.detach(), self.distance_and_source_max, self.distance_and_source_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=ref_distance_and_amplitude[:, 0],
                                    pred=pred_distance_and_amplitude[:, 0],
                                    epoch=epoch,
                                    mode=mode)

        train_loss = super()._metric_average(total_loss, f'{mode}_loss')
        self.losses[mode].append(train_loss)

    def _test(self, data_loader, epoch, mode):
        self.model.eval()
        total_loss = 0.
        for i, (imgs, _, series, release_points) in enumerate(data_loader):
            imgs, series = imgs.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,2].to('cpu')
            
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)

            ref_distance_and_amplitude = imgs[:,:2]
            levelset = torch.unsqueeze(imgs[:,2], 1)
            
            # (distance function and source amplitude)
            pred_distance_and_amplitude = self.model(levelset, series)
            loss = self.criterion(pred_distance_and_amplitude, ref_distance_and_amplitude)
            
            total_loss += loss.item() / len(data_loader.sampler)

            if i == 0:
                ref_distance_and_amplitude  = super()._postprocess(ref_distance_and_amplitude.detach(),  self.distance_and_source_max, self.distance_and_source_min)
                pred_distance_and_amplitude = super()._postprocess(pred_distance_and_amplitude.detach(), self.distance_and_source_max, self.distance_and_source_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=ref_distance_and_amplitude[:, 0],
                                    pred=pred_distance_and_amplitude[:, 0],
                                    epoch=epoch,
                                    mode=mode)

        test_loss = super()._metric_average(total_loss, f'{mode}_loss')
        self.losses[mode].append(test_loss)

    def _infer(self, data_loader, epoch, mode):
        self.model.eval()

        for i, (indices, imgs, concentrations, series, release_points, flows_and_sources) in enumerate(data_loader):
            imgs, concentrations, series = imgs.to(self.device), concentrations.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,2].to('cpu')

            # Normalization
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            # Keeping release data
            ref_distance_and_amplitude = imgs[:,:2]
            levelset = torch.unsqueeze(imgs[:,2], 1)
            
            # (distance function and source amplitude)
            pred_distance_and_amplitude = self.model(levelset, series)
            
            # De-normalize
            ref_distance_and_amplitude  = super()._postprocess(ref_distance_and_amplitude.detach(),  self.distance_and_source_max, self.distance_and_source_min)
            pred_distance_and_amplitude = super()._postprocess(pred_distance_and_amplitude.detach(), self.distance_and_source_max, self.distance_and_source_min)
            series                      = super()._postprocess(series, self.series_max, self.series_min)

            self.data_saver.save(levelset=levelset_cpu,
                                 release_points=release_points,
                                 flows_and_sources=flows_and_sources,
                                 ref_distance_and_amplitude=ref_distance_and_amplitude,
                                 pred_distance_and_amplitude=pred_distance_and_amplitude,
                                 indices=indices,
                                 series=series,
                                 mode=mode,
                                 epoch=epoch)
