import torch
from ._base_trainer import _BaseTrainer

class CityTransformerInverseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformerInverse'
        self.in_channels = 1
        self.out_channels = 1

    def _train(self, data_loader, epoch):
        mode = 'train'
        self.model.train()
        total_loss = 0.
        for i, (imgs, _, series, release_points) in enumerate(data_loader):
            imgs, series = imgs.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,1].to('cpu')
            
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            ref_release = torch.unsqueeze(imgs[:,0], 1)
            levelset = torch.unsqueeze(imgs[:,1], 1)
            
            self.model.zero_grad()
            
            pred_release = self.model(levelset, series)
            loss = self.criterion(pred_release, ref_release)
            
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
                ref_release = super()._postprocess(ref_release.detach(), self.release_sdf_max, self.release_sdf_min)
                pred_release = super()._postprocess(pred_release.detach(), self.release_sdf_max, self.release_sdf_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=ref_release,
                                    pred=pred_release,
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
            levelset_cpu = imgs[:,1].to('cpu')
            
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            ref_release = torch.unsqueeze(imgs[:,0], 1)
            levelset = torch.unsqueeze(imgs[:,1], 1)
            
            pred_release = self.model(levelset, series)
            loss = self.criterion(pred_release, ref_release)
            
            total_loss += loss.item() / len(data_loader.sampler)

            if i == 0:
                ref_release = super()._postprocess(ref_release.detach(), self.release_sdf_max, self.release_sdf_min)
                pred_release = super()._postprocess(pred_release.detach(), self.release_sdf_max, self.release_sdf_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=ref_release,
                                    pred=pred_release,
                                    epoch=epoch,
                                    mode=mode)

        test_loss = super()._metric_average(total_loss, f'{mode}_loss')
        self.losses[mode].append(test_loss)

    def _infer(self, data_loader, epoch, mode):
        self.model.eval()

        for i, (indices, imgs, concentrations, series, release_points, flow_directions) in enumerate(data_loader):
            imgs, concentrations, series = imgs.to(self.device), concentrations.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,1].to('cpu')

            # Normalization
            imgs   = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            # Keeping release data
            ref_release = torch.unsqueeze(imgs[:,0], 1)
            levelset    = torch.unsqueeze(imgs[:,1], 1)
            
            pred_release = self.model(levelset, series)
            
            # De-normalize
            ref_release  = super()._postprocess(ref_release.detach(), self.release_sdf_max, self.release_sdf_min)
            pred_release = super()._postprocess(pred_release.detach(), self.release_sdf_max, self.release_sdf_min)
            series       = super()._postprocess(series, self.series_max, self.series_min)

            self.data_saver.save(levelset=levelset_cpu,
                                 release_points=release_points,
                                 flow_directions=flow_directions,
                                 ref_release=ref_release,
                                 pred_release=pred_release,
                                 indices=indices,
                                 series=series,
                                 mode=mode,
                                 epoch=epoch)
