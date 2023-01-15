import torch
from ._base_trainer import _BaseTrainer

class CityTransformerTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'CityTransformer'
        self.in_channels = 2
        self.out_channels = 2
        self.nb_source_amps = 1
        self.randomize_source_amps = False

    def _train(self, data_loader, epoch):
        mode = 'train'
        self.model.train()
        total_loss = 0.
        for i, (imgs, concentrations, series, release_points) in enumerate(data_loader):
            imgs, concentrations, series = imgs.to(self.device), concentrations.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,1].to('cpu')

            ref_plume = concentrations[:, :-1]
            zeros_map = torch.unsqueeze(concentrations[:, -1], 1)
            
            imgs      = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            ref_plume = super()._preprocess(ref_plume, self.concentrations_max, self.concentrations_min)
            concentrations = torch.cat([ref_plume, zeros_map], axis=1)
            
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            self.model.zero_grad()
            
            preds = self.model(imgs, series)
            loss = self.criterion(preds, concentrations)

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
                # References 
                ref_plume = concentrations[:, :-1].detach()
                ref_zeros_map = torch.unsqueeze(concentrations[:, -1], 1).detach()
                
                # Predictions
                pred_plume = preds[:, :-1].detach()
                pred_zeros_map = torch.unsqueeze(preds[:, -1], 1).detach()
                
                # Denomalized
                ref_plume = super()._postprocess(ref_plume, self.concentrations_max, self.concentrations_min)
                pred_plume = super()._postprocess(pred_plume, self.concentrations_max, self.concentrations_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=(ref_plume, ref_zeros_map), 
                                    pred=(pred_plume, pred_zeros_map), 
                                    mode=mode,
                                    epoch=epoch)

        train_loss = super()._metric_average(total_loss, f'{mode}_loss')
        self.losses[mode].append(train_loss)

    def _test(self, data_loader, epoch, mode):
        self.model.eval()
        total_loss = 0.
        for i, (imgs, concentrations, series, release_points) in enumerate(data_loader):
            imgs, concentrations, series = imgs.to(self.device), concentrations.to(self.device), series.to(self.device)
            ref_plume = concentrations[:, :-1]
            zeros_map = torch.unsqueeze(concentrations[:, -1], 1)

            # Keep Object shape
            levelset_cpu = imgs[:,1].to('cpu')
            
            imgs      = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            ref_plume = super()._preprocess(ref_plume, self.concentrations_max, self.concentrations_min)
            concentrations = torch.cat([ref_plume, zeros_map], axis=1)
            
            series = super()._preprocess(series, self.series_max, self.series_min)
            
            preds = self.model(imgs, series)
            loss = self.criterion(preds, concentrations)

            total_loss += loss.item() / len(data_loader.sampler)

            if i == 0:
                # References 
                ref_plume = concentrations[:, :-1].detach()
                ref_zeros_map = torch.unsqueeze(concentrations[:, -1], 1).detach()
                
                # Predictions
                pred_plume = preds[:, :-1].detach()
                pred_zeros_map = torch.unsqueeze(preds[:, -1], 1).detach()
                
                # Denomalized
                ref_plume = super()._postprocess(ref_plume, self.concentrations_max, self.concentrations_min)
                pred_plume = super()._postprocess(pred_plume, self.concentrations_max, self.concentrations_min)

                self.img_saver.save(levelset=levelset_cpu,
                                    release_points=release_points,
                                    ref=(ref_plume, ref_zeros_map), 
                                    pred=(pred_plume, pred_zeros_map), 
                                    mode=mode,
                                    epoch=epoch)

        test_loss = super()._metric_average(total_loss, f'{mode}_loss')
        self.losses[mode].append(test_loss)

    def _infer(self, data_loader, epoch, mode):
        self.model.eval()

        for i, (indices, imgs, concentrations, series, release_points, flows_and_sources) in enumerate(data_loader):
            imgs, concentrations, series = imgs.to(self.device), concentrations.to(self.device), series.to(self.device)

            # Keep Object shape
            levelset_cpu = imgs[:,1].to('cpu')

            ref_plume = concentrations[:, :-1]
            ref_zeros_map = torch.unsqueeze(concentrations[:, -1], 1)

            # Keeping release data (and reshape)
            sdf_release = torch.unsqueeze(imgs[:,1], 1)
           
            # Normalization
            imgs      = super()._preprocess(imgs, self.imgs_max, self.imgs_min)
            ref_plume = super()._preprocess(ref_plume, self.concentrations_max, self.concentrations_min)
            series    = super()._preprocess(series, self.series_max, self.series_min)
            
            # NN prediction
            preds = self.model(imgs, series)
            
            # Predictions
            pred_plume = preds[:, :-1].detach()
            pred_zeros_map = torch.unsqueeze(preds[:, -1], 1).detach()

            ## De-normalize
            ref_plume  = super()._postprocess(ref_plume, self.concentrations_max, self.concentrations_min)
            pred_plume = super()._postprocess(pred_plume, self.concentrations_max, self.concentrations_min)
            series     = super()._postprocess(series, self.series_max, self.series_min)

            self.data_saver.save(levelset=levelset_cpu,
                                 sdf_release=sdf_release,
                                 release_points=release_points,
                                 flows_and_sources=flows_and_sources,
                                 ref=(ref_plume, ref_zeros_map),
                                 pred=(pred_plume, pred_zeros_map),
                                 indices=indices,
                                 series=series,
                                 mode=mode,
                                 epoch=epoch)
