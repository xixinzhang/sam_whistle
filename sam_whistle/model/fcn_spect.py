import torch
from torch import nn
import numpy as np


from sam_whistle.config import FCNSpectConfig
from .fcn_patch import Detection_ResNet_BN2
from .loss import Charbonnier_loss

class FCN_Spect(nn.Module):
    """"Patch FCN applied to spectrograms"""
    def __init__(self, cfg: FCNSpectConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_size = self.cfg.spect_cfg.patch_size
        self.fcn = Detection_ResNet_BN2(cfg.width)

    def init_patch_ls(self, shape):
        h, w = shape
        patch_size = self.patch_size
        stride = self.cfg.spect_cfg.patch_stride if self.training else patch_size
        i_starts = torch.arange(0, h - patch_size + 1, stride) 
        j_starts = torch.arange(0, w - patch_size + 1, stride)
        if (h - patch_size) % stride != 0:
            i_starts = torch.cat([i_starts, torch.tensor([h - patch_size])])

        if (w - patch_size) % stride != 0:
            j_starts = torch.cat([j_starts, torch.tensor([w - patch_size])])

        i_grid, j_grid = torch.meshgrid(i_starts, j_starts, indexing='ij')
        self.patch_starts = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=-1)
        self.patch_num = len(self.patch_starts)

    def order_pick_patch(self):
        if self.cfg.random_patch_order and self.training:
            order = np.random.permutation(self.patch_num)
            self.pick_patch= self.patch_starts[order]
        else:
            self.pick_patch = self.patch_starts

    def forward(self, spect):
        patch_size = self.patch_size

        if self.training:
            patch_count = torch.zeros_like(spect)
            pred_image = torch.zeros_like(spect)

            for p in self.pick_patch:
                i, j = p
                patch = spect[..., i:i+patch_size, j:j+patch_size]
                patch_pred = self.fcn(patch)

                pred_image[..., i:i+patch_size, j:j+patch_size] += patch_pred
                patch_count[..., i:i+patch_size, j:j+patch_size] += 1
               
            pred_image /= patch_count
            return pred_image