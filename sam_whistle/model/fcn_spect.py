import torch
from torch import nn
import numpy as np


from sam_whistle.config import Args
from .fcn_patch import Detection_ResNet_BN2
from .loss import Charbonnier_loss

class FCN_Spect(nn.Module):
    """"FCN applied to spectrograms"""
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        self.patch_size = args.patch_size
        self.fcn = Detection_ResNet_BN2(args.pu_width)

    def init_patch_ls(self, shape):
        h, w = shape
        patch_size = self.patch_size
        stride = self.args.patch_stride if self.training else patch_size
        i_starts = torch.arange(0, h - patch_size + 1, stride) 
        j_starts = torch.arange(0, w - patch_size + 1, stride)
        i_grid, j_grid = torch.meshgrid(i_starts, j_starts, indexing='ij')
        self.patch_starts = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=-1)
        self.patch_num = len(self.patch_starts)

    def order_pick_patch(self):
        if self.args.random_patch_order and self.training:
            order = np.random.permutation(self.patch_num)
            self.pick_patch= self.patch_starts[order]
        else:
            self.pick_patch = self.patch_starts

    def forward(self, spect, mask):
        patch_size = self.patch_size

        if self.training:
            if self.args.slide_mean:
                patch_count = torch.zeros_like(spect)
                pred_image = torch.zeros_like(spect)
            else:
                pred_patches = []
                gt_patches = []

            for p in self.pick_patch:
                i, j = p
                patch = spect[..., i:i+patch_size, j:j+patch_size]
                patch_pred = self.fcn(patch)

                if self.args.slide_mean:
                    pred_image[..., i:i+patch_size, j:j+patch_size] += patch_pred
                    patch_count[..., i:i+patch_size, j:j+patch_size] += 1
                else:
                    patch_y = mask[..., i:i+patch_size, j:j+patch_size]
                    pred_patches.append(patch_pred)
                    gt_patches.append(patch_y)

            if self.args.slide_mean:
                pred_image /= patch_count
                return pred_image, mask
            else:
                pred_image = torch.cat(pred_patches, dim=0)  # batched pathes
                gt_image = torch.cat(gt_patches, dim=0)
                return pred_image, gt_image

        elif not self.training:
            pred_patches = []
            gt_patches = []
            for p in self.pick_patch:
                i, j = p
                patch = spect[..., i:i+patch_size, j:j+patch_size]
                patch_pred = self.fcn(patch)
                patch_y = mask[..., i:i+patch_size, j:j+patch_size]
                pred_patches.append(patch_pred)
                gt_patches.append(patch_y)
        
            pred_image = torch.cat(pred_patches, dim=0)
            gt_image = torch.cat(gt_patches, dim=0)
            return pred_image, gt_image
                    