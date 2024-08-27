from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np
import os
from copy import deepcopy
from torch.nn.functional import normalize, threshold

from sam_whistle.config import Args
from sam_whistle import utils
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

class SAM_whistle(nn.Module):

    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.device = args.device
        checkpoint = self.get_checkpoint(self.args.model_type)
        self.sam_model = sam_model_registry[self.args.model_type](checkpoint=checkpoint)
        self.transform= ResizeLongestSide(self.sam_model.image_encoder.img_size)
        if args.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif args.loss_fn == "dice":
            self.loss_fn = DiceLoss()
    
    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.args.sam_ckpt_path, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.args.sam_ckpt_path, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.args.sam_ckpt_path, "sam_vit_h_4b8939.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def forward(self, data):
        spect, _, _, gt_mask, points= data  # BhWC, BHW, (BN2,BN)
        spect = spect.permute(0, 3, 1, 2).to(self.device)
        transformed_spect = self.transform.apply_image_torch(spect)
        input_spect = self.sam_model.preprocess(transformed_spect*255)
        coords, labels = points
        coords = self.transform.apply_coords_torch(coords, spect.shape[-2:])
        coords = coords.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            spect_embedding = self.sam_model.image_encoder(input_spect)
            sparse_embedding, dense_embedding = self.sam_model.prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=None
            )
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=spect_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False
        )

        upscaled_masks, low_mask = self.sam_model.postprocess_masks(
            low_res_masks,
            input_size=transformed_spect.shape[-2:],
            original_size=spect.shape[-2:]
        )
        binary_masks = normalize(threshold(upscaled_masks, 0, 0))
        gt_mask = gt_mask.unsqueeze(1).to(self.device)
        
        # loss
        loss = self.loss_fn(binary_masks, gt_mask)
        
        return loss, binary_masks, low_mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice