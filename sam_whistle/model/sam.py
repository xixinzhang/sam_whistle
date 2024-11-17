from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np
import os
from copy import deepcopy
from torch.nn.functional import normalize, threshold, interpolate


from sam_whistle import config
from sam_whistle.utils.visualize import visualize_array

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.common import LayerNorm2d


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SAM_whistle(nn.Module):
    def __init__(self, cfg: config.SAMConfig):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        checkpoint = self.get_checkpoint(self.cfg.model_type)
        self.sam_model = sam_model_registry[self.cfg.model_type](checkpoint=checkpoint)

        self.transform= ResizeLongestSide(self.sam_model.image_encoder.img_size)
        self.img_encoder = self.sam_model.image_encoder
        self.prompt_encoder = self.sam_model.prompt_encoder

        # naive decoder
        if self.cfg.sam_decoder:
            self.decoder = self.sam_model.mask_decoder
        else:
            encoder_output_dim = 256 # sam
            activation = nn.GELU
            self.decoder =nn.Sequential(
                    nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    activation(),
                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                    activation(),
                    nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
                )

        if self.cfg.freeze_img_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
        if self.cfg.freeze_mask_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        if self.cfg.freeze_prompt_encoder: 
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
 
    
    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam_vit_h_4b8939.pth")
        else:
            raise ValueError("Model type error!")
        return checkpoint

    def forward(self, spect):
        if self.cfg.use_prompt:
            spect, gt_mask, points= spect  # BhWC, BHW, (BN2,BN)
            coords, labels = points
            coords = self.transform.apply_coords_torch(coords, spect.shape[-2:])
            coords = coords.to(self.device)
            labels = labels.to(self.device)
            sparse_embedding, dense_embedding = self.sam_model.prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=None
            )   
   
        spect = spect.to(self.device)
        transformed_spect = self.transform.apply_image_torch(spect)
        input_spect = self.sam_model.preprocess(transformed_spect*255)  # [0, 1]
        
        spect_embedding = self.sam_model.image_encoder(input_spect)  # (B, 256, 64, 64)

        if self.cfg.sam_decoder:
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=spect_embedding,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding,
                multimask_output=False
            )

            mask_logits, low_mask = self.sam_model.postprocess_masks(
                low_res_masks,
                input_size=transformed_spect.shape[-2:],
                original_size=spect.shape[-2:]
            )
            # binary_masks = normalize(threshold(upscaled_masks, 0, 0))
            upscaled_masks = torch.sigmoid(mask_logits)
        else:
            mask_logits = self.decoder(spect_embedding)

            mask_logits = mask_logits[..., :transformed_spect.shape[-2], : transformed_spect.shape[-1]]
            low_mask = mask_logits
            mask_logits = interpolate(mask_logits, spect.shape[-2:], mode="bilinear", align_corners=False)
            upscaled_masks = torch.sigmoid(mask_logits)

        pred_mask = upscaled_masks
        return pred_mask


