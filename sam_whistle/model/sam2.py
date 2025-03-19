import os
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.functional import interpolate, normalize, threshold
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_utils import LayerNorm2d
from sam_whistle import config
from sam_whistle.utils.visualize import visualize_array


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SAM2_whistle(nn.Module):
    def __init__(self, cfg: config.SAM2Config):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        sam2_checkpoint, model_cfg = self.get_ckpt_and_cfg(self.cfg.model_type)
        self.model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        resolution = self.model.image_size

        self.transform= Compose([
            # ToTensor(),
            Resize((resolution, resolution)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self.img_encoder = self.model.image_encoder
        self.prompt_encoder = self.model.sam_prompt_encoder

        # naive decoder
        if self.cfg.sam_decoder:
            self.decoder = self.model.sam_mask_decoder
        else:
            encoder_output_dim = 256 # sam
            activation = nn.GELU
            self.decoder =nn.Sequential(
                    nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    activation(),
                    nn.Conv2d(128, 64, kernel_size=1, stride=1),
                    LayerNorm2d(64),
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
 
    
    def get_ckpt_and_cfg(self, model_type):
        if model_type == "hiera_b":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam2.1_hiera_base_plus.pt")
            model_cfg = os.path.join(self.cfg.sam_cfg, "sam2.1_hiera_b+.yaml")
        elif model_type == "hiera_l":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam2.1_hiera_large.pt")
            model_cfg = os.path.join(self.cfg.sam_cfg, "sam2.1_hiera_l.yaml")
        elif model_type == "hiera_s":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam2.1_hiera_small.pt")
            model_cfg = os.path.join(self.cfg.sam_cfg, "sam2.1_hiera_s.yaml")
        elif model_type == "hiera_t":
            checkpoint = os.path.join(self.cfg.ckpt_dir, "sam2.1_hiera_tiny.pt")
            model_cfg = os.path.join(self.cfg.sam_cfg, "sam2.1_hiera_t.yaml")
        else:
            raise ValueError("Model type error!")
        return checkpoint, model_cfg

    def forward(self, spect):
        """
        Args:
            spect: (B, C, H, W)
        """
        b, _, h, w = spect.shape  # (H, W)
        input_spect = self.transform(spect)
        # transformed_spect = self.transform.apply_image_torch(spect)
        # input_spect = self.sam2_model.preprocess(transformed_spect*255)  # [0, 1]
        
        # spect_embedding = self.sam2_model.image_encoder(input_spect)  # (B, 256, 64, 64)

        backbone_out = self.model.forward_image(input_spect)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(b, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings, high_res_features =  feats[-1], feats[:-1]

        dc0, ln0, act0, dc1, ln1, act1, dc2, act2, dc3 = self.decoder
        feat_s0, feat_s1 = high_res_features
        image_embeddings = act0(ln0(dc0(image_embeddings)))
        upscaled_embedding = act1(ln1(dc1(image_embeddings) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) 


        low_mask = dc3(upscaled_embedding)

        mask_logits = interpolate(low_mask, (h, w), mode="bilinear", align_corners=False)
        upscaled_masks = torch.sigmoid(mask_logits)

        pred_mask = upscaled_masks
        return pred_mask


