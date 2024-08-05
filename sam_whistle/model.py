import torch
from torch import nn
import numpy as np
import os
from copy import deepcopy

from sam_whistle.config import Args
from sam_whistle import utils
from segment_anything import sam_model_registry, SamPredictor

class SAM_whistle(nn.Module):

    def __init__(self, args: Args):
        super().__init__()
        self.args = args
    
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

    def setup(self):
        checkpoint = self.get_checkpoint(self.args.model_type)
        self.sam_model = sam_model_registry[self.args.model_type](checkpoint=checkpoint)
        self.annotator = SamPredictor(self.sam_model)
        
        self.sam_model.train()
        if self.args.fintune_decoder_type == "sam":
            self.mask_decoder = deepcopy(self.sam_model.mask_decoder)
        else:
            raise ValueError("Model type error!")


    def forward(self, data):
        spect, bboxes, contours = data
        promts = utils.get_point_prompts(data, self.args.num_pos_points, self.args.num_neg_points, box_pad= self.args.box_pad)
        pseudo_masks = []
        for p in range(len(promts)):
            points, labels = promts[p]
            for i in range(self.args.ann_iters):
                if i == 0:
                    mask_input = None
                else:
                    mask_input = logits[np.argmin(scores)] 
                    mask_input=mask_input[None, :, :]

                masks, scores, logits  = self.annotator.predict(
                    point_coords=points,
                    point_labels=labels,
                    mask_input=mask_input,
                    multimask_output=True,
                )
            pseudo_masks.append(masks[np.argmin(scores)])
        pseudo_mask = utils.combine_masks(pseudo_masks)
        return pseudo_mask
        # get pseudo mask

        # output mask

        # loss

        return