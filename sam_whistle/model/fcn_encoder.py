import torch
from torch import nn
from torch.nn.functional import normalize, threshold, interpolate, pad


from sam_whistle.model import Detection_ResNet_BN2
from sam_whistle.config import Args
from sam_whistle.visualization import visualize_array
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.common import LayerNorm2d

class FCN_encoder(nn.Module):
    """FCN as an encoder, applied on spectrograms"""
    def __init__(self, args: Args, img_size = 1024):
        super().__init__()
        self.args = args
        self.patch_size = args.patch_size

        self.img_encoder = Detection_ResNet_BN2(args.pu_width)
        # self._load_fcn_ckpt()
        self.downsample = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 256, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm2d(256),

        )
        encoder_output_dim = 256 # sam (B, 256, 64, 64)
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
        self.encoder_img_size = img_size
        self.transform = ResizeLongestSide(img_size)

    def init_patch_ls(self):
        h, w = self.encoder_img_size, self.encoder_img_size
        patch_size = self.patch_size
        # stride = self.args.patch_stride
        stride = patch_size
        i_starts = torch.arange(0, h - patch_size + 1, stride) 
        j_starts = torch.arange(0, w - patch_size + 1, stride)
        if (h - patch_size) % stride != 0:
            i_starts = torch.cat([i_starts, torch.tensor([h - patch_size])])

        if (w - patch_size) % stride != 0:
            j_starts = torch.cat([j_starts, torch.tensor([w - patch_size])])

        i_grid, j_grid = torch.meshgrid(i_starts, j_starts, indexing='ij')
        self.patch_starts = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=-1)
        self.patch_num = len(self.patch_starts)

    def _load_fcn_ckpt(self,):
        print(f"Loading FCN model at {self.args.pu_model_path}")
        self.img_encoder.load_state_dict(torch.load(self.args.pu_model_path))


    def preprocess(self, spect:torch.Tensor):
        # Pad
        h, w = spect.shape[-2:]
        padh = self.encoder_img_size - h
        padw = self.encoder_img_size - w
        spect = pad(spect, (0, padw, 0, padh))
        return spect

    def forward(self, spect:torch.Tensor):
        transformed_spect = self.transform.apply_image_torch(spect)
        input_spect = self.preprocess(transformed_spect)

        patch_size = self.patch_size
        patch_count = torch.zeros_like(input_spect)
        pred_image = torch.zeros_like(input_spect)

        for p in self.patch_starts:
            i, j = p
            patch = input_spect[..., i:i+patch_size, j:j+patch_size]
            patch_pred = self.img_encoder(patch)
            pred_image[..., i:i+patch_size, j:j+patch_size] += patch_pred
            patch_count[..., i:i+patch_size, j:j+patch_size] += 1
        
        spect_feat = pred_image / patch_count # (B, 1, 1024, 1024)
        upsampled_feat = self.downsample(spect_feat)

        mask_logits = self.decoder(upsampled_feat) # input should be (B, 256, 64, 64)

        mask_logits = mask_logits[..., :transformed_spect.shape[-2], : transformed_spect.shape[-1]]
        mask_logits = interpolate(mask_logits, spect.shape[-2:], mode="bilinear", align_corners=False)
        pred_mask = torch.sigmoid(mask_logits)
        
        return pred_mask

