import os.path
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


class WhistleCOCO(VisionDataset):
    """Adapted from torchvision.datasets.CocoDetection
    https://pytorch.org/vision/main/generated/torchvision.datasets.CocoDetection.html

    Args:
    root (str or ``pathlib.Path``): Root directory where images are downloaded to.
    annFile (string): Path to json annotation file.
    transform (callable, optional): A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    transforms (callable, optional): A function/transform that takes input sample and its target as entry
        and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.audio_to_image = self._get_audio_to_image()

    def _get_audio_to_image(self) -> dict:
        """Map audio filenames to image ids in dataset."""
        audio_to_image = defaultdict(list)
        for img_id, img_info in self.coco.imgs.items():
            audio_filename = img_info['audio_filename']
            audio_to_image[audio_filename].append(img_id)
        return audio_to_image

    def _load_image(self, id: int) -> np.ndarray:
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        masks = [anno["segmentation"] for anno in target] # [[poly1, poly2, ...], [poly1, poly2, ...], ...]

        height, width = image.shape[:2]
        bitmap_mask = np.stack([poly2mask(poly, height, width) for poly in masks], axis=0)
        bitmap_mask = (np.sum(bitmap_mask, axis=0, keepdims=True) > 0).astype(np.uint8)

        spec = image.transpose(2, 0, 1).astype(np.float32) / 255 # (C, H, W)
        gt_mask = bitmap_mask # (1, H, W)
        info = dict(
            image_id=id,
            category_id = 1,
            audio_filename = self.coco.imgs[id]['audio_filename'],
            start_frame = self.coco.imgs[id]['start_frame'],
        )

        data =  {
            "img": spec,  # (C, H, W) [0, 1]
            "mask": gt_mask, # (1, H, W) (binary mask)
            'info': info
        }

        return data

    def __len__(self) -> int:
        return len(self.ids)
    

def poly2mask(poly, height, width):
    """Convert polygon to binary mask."""
    if isinstance(poly, list):
        rles = maskUtils.frPyObjects(poly, height, width)
        rle = maskUtils.merge(rles)
    
    mask = maskUtils.decode(rle)
    return mask

class WhistlePatchCOCO(WhistleCOCO):
    """Adapted from WhistleCOCO to load patches of the original images."""

    def __init__(self, train=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches = []
        self.masks = []
        self.labels = []
        self.train = train
        self._prepare_patches_coordinates()
        self._prepare_balanced_patches()

    def _prepare_patches_coordinates(self,):
        img_id = self.ids[0]
        image = self._load_image(img_id)
        height, width = image.shape[:2]
        self.patch_size = 50
        self.stride = 25
        self.patch_topleft = []

        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                self.patch_topleft.append((y, x))

    def _prepare_balanced_patches(self,):
        """Process all the images and split them into patches"""
        for img_id in self.ids:
            image = self._load_image(img_id)
            target = self._load_target(img_id)
            masks = [anno["segmentation"] for anno in target]
            
            height, width = image.shape[:2]
            bitmap_mask = np.stack([poly2mask(poly, height, width) for poly in masks], axis=0)
            bitmap_mask = (np.sum(bitmap_mask, axis=0, keepdims=True) > 0).astype(np.uint8)
            # Split the image and mask into patches
            self._collect_patches(image, bitmap_mask[0])  # (H, W, 3), (H, W)

        # Balance the patches
        if self.train:
            pos_num = sum(self.labels)
            neg_num = len(self.labels) - pos_num
            if neg_num > pos_num:
                neg_indices = np.where(np.array(self.labels) == 0)[0]
                np.random.shuffle(neg_indices)
                neg_indices = neg_indices[:pos_num]
                pos_indices = np.where(np.array(self.labels) == 1)[0]
                indices = np.concatenate([pos_indices, neg_indices])
                patches = np.asarray(self.patches)[indices] # (N, W, C)
                masks = np.asarray(self.masks)[indices] # (H, W)
            else:
                raise ValueError("The number of positive patches is greater than the number of negative patches.")
            
            print(f'Balanced patches included positive {len(pos_indices)} and negative {len(neg_indices)} patches.')
        else:
            patches = np.asarray(self.patches)
            masks = np.asarray(self.masks)
            print(f'All patches included positive {len(np.where(np.array(self.labels) == 1)[0])} and negative {len(np.where(np.array(self.labels) == 0)[0])} patches.')
        
        del self.labels
        self.data_list = [dict(img=patch, mask=mask) for patch, mask in zip(patches, masks)]


    def _collect_patches(self, image, mask):
        """Split the image and mask into patches."""
        for y, x in tqdm(self.patch_topleft):
            patch = image[y:y + self.patch_size, x:x + self.patch_size]
            mask_patch = mask[y:y + self.patch_size, x:x + self.patch_size]
            self.patches.append(patch)
            self.masks.append(mask_patch)
            self.labels.append(np.sum(mask_patch) > 0)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        data = self.data_list[index]
        spec = data['img'].transpose(2, 0, 1).astype(np.float32) / 255 # (C, H, W)
        spec = spec[:1, ...] # (1, H, W)
        gt_mask = data['mask'].astype(np.uint8)[None, ...] # (1, H, W)

        return {
            "img": spec,  # (C, H, W) [0, 1]
            "mask": gt_mask, # (1, H, W) (binary mask)
        }


if __name__ == "__main__":
    # train_set = WhistleCOCO(root='data/dclde/spec_coco/train/data', annFile='data/dclde/spec_coco/train/labels.json')
    # print(len(train_set))
    # iter_train = iter(train_set)
    # data = next(iter_train)
    # print(data)

    test_set_patch = WhistlePatchCOCO(root='data/dclde/cross_coco/train/data', annFile='data/dclde/cross_coco/train/labels.json')
    print(len(test_set_patch))
    iter_train_patch = iter(test_set_patch)
    data_patch = next(iter_train_patch)
    for k, v in data_patch.items():
        print(k, v.shape)


