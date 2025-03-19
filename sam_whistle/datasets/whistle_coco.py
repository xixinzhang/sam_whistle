import os.path
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from torchvision.datasets.vision import VisionDataset


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


if __name__ == "__main__":
    train_set = WhistleCOCO(root='data/dclde/spec_coco/train/data', annFile='data/dclde/spec_coco/train/labels.json')
    print(len(train_set))
    iter_train = iter(train_set)
    data = next(iter_train)
    print(data)