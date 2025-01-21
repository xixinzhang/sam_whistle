from collections import defaultdict
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate
from PIL import Image
import numpy as np
from pathlib import Path
import tyro
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pickle
import random
from sam_whistle import utils, config
from sam_whistle.config import DWConfig, SAMConfig


def custom_collate_fn(batch):
    specs = [item['img'] for item in batch]
    masks = [item['mask'] for item in batch]
    infos = [item['info'] for item in batch]

    return {
        'img': default_collate(specs),
        'mask': default_collate(masks),
        'info': infos
    }


class RandomBrightnessContrast:
    def __init__(self, 
                 brightness_limit=0.2,
                 contrast_limit=0.2,
                 brightness_prob=0.5,
                 contrast_prob=0.5):
        """
        Args:
            brightness_limit (float): Range for brightness adjustment [-limit, +limit]
            contrast_limit (float): Range for contrast adjustment [-limit, +limit]
            brightness_prob (float): Probability of applying brightness
            contrast_prob (float): Probability of applying contrast
        """
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
    
    def random_brightness(self, spec: np.ndarray):
        """Apply brightness adjustment preserving precision"""
        factor = 1.0 + (2 * np.random.random() - 1) * self.brightness_limit
        return spec * factor
    
    def random_contrast(self, spec: np.ndarray):
        """Apply contrast adjustment preserving precision"""
        factor = 1.0 + (2 * np.random.random() - 1) * self.contrast_limit
        mean = np.mean(spec, axis=(-2, -1), keepdims=True)
        return (spec - mean) * factor + mean
    
    def __call__(self, spec: np.ndarray, debug=False):
        """
        Args:
            spec: Input spectrogram (freq, time) or (batch, freq, time)
            debug: If True, return debug info
        """
        original_range = (np.min(spec), np.max(spec))
        original_mean = np.mean(spec)
        
        # Apply augmentations based on probability
        if np.random.random() < self.brightness_prob:
            spec = self.random_brightness(spec)
        
        if np.random.random() < self.contrast_prob:
            spec = self.random_contrast(spec)
            
        if debug:
            final_range = (np.min(spec), np.max(spec))
            return spec, {
                'original_range': original_range,
                'final_range': final_range,
                'original_mean': original_mean,
                'final_mean': np.mean(spec)
            }
        
        return spec
            
class WhistleDataset(Dataset):
    def __init__(self, cfg: SAMConfig, split='train', spect_nchan=3, transform=False):
        self.cfg = cfg
        self.spect_cfg = cfg.spect_cfg
        self.debug = cfg.debug
        
        self.split = split
        self.root_dir = Path(self.cfg.root_dir)
        self.processed_dir = self.root_dir / 'processed'    
        self.audio_dir = self.root_dir / 'audio'
        self.anno_dir = self.root_dir / 'anno'
        self.meta_file = self.root_dir / self.cfg.meta_file
        self.meta = self._get_dataset_meta()
        self.idx2file = {i: stem for i, stem in enumerate(self.meta)}

        self.interp = self.spect_cfg.interp
        self.spect_nchan = spect_nchan

        if transform and split == 'train':
            self.transform = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
        else:
            self.transform = None
        
        self.raw_data = self._preprocess(cfg.save_pre)
        self.spect_lens = []
        self.data = self._get_data()
        self.spec_prob = [l/ sum(self.spect_lens) for l in self.spect_lens]

        if split == 'test':
            self.test_blocks = []
            for i, stem in enumerate(self.meta):
                n_blocks = self.spect_lens[i] // self.spect_cfg.block_size
                slices = [(i, slice(j*self.spect_cfg.block_size, (j+1)*self.spect_cfg.block_size)) for j in range(n_blocks)]
                self.test_blocks.extend(slices)
        
        if split == 'train':
            self.train_blocks = []
            for i, stem in enumerate(self.meta):
                n_blocks = self.spect_lens[i] // self.spect_cfg.block_size
                slices = [(i, slice(j*self.spect_cfg.block_size, (j+1)*self.spect_cfg.block_size)) for j in range(n_blocks)]
                if self.cfg.spect_cfg.balance_blocks:
                    res = []
                    for i, s in slices:
                        if self.data[i]['mask'][..., s].sum() > 0:
                            res.append((i, s))
                        slices = res
                self.train_blocks.extend(slices)


    def __len__(self):
        if self.split == 'train':
            return len(self.train_blocks)
        elif self.split == 'test':
            return len(self.test_blocks)
        else:
            raise ValueError(f'{self.split} not supported')

    def __getitem__(self, idx):
        """"
        Returns:
            spect: C, H, W
            gt_mask: 1, H, W
        """
        if self.split == 'train':
            if not self.debug:
                spect_idx = np.random.choice(len(self.data), p=self.spec_prob)
                spec_len = self.spect_lens[spect_idx]
                block_start = np.random.randint(0, spec_len - self.spect_cfg.block_size)
                block_end = block_start + self.spect_cfg.block_size
                block_slice = slice(block_start, block_end)
            else:
                spect_idx, block_slice = self.train_blocks[idx]
        else:
            spect_idx, block_slice = self.test_blocks[idx]

        spect = self.data[spect_idx]['img'][..., block_slice]  # [H, W]
        gt_mask = self.data[spect_idx]['mask'][..., block_slice]  # [H, W]

        if spect.ndim == 2:
            spect = np.stack([spect] * self.spect_nchan, axis=0) # [C, H, W]
        if gt_mask.ndim == 2:
            gt_mask = np.expand_dims(gt_mask, 0) # [1, H, W]

        if self.transform:
            spect = self.transform(spect)


        data =  {
            "img": spect, 
            "mask": gt_mask,
            "info": {'spec_idx':spect_idx, 'block_slice':block_slice}
        }

        return data
    
    def _get_dataset_meta(self):
        if self.debug and not self.cfg.all_data:
            meta ={
                'train':['palmyra092007FS192-071012-010614'],
                'test':['Qx-Dc-CC0411-TAT11-CH2-041114-154040-s']
            }
        else:
            meta = json.load(open(self.meta_file, 'r'))

        return meta[self.split]
        
        
    def _preprocess(self, save = False):
        raw_data = {}
        for stem in self.meta:
            spec_power_db, gt_mask = self._wave_to_data(stem, save=save)
            raw_data[stem] = {'img':spec_power_db, 'mask':gt_mask}
        return raw_data

    def _wave_to_data(self, stem, save=False):
        """process one audio file to spectrogram images and get annotations"""
        # spcet
        audio_file = self.audio_dir / f'{stem}.wav'
        bin_file = self.anno_dir/ f'{stem}.bin'
        waveform, sample_rate = utils.load_wave_file(audio_file) # [L]
        spec_power_db= utils.wave_to_spect(waveform, sample_rate, **vars(self.spect_cfg)) # [F, T]
        spec_power_db = np.flip(spec_power_db, [-2])

        # annotations
        shape = spec_power_db.shape[-2:]
        annos = utils.load_annotation(bin_file)

        spec_annos = []
        for anno in tqdm(annos):
            spec_anno = utils.anno_to_spect_point(anno, shape[-2], self.spect_cfg.hop_ms, self.spect_cfg.freq_bin)
            spec_annos.append(spec_anno)
        
        # filter out data w/o annotation
        # target at data like palmyra092007FS192-070924-210000
        x_bound = np.max([contour[:, 0].max() for contour in spec_annos])
        x_bound = int(np.ceil(x_bound))
        spec_power_db = spec_power_db[..., :x_bound]
        gt_mask = self._get_gt_masks(shape, spec_annos, interp=self.interp)
        
        # save spec and annotation
        if save:
            path = self.processed_dir / f'{self.split}/{stem}'
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            np.save(self.processed_dir / f'{self.split}/{stem}/spec.npy', spec_power_db)
            np.save(self.processed_dir / f'{self.split}/{stem}/mask.npy', gt_mask)

        print(f'Loaded spectrogram from {stem}.wav, shape: {spec_power_db.shape}, min: {spec_power_db.min():.2f}, max: {spec_power_db.max():2f}')
        print(f'Loaded mask from {stem}.bin, shape: {gt_mask.shape}')

        return spec_power_db, gt_mask

    def _get_gt_masks(self, shape, contours, interp='linear'):
        """"Get binary mask from each contour"""
        # extract mask from contours
        mask= np.zeros(shape, dtype=np.uint8)
        for i, contour in enumerate(contours):
            new_x, new_y = utils.get_dense_anno_points(contour, origin=self.spect_cfg.origin_annos ,interp=interp)
            new_x = np.maximum(0, np.minimum(new_x, shape[-1]-1)).astype(int)
            new_y = np.maximum(0, np.minimum(new_y, shape[-2]-1)).astype(int)
            mask[new_y, new_x] = 1
        return  mask
    
    def _get_data(self):
        """Load all spectrogram and gt mask, normalize and crop
        
        Returns:
            spect: H, W, C
            gt_mask: H, W, 1
        """
        data = {}
        for i, stem in enumerate(self.meta):
            spect = self.raw_data[stem]['img']
            gt_mask = self.raw_data[stem]['mask']
            assert gt_mask.ndim == 2, 'mask should be 2D as input to cv2.dilate'
            # Get gt mask from annotation
            # Quaility of annotation varies and some annotation are missing
            self.spect_lens.append(spect.shape[-1])
            if not self.spect_cfg.skeleton:
                gt_mask = utils.dilate_mask(gt_mask, kernel_size= self.spect_cfg.kernel_size)
            else:
                gt_mask = utils.skeletonize_mask(gt_mask)

            spect = utils.normalize_spect(spect, self.spect_cfg.normalize, min=self.cfg.spect_cfg.fix_min, max= self.cfg.spect_cfg.fix_max)
            if self.spect_cfg.crop:
                spect = spect[-self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1, :]
                gt_mask = gt_mask[-self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1, :]
            data[i] = {'img':spect, 'mask':gt_mask}
        return data
            

    def _save_spect(self, spec_db_li, span_ids, save_dir, normalized=False, gt=False, ann_dict=None):
        for i in tqdm(span_ids, desc='save spectrogram'):
            sub_spec = spec_db_li[i]
            img_path = save_dir / f'{str(i).zfill(5)}.png'
            # save image
            image_array = (sub_spec.numpy()*255).astype(np.uint8)[0]
            image = Image.fromarray(image_array)
            image.save(img_path)
            # save normalized spectrogram
            if normalized:
                np.savez_compressed(img_path.with_suffix('.npz'), sub_spec.numpy())
                # torch.save(sub_spec, img_path.with_suffix('.pt'))
            
            height, width = image_array.shape          
            # save image with annotation
            if gt:
                assert ann_dict is not None
                fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                ax.imshow(image_array, cmap='gray', vmin=0, vmax=255)
                if i in ann_dict:
                    for contour in ann_dict[i]['contours']:
                        contour = np.array(contour)
                        x = np.minimum(contour[:, 0], width-1)
                        y = np.minimum(contour[:, 1], height-1) 
                        c = (np.random.rand(3)*0.6 + 0.4).tolist()
                        ax.scatter(x, y, color=c, s=1)
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                fig.savefig(save_dir / (img_path.stem + '_gt.png'), dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig)


class WhistlePatch(WhistleDataset):
    def __init__(self, cfg: DWConfig, split='train', spect_nchan=1, transform=False):
        super().__init__(cfg, split, spect_nchan=spect_nchan, transform=transform)
        self.cfg = cfg
        self.spect_cfg = cfg.spect_cfg
        self.patch_size = self.spect_cfg.patch_size
        if split == 'train':
            self.stride = self.spect_cfg.patch_stride
        elif split == 'test':
            self.stride = self.patch_size
        else:
            raise ValueError(f'{split} not supported')

        self.patches_dir = self.processed_dir / 'patches'/ split

        if self.spect_cfg.cached_patches:
            pos_path = self.patches_dir / 'pos_patches.pkl'
            neg_path = self.patches_dir / 'neg_patches.pkl'
            if pos_path.exists() and neg_path.exists():
                self.pos_patches = pickle.load(open(pos_path, 'rb'))
                self.neg_patches = pickle.load(open(neg_path, 'rb'))
            else:
                raise FileNotFoundError(f'Cached patches not found in {self.pos_patches_path} or {self.neg_patches_path}')
        else:
            if self.patches_dir.exists():
                shutil.rmtree(self.patches_dir)
            self.patches_dir.mkdir(parents=True, exist_ok=True)

            self.pos_patches = []
            self.neg_patches = []
            for spec_idx, data in self.data.items():
                spect = data['img']
                gt_mask = data['mask']
                H, W = spect.shape[-2:]

                # Extract patches from the image and mask and keep last complete patch
                i_coords = list(range(0, H - self.patch_size + 1, self.stride))
                j_coords = list(range(0, W - self.patch_size + 1, self.stride))
                if H % self.patch_size != 0:
                        i_coords.append(H - self.patch_size)
                if W % self.patch_size != 0:
                    j_coords.append(W - self.patch_size)
                ii, jj = np.meshgrid(i_coords, j_coords, indexing='ij')
                start_positions = np.column_stack((ii.ravel(), jj.ravel()))

                for i, j in start_positions:
                    label = np.sum(gt_mask[..., i:i+self.patch_size, j:j+self.patch_size]) > 0
                    patch = {
                        'spec_idx': spec_idx,
                        'loc':(i, j),
                        'label': label
                    }
                    if label:
                        self.pos_patches.append(patch)
                    else:
                        self.neg_patches.append(patch)

                
            with open (self.patches_dir / 'pos_patches.pkl', 'wb') as f:
                pickle.dump(self.pos_patches, f)
            with open (self.patches_dir / 'neg_patches.pkl', 'wb') as f:
                pickle.dump(self.neg_patches, f)

        print(f'Positive patches: {len(self.pos_patches)}, Negative patches: {len(self.neg_patches)}')
        
        if split == 'train' and self.spect_cfg.balance_patches:
            balanced_size = min(len(self.pos_patches), len(self.neg_patches))
            if balanced_size <= len(self.neg_patches):
                self.patch_data = self.pos_patches + random.sample(self.neg_patches, balanced_size)
            else:
                self.patch_data = random.sample(self.pos_patches, balanced_size) + self.neg_patches
            print(f'Balanced {self.split } dataset has {len(self.patch_data)} patches')
        else:
            self.patch_data = self.pos_patches + self.neg_patches
            print(f'Imbalanced {self.split } dataset has {len(self.patch_data)} patches')

    def __len__(self):
        return len(self.patch_data)

        
    def __getitem__(self, idx):
        patch_meta = self.patch_data[idx]
        spec_idx = patch_meta['spec_idx']
        i, j = patch_meta['loc']
        patch= self.data[spec_idx]['img'][..., i:i+self.patch_size, j:j+self.patch_size]
        patch_mask = self.data[spec_idx]['mask'][..., i:i+self.patch_size, j:j+self.patch_size]
        label = patch_meta['label']

        if patch.ndim == 2:
            patch = np.stack([patch] * self.spect_nchan, axis=0) # [C, H, W, ]
        if patch_mask.ndim == 2:
            patch_mask = np.expand_dims(patch_mask, 0) # [1, H, W]

        if self.transform:
            patch = self.transform(patch)

        data = {
            'img': patch,
            'mask': patch_mask,
            'info': {'spec_idx': spec_idx, 'loc': (i, j), 'label': label}
        }
        return data

def check_spect_dataset(cfg:SAMConfig, output_dir='outputs/debug'):
    train_set = WhistleDataset(cfg, 'train', spect_nchan=1, transform=True)
    test_set = WhistleDataset(cfg, 'test', spect_nchan=1, transform=True)
    print(f'Train blocks: {len(train_set)}, Test blocks: {len(test_set)}')
    
    for stem in train_set.meta:
        save_dir=f'{output_dir}/train/{stem}'
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
    for i, data in enumerate(tqdm(train_set, desc='check train set')):
        spect, mask, info = data['img'], data['mask'], data['info']
        print(i, spect.min(), spect.max())
        # spect = utils.normalize_spect(spect, 'minmax')
        spect = np.clip(spect * 0.5 + 0.5, 0, 1)

        spec_id = info['spec_idx']
        stem = train_set.meta[spec_id]
        save_dir=f'{output_dir}/train/{stem}'
        utils.visualize_array(spect, cmap='bone', filename=f'train_{spec_id}_{info["block_slice"].start}_1.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray', filename=f'train_{spec_id}_{info["block_slice"].start}_2.mask', save_dir = save_dir)

    for stem in test_set.meta:
        save_dir=f'{output_dir}/test/{stem}'
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
    for i, data in enumerate(tqdm(test_set, desc='check test set')):
        spect, mask, info = data['img'], data['mask'], data['info']
        print(i, spect.min(), spect.max())
        # spect = utils.normalize_spect(spect, 'minmax')
        spect = np.clip(spect * 0.5 + 0.5, 0, 1)

        spec_id = info['spec_idx']
        stem = test_set.meta[spec_id]
        save_dir=f'{output_dir}/test/{stem}'
        utils.visualize_array(spect, cmap='bone',filename=f'test_{spec_id}_{info["block_slice"].start}_1.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray',filename=f'test_{spec_id}_{info["block_slice"].start}_2.mask', save_dir = save_dir)

def check_spect_block(cfg:SAMConfig, spec_idx=0, start=0, split = 'train', ):
    data_set = WhistleDataset(cfg, split)
    check_block = data_set.data[spec_idx]
    spect = check_block['img'][..., start:start+ cfg.spect_cfg.block_size]
    mask = check_block['mask'][..., start:start+ cfg.spect_cfg.block_size]
    # spect = utils.normalize_spect(spect, 'minmax')
    # spect = np.clip(spect * 0.5 + 0.5, 0, 1)
    utils.visualize_array(spect, cmap='bone')
    utils.visualize_array(mask, cmap='gray')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='spect')
    parser.add_argument('--output_dir', type=str, default='outputs/debug')
    args, remaining = parser.parse_known_args()
    if args.data_type == 'spect':
        cfg = tyro.cli(SAMConfig, args=remaining)
        check_spect_dataset(cfg, args.output_dir)
        check_spect_block(cfg, 0, 0, 'test')
    elif args.data_type == 'patch':
        cfg = tyro.cli(DWConfig, args=remaining)
        train_set = WhistlePatch(cfg, 'train')
        img, mask = train_set[0]['img'], train_set[0]['mask']
        label = train_set[0]['info']['label']
        print(img.shape, mask.shape, label, mask.sum())
        utils.visualize_array(img, filename='train_patch_img')
        utils.visualize_array(img, mask=mask.astype(int), random_colors=False, mask_alpha=1, filename='train_patch_mask')
        test_set = WhistlePatch(cfg, 'test')
        print(len(train_set), len(test_set))
        img, mask = test_set[0]['img'], test_set[0]['mask']
        label = test_set[0]['info']['label']
        print(img.shape, mask.shape, label, mask.sum())
        utils.visualize_array(img, filename='test_patch_img')
        utils.visualize_array(img, mask=mask.astype(int), random_colors=False, mask_alpha=1, filename='test_patch_mask')
    else:
        raise ValueError(f'{args.data_type} not supported')
