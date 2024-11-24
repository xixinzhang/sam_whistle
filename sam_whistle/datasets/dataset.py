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

from segment_anything.utils.transforms import ResizeLongestSide
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
            
class WhistleDataset(Dataset):
    def __init__(self, cfg: SAMConfig, split='train', spect_nchan=3):
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

        if self.cfg.preprocess:
            self._preprocess()
        else:
            # check if all files are processed
            for stem in self.meta:
                if not (self.processed_dir / f'{split}/{stem}/spec.pt').exists():
                    raise FileNotFoundError(f'{stem}.wav not found in split: {split}')
                if not (self.processed_dir / f'{split}/{stem}/mask.npy').exists():
                    raise FileNotFoundError(f'{stem}.bin not found in split: {split}')
        
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
                self.train_blocks.extend(slices)

    def __len__(self):
        if self.split == 'train':
            if not self.debug and self.spect_cfg.block_multi > 1:
                return len(self.train_blocks) * self.spect_cfg.block_multi
            else:
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
            if not self.debug and self.spect_cfg.block_multi > 1:
                spect_idx = np.random.choice(len(self.data), p=self.spec_prob)
                spec_len = self.spect_lens[spect_idx]
                block_start = np.random.randint(0, spec_len - self.spect_cfg.block_size)
                block_end = block_start + self.spect_cfg.block_size
                block_slice = slice(block_start, block_end)
            else:
                spect_idx, block_slice = self.train_blocks[idx]
        else:
            spect_idx, block_slice = self.test_blocks[idx]

        spect = self.data[spect_idx]['img'][..., block_slice]
        gt_mask = self.data[spect_idx]['mask'][..., block_slice]

        if self.spect_nchan == 3:
            spect = torch.cat([spect, spect, spect], axis=0) # [C, H, W]

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
        
        
    def _preprocess(self):
        for stem in self.meta:
            self._wave_to_data(stem, empty=True)

    def _wave_to_data(self, stem, empty=False):
        """process one audio file to spectrogram images and get annotations"""
        # spcet
        audio_file = self.audio_dir / f'{stem}.wav'
        bin_file = self.anno_dir/ f'{stem}.bin'
        waveform, sample_rate = utils.load_wave_file(audio_file) # [C, L] Channel first
        spec_power_db= utils.wave_to_spect(waveform, sample_rate, **vars(self.spect_cfg)) # [C, F, T]
        spec_power_db = torch.flip(spec_power_db, [-2])

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
        if empty:
            path = self.processed_dir / f'{self.split}/{stem}'
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        print(f'Loaded spectrogram from {stem}.wav, shape: {spec_power_db.shape}, min: {spec_power_db.min():.2f}, max: {spec_power_db.max():2f}')
        print(f'Loaded mask from {stem}.bin, shape: {gt_mask.shape}')

        torch.save(spec_power_db, self.processed_dir / f'{self.split}/{stem}/spec.pt')
        np.save(self.processed_dir / f'{self.split}/{stem}/mask.npy', gt_mask)


    def _get_gt_masks(self, shape, contours, interp='linear'):
        """"Get binary mask from each contour"""
        # extract mask from contours
        mask= np.zeros(shape)
        for i, contour in enumerate(contours):
            x, y = contour[:, 0], contour[:, 1]
            x_min, x_max = x.min(), x.max()
            x_order = np.argsort(x)
            x = x[x_order]
            y = y[x_order]

            length = len(x)
            if self.spect_cfg.origin_annos:
                new_x = x
                new_y = y
            else:
                new_x = np.linspace(x_min, x_max, length*10, endpoint=True)
                new_y = utils.interpolate_anno_point(new_x, x, y, interp)
            
            new_x = np.maximum(0, np.minimum(new_x, shape[-1]-1)).astype(int)
            new_y = np.maximum(0, np.minimum(new_y, shape[-2]-1)).astype(int)

            for y, x in zip(new_y, new_x):
                mask[y, x] = 1
        return  mask
    
    def _get_data(self):
        """Load all spectrogram and gt mask, normalize and crop
        
        Returns:
            spect: C, H, W
            gt_mask: H, W
        """
        data = {}
        for i, stem in enumerate(self.meta):
            spect = torch.load(self.processed_dir / f'{self.split}/{stem}/spec.pt', weights_only=False)
            gt_mask = np.load(self.processed_dir / f'{self.split}/{stem}/mask.npy')
            assert gt_mask.ndim == 2, 'mask should be 2D as input to cv2.dilate'
            # Get gt mask from annotation
            # Quaility of annotation varies and some annotation are missing
            self.spect_lens.append(spect.shape[-1])
            if not self.spect_cfg.skeleton:
                gt_mask = utils.dilate_mask(gt_mask)
            else:
                gt_mask = utils.skeletonize_mask(gt_mask)

            spect = utils.normalize_spect(spect)
            if self.spect_cfg.crop:
                spect = spect[:, -self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1]
                gt_mask = gt_mask[-self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1]
                
            data[i] = {'img':spect, 'mask':gt_mask[None]}
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
    def __init__(self, cfg: DWConfig, split='train', spect_nchan=1):
        super().__init__(cfg, split, spect_nchan=spect_nchan)
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
        
        if self.split == 'train' and  self.spect_cfg.balance_patches:
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

        data = {
            'img': patch,
            'mask': patch_mask,
            'info': {'spec_idx': spec_idx, 'loc': (i, j), 'label': label}
        }
        return data

def check_spect_dataset(cfg:SAMConfig):
    train_set = WhistleDataset(cfg, 'train', spect_nchan=1)
    test_set = WhistleDataset(cfg, 'test', spect_nchan=1)
    print(f'Train blocks: {len(train_set)}, Test blocks: {len(test_set)}')
    
    for stem in train_set.meta:
        shutil.rmtree(f'outputs/debug/train/{stem}', ignore_errors=True)
    for i, data in enumerate(tqdm(train_set, desc='check train set')):
        spec, mask, info = data['img'], data['mask'], data['info']
        spec_id = info['spec_idx']
        stem = train_set.meta[spec_id]
        save_dir=f'outputs/debug/train/{stem}'
        utils.visualize_array(spec, cmap='bone', filename=f'train_{spec_id}_{info["block_slice"].start}_1.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray', filename=f'train_{spec_id}_{info["block_slice"].start}_2.mask', save_dir = save_dir)

    for stem in test_set.meta:
        shutil.rmtree(f'outputs/debug/test/{stem}', ignore_errors=True)
    for i, data in enumerate(tqdm(test_set, desc='check test set')):
        spec, mask, info = data['img'], data['mask'], data['info']
        spec_id = info['spec_idx']
        stem = test_set.meta[spec_id]
        save_dir=f'outputs/debug/test/{stem}'
        utils.visualize_array(spec, cmap='bone',filename=f'test_{spec_id}_{info["block_slice"].start}_1.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray',filename=f'test_{spec_id}_{info["block_slice"].start}_2.mask', save_dir = save_dir)

def check_spect_block(cfg:SAMConfig, spec_idx=0, start=0, split = 'train', ):
    data_set = WhistleDataset(cfg, split)
    check_block = data_set.data[spec_idx]
    spect = check_block['img'][..., start:start+ cfg.spect_cfg.block_size]
    mask = check_block['mask'][..., start:start+ cfg.spect_cfg.block_size]
    utils.visualize_array(spect, cmap='bone')
    utils.visualize_array(mask, cmap='gray')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='spect')
    args, remaining = parser.parse_known_args()
    if args.data_type == 'spect':
        cfg = tyro.cli(SAMConfig, args=remaining)
        check_spect_dataset(cfg)
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
