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
import lmdb
import pickle
import random
import os 

from segment_anything.utils.transforms import ResizeLongestSide
from sam_whistle import utils, config


def custom_collate_fn(batch):
    specs = [item['spect'] for item in batch]
    masks = [item['gt_mask'] for item in batch]
    infos = [item['info'] for item in batch]

    return {
        'spect': default_collate(specs),
        'gt_mask': default_collate(masks),
        'info': infos
    }
            
class WhistleDataset(Dataset):
    def __init__(self, cfg: config.SAMConfig, split='train', img_size=1024, spec_nchan=3):
        self.cfg = cfg
        self.spect_cfg = cfg.spect_config
        self.debug = cfg.debug
        
        self.split = split
        self.root_dir = Path(self.cfg.root_dir)
        self.processed_dir = self.root_dir / 'processed'    
        self.audio_dir = self.root_dir / 'audio'
        self.anno_dir = self.root_dir / 'annotation'
        self.meta_file = self.root_dir / self.cfg.meta_file
        self.meta = self._get_dataset_meta()
        self.idx2file = {i: stem for i, stem in enumerate(self.meta)}

        self.spec_nchan = spec_nchan
        self.transform = ResizeLongestSide(img_size)

        if self.cfg.preprocess:
            self._preprocess()
        else:
            # check if all files are processed
            for stem in self.meta:
                if not (self.processed_dir / f'{split}/{stem}/spec.pt').exists():
                    raise FileNotFoundError(f'{stem}.wav not found in split: {split}')
                if not (self.processed_dir / f'{split}/{stem}/anno.pkl').exists():
                    raise FileNotFoundError(f'{stem}.bin not found in split: {split}')
        
        self.spec_lens = []
        self.interp = self.spect_cfg.interp
        self.data = self._annotation_to_mask()
        self.spec_prob = [l/ sum(self.spec_lens) for l in self.spec_lens]

        if split == 'test':
            self.test_blocks = []
            for i, stem in enumerate(self.meta):
                n_blocks = self.spec_lens[i] // self.spect_cfg.block_size
                slices = [(i, slice(j*self.spect_cfg.block_size, (j+1)*self.spect_cfg.block_size)) for j in range(n_blocks)]
                self.test_blocks.extend(slices)
        
        if split == 'train':
            self.train_blocks = []
            for i, stem in enumerate(self.meta):
                n_blocks = self.spec_lens[i] // self.spect_cfg.block_size
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
        if self.split == 'train':
            if not self.debug and self.spect_cfg.block_multi > 1:
                spect_idx = np.random.choice(len(self.data), p=self.spec_prob)
                spec_len = self.spec_lens[spect_idx]
                block_start = np.random.randint(0, spec_len - self.spect_cfg.block_size)
                block_end = block_start + self.spect_cfg.block_size
                block_slice = slice(block_start, block_end)
            else:
                spect_idx, block_slice = self.train_blocks[idx]
        else:
            spect_idx, block_slice = self.test_blocks[idx]

        spect = self.data[spect_idx]['spect'][:, :, block_slice]
        gt_mask = self.data[spect_idx]['mask'][:, block_slice][None]

        if self.spec_nchan == 3:
            spect = torch.cat([spect, spect, spect], axis=0) # [C, H, W]

        # crop high and low freq
        if self.spect_cfg.crop:
            spect = spect[:, -self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1]
            gt_mask = gt_mask[:, -self.spect_cfg.crop_top: -self.spect_cfg.crop_bottom+1]
    
        data =  {
            "spect": spect, 
            "gt_mask": gt_mask,
            "info": {'spec_idx':spect_idx, 'block_slice':block_slice}
        }

        return data
    
    def _get_dataset_meta(self):
        if self.debug and not self.cfg.all_data:
            meta ={
                'train':['palmyra092007FS192-071012-010614'],
                'test':['palmyra092007FS192-070924-205730']
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
        spec_power_db= utils.wave_to_spect(waveform, sample_rate, **vars(self.spect_cfg))
        print(f'\nLoaded spectrogram from {stem}, shape: {spec_power_db.shape},max: {spec_power_db.max():2f}, min: {spec_power_db.min():.2f}')
        spec_power_db = utils.flip_and_normalize_spect(spec_power_db)

        # annotations
        height = spec_power_db.shape[-2]
        annos = utils.load_annotation(bin_file)

        spec_annos = []
        for anno in tqdm(annos):
            spec_anno = utils.anno_to_spec_point(anno, height, self.spect_cfg.hop_ms, self.spect_cfg.freq_bin)
            spec_annos.append(spec_anno)

        # save spec and annotation
        if empty:
            path = self.processed_dir / f'{self.split}/{stem}'
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        torch.save(spec_power_db, self.processed_dir / f'{self.split}/{stem}/spec.pt')
        with open(self.processed_dir / f'{self.split}/{stem}/anno.pkl', 'wb') as f:
            pickle.dump(spec_annos, f)
        

    def _load_processed_data(self, stem=None):
        """load preprocessed data"""
        spect = torch.load(self.processed_dir / f'{self.split}/{stem}/spec.pt')
        with open(self.processed_dir / f'{self.split}/{stem}/anno.pkl', 'rb') as f:
            annos = pickle.load(f)
        return  spect, annos

    def _get_gt_masks(self, spect, contours, interp='linear'):
        """"Get binary mask from each contour"""
        # filter out data w/o annotation
        # target at data like palmyra092007FS192-070924-210000
        x_bound = np.max([contour[:, 0].max() for contour in contours])
        x_bound = int(np.ceil(x_bound))
        spect = spect[..., :x_bound]
        
        # extract mask from contours
        shape = spect.shape[-2:] 
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
            
            new_x = np.maximum(0, np.minimum(new_x, shape[1]-1)).astype(int)
            new_y = np.maximum(0, np.minimum(new_y, shape[0]-1)).astype(int)

            for y, x in zip(new_y, new_x):
                mask[y, x] = 1
        return spect, mask
    
    def _annotation_to_mask(self):
        data = {}
        for i, stem in enumerate(tqdm(self.meta, desc=f"Masking {self.split}'s annotations")):
            spec, annos = self._load_processed_data(stem)
            # Get gt mask from annotation
            # Quaility of annotation varies and some annotation are missing
            spec, gt_mask = self._get_gt_masks(spec, annos, interp=self.interp)
            self.spec_lens.append(spec.shape[-1])
            if not self.spect_cfg.skeleton:
                gt_mask = utils.dilate_mask(gt_mask)
                pass
            else:
                gt_mask = utils.skeletonize_mask(gt_mask)
            data[i] = {'spect':spec, 'mask':gt_mask}
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
    def __init__(self, args:config.Args, split='train', img_size=None):
        super().__init__(args, split, img_size)
        self.patch_size = args.patch_size
        if split == 'train':
            self.stride = args.patch_stride
        else:
            self.stride = self.patch_size

        self.lmdb_path = self.processed_dir / f'patch_{split}.lmdb'
        self.balanced_lmdb_path = self.processed_dir / f'balanced_patch_{split}.lmdb'
        self.lmdb_path = str(self.lmdb_path)
        self.balanced_lmdb_path = str(self.balanced_lmdb_path)
        if args.patch_cached:
            if not os.path.exists(self.lmdb_path):
                raise FileNotFoundError(f'Patch cache file: {self.lmdb_path} not found')
            if os.path.exists(self.balanced_lmdb_path) and not args.balanced_cached:
                shutil.rmtree(self.balanced_lmdb_path)
        else:
            if os.path.exists(self.lmdb_path):
                shutil.rmtree(self.lmdb_path)
            spect_list = []
            mask_list = []
            for idx in tqdm(range(len(self.spect_ids))):
                spect_path = self.idx2file[idx]
                # print(spect_path)
                file_id = spect_path.parent.name
                spect_split_id = int(spect_path.stem)
                spect = np.load(spect_path)['arr_0']
                spect = np.stack([spect, spect, spect], axis=-1)
                ann_dict = self.all_ann_dict[file_id]
                anns = ann_dict[str(spect_split_id)] # {'contours', 'bboxes', 'masks'}
                bboxes = []
                # masks = []
                contours = []
                # height, width, _ = spect.shape
                for contour, bbox in zip(anns['contours'], anns['bboxes']):
                    contours.append(contour)
                    bboxes.append(bbox)
                # spect, masks, bboxes = self.transform(spect, masks, np.array(bboxes))
                bboxes = np.stack(bboxes, axis=0) # [num_obj, 4]
                contours = [np.array(contour) for contour in contours]
                masks = self._get_gt_masks(spect.shape[:2], contours, interp=self.interp)
                gt_mask = utils.combine_masks(masks)
                expand_gt_mask = self._dilate_mask(gt_mask)

                if args.crop:
                    spect = spect[-self.crop_top: -self.crop_bottom+1]
                    expand_gt_mask = expand_gt_mask[-self.crop_top: -self.crop_bottom+1]

                # spect= spect.transpose(-1, 0, 1) # [C, H, W]
                # expand_gt_mask = expand_gt_mask[np.newaxis, :, :]

                spect_list.append(spect)
                mask_list.append(expand_gt_mask)
            
            self._store_patches_in_lmdb(spect_list, mask_list, self.patch_size, self.stride)
        


        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        print(f'Original lmdb for {split} has {self.env.stat()['entries']} entries')  
    
        if self.split == 'train':
            if not self.spect_cfg.balanced_cached:
                print('Balancing positive and negative patches...')
                self._get_balanced_lmdb()
            self.balanced_env = lmdb.open(self.balanced_lmdb_path, readonly=True, lock=False)
            print(f'Balenced lmdb  for {split} has {self.balanced_env.stat()['entries']} entries')

    def _extract_patches(self, image, mask, patch_size=50, stride=25):
        """
        Extracts patches from an image and corresponding mask.

        Args:
            image (np.ndarray): The image (H, W, C) for multi-channel.
            mask (np.ndarray): The corresponding mask (H, W).
            patch_size (int): The size of each patch.
            stride (int): The stride for overlapping patches.

        Returns:
            image_patches (list): List of image patches.
            mask_patches (list): List of corresponding mask patches.
            metadata (list): Metadata with (img_idx, i, j) where (i, j) is the top-left corner.
        """
        h, w = image.shape[:2]
        image_patches = []
        mask_patches = []
        metadata = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                image_patch = image[i:i+patch_size, j:j+patch_size, :]
                mask_patch = mask[i:i+patch_size, j:j+patch_size]
                image_patches.append(image_patch)
                mask_patches.append(mask_patch)
                metadata.append((i, j))  # Record the top-left corner of the patch
        return image_patches, mask_patches, metadata

    def _store_patches_in_lmdb(self, image_list, mask_list, patch_size=50, stride=25):
        """
        Partition images and masks into patches, then store them in LMDB with metadata.
        mark the patch is positive or negative (foreground or background).

        Args:
            image_list (list of np.ndarray): List of images.
            mask_list (list of np.ndarray): List of masks corresponding to images.
            lmdb_path (str): Path to store the LMDB database.
            patch_size (int): Size of each patch.
            stride (int): Stride between patches.
        """
        # Open LMDB database
        env = lmdb.open(self.lmdb_path, map_size=int(1e12)) 

        patch_id = 0  # Unique patch identifier
        with env.begin(write=True) as txn:
            for img_idx, (image, mask) in enumerate(tqdm(zip(image_list, mask_list), total=len(image_list))):
                # Extract patches from the image and mask
                image_patches, mask_patches, metadata = self._extract_patches(image, mask, patch_size, stride)
                
                # Store patches with their positive/negative label
                for img_patch, mask_patch, meta in zip(image_patches, mask_patches, metadata):
                    # Determine if the patch is positive or negative
                    if np.any(mask_patch > 0):  # Positive patch: contains non-zero pixels (foreground)
                        label = 'positive'
                    else:  # Negative patch: all pixels are zero (background)
                        label = 'negative'

                    # Store the patch data along with the label
                    data = {
                        'image_patch': img_patch,
                        'mask_patch': mask_patch,
                        'image_index': img_idx,
                        'location': meta,
                        'label': label  # 'positive' or 'negative'
                    }
                    txn.put(f'patch_{patch_id}'.encode(), pickle.dumps(data))
                    patch_id += 1  # Increment patch ID
    
    def _get_balanced_lmdb(self, lmdb_batch_size=1000,):
        """
        Create a new LMDB with balanced positive and negative patches.

        Args:
            lmdb_path (str): Path to LMDB file.

        Returns:
            positive_patches (list): List of all positive patches from LMDB.
            negative_patches (list): List of all negative patches from LMDB.
        """
        positive_patches = []
        negative_patches = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, patch_data in tqdm(cursor, total=self.env.stat()['entries']):
                data = pickle.loads(patch_data)
                # Separate positive and negative patches based on label
                if data['label'] == 'positive':
                    positive_patches.append(data)
                else:
                    negative_patches.append(data)

        min_size = min(len(positive_patches), len(negative_patches))
        print(f"Positive patches: {len(positive_patches)}, Negative patches: {len(negative_patches)}")

        # Randomly downsample the larger set
        balanced_positive = random.sample(positive_patches, min_size) if len(positive_patches) > min_size else positive_patches
        balanced_negative = random.sample(negative_patches, min_size) if len(negative_patches) > min_size else negative_patches

        balanced_env = lmdb.open(self.balanced_lmdb_path, map_size=int(1e12))

        patch_id = 0  # New patch ID for the balanced dataset
        for i in tqdm(range(0, min_size, lmdb_batch_size)):
            # Batch the writes in chunks
            with balanced_env.begin(write=True) as txn:
                for batch_idx in range(i, min(i + lmdb_batch_size, min_size)):
                    # Add positive patches
                    txn.put(f'patch_{patch_id}'.encode(), pickle.dumps(balanced_positive[batch_idx]))
                    patch_id += 1
                    # Add negative patches
                    txn.put(f'patch_{patch_id}'.encode(), pickle.dumps(balanced_negative[batch_idx]))
                    patch_id += 1
                

    def _retrieve_patch_from_lmdb(self, patch_id):
        """
        Retrieve a patch and its metadata from LMDB.

        Args:
            lmdb_path (str): Path to LMDB file.
            patch_id (int): The unique patch ID.

        Returns:
            dict: A dictionary containing the image patch, mask patch, metadata, and label.
        """
        if self.split == 'train':
            with self.balanced_env.begin() as txn:
                patch_data = txn.get(f'patch_{patch_id}'.encode())
                data = pickle.loads(patch_data)
                return data
        elif self.split == 'test':
            with self.env.begin() as txn:
                patch_data = txn.get(f'patch_{patch_id}'.encode())
                data = pickle.loads(patch_data)
                return data
            
    def __len__(self):
        if self.split == 'train':
            return self.balanced_env.stat()['entries']
        elif self.split == 'test':
            return self.env.stat()['entries']
        
    def __getitem__(self, idx):
        data = self._retrieve_patch_from_lmdb(idx)
        img_patch = data['image_patch'][..., [0]]
        mask_patch = data['mask_patch']

        img_patch= img_patch.transpose(-1, 0, 1) # [C, H, W]
        mask_patch = mask_patch[np.newaxis, :, :]

        if self.split == 'train':
            return img_patch, mask_patch
        elif self.split == 'test': 
            meta = {"image_index": data['image_index'], "location": data['location']}
            return img_patch, mask_patch, meta


def check_spec_dataset(cfg:config.SAMConfig):
    train_set = WhistleDataset(cfg, 'train', spec_nchan=1)
    test_set = WhistleDataset(cfg, 'test', spec_nchan=1)
    print(f'Train blocks: {len(train_set)}, Test blocks: {len(test_set)}')
    
    for stem in train_set.meta:
        shutil.rmtree(f'outputs/debug/train/{stem}', ignore_errors=True)
    for i, data in enumerate(tqdm(train_set, desc='check train set')):
        spec, mask, info = data['spect'], data['gt_mask'], data['info']
        spec_id = info['spec_idx']
        stem = train_set.meta[spec_id]
        save_dir=f'outputs/debug/train/{stem}'
        utils.visualize_array(spec, cmap='bone', filename=f'train_{info["block_slice"].start}_0.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray', filename=f'train_{info["block_slice"].start}_1.mask', save_dir = save_dir)

    for stem in test_set.meta:
        shutil.rmtree(f'outputs/debug/test/{stem}', ignore_errors=True)
    for i, data in enumerate(tqdm(test_set, desc='check test set')):
        spec, mask, info = data['spect'], data['gt_mask'], data['info']
        spec_id = info['spec_idx']
        stem = test_set.meta[spec_id]
        save_dir=f'outputs/debug/test/{stem}'
        utils.visualize_array(spec, cmap='bone',filename=f'test_{info["block_slice"].start}_0.spec', save_dir = save_dir)
        utils.visualize_array(mask, cmap='gray',filename=f'test_{info["block_slice"].start}_1.gt', save_dir = save_dir)

def check_spec_block(cfg:config.SAMConfig, spec_idx, start, split = 'train', ):
    data_set = WhistleDataset(cfg, split)
    check_block = data_set.data[spec_idx]
    spect = check_block['spect'][..., start:start+ cfg.spect_config.block_size]
    mask = check_block['mask'][..., start:start+ cfg.spect_config.block_size]
    utils.visualize_array(spect, cmap='magma')
    utils.visualize_array(mask, cmap='gray')

if __name__ == "__main__":
    cfg = tyro.cli(config.SAMConfig)
    check_spec_dataset(cfg)
    # check_spec_block(0, 21000, 'test')

    # WhistlePatch
    # train_set = WhistlePatch(args, 'train')
    # test_set = WhistlePatch(args, 'test')
    # img, mask = train_set[0]
    # print(img.shape, mask.shape)
    # visualization.visualize_array(img, mask=mask, random_colors=False, filename='patch')
    # visualization.visualize_array(img, random_colors=False, filename='patch2')