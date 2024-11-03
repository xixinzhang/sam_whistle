import shutil
import torch
from torch import nn
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import tyro
import struct 
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import networkx as nx
import sknw
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d, splev, splrep
from numpy.polynomial import Polynomial
import cv2
import lmdb
import pickle
import random
import os 

from segment_anything.utils.transforms import ResizeLongestSide
from .utils import simplify_path, simplify_graph, graph_to_mask, sknw_graph_2_key_mask, graph_to_keymask
from sam_whistle import utils, config, visualization


class WhistleDataset(Dataset):
    def __init__(self, args:config.Args, split='train', img_size=1024, spect_nchan=3):
        self.args = args
        self.data_dir = Path(args.path)
        self.preprocess_dir = self.data_dir / 'preprocessed'       
        self.meta = self._get_dataset_meta()
        self.split = split
        self.spect_nchan = spect_nchan
        self.transform = ResizeLongestSide(img_size)
        if args.preprocess:
            # self._wave2spect("palmyra092007FS192-071012-010614", "train", empty=True)
            # self._wave2spect("palmyra092007FS192-070924-205730", "test", empty=True)
            self.preprocess(split)
        else:
            for stem in self.meta['train']:
                if not (self.preprocess_dir / f'train/{stem}').exists():
                    raise FileNotFoundError(f'{stem} not found in train')
        if args.crop:
            self.freq_resolution = 1000 // args.frame_ms
            self.time_resolution = args.hop_ms
            self.crop_bottom = args.min_freq // self.freq_resolution
            self.crop_top = args.max_freq // self.freq_resolution

        if split == 'train':
            train_files = list((self.preprocess_dir / 'train').glob('*/*.npz'))
            self.idx2file = {i: f for i, f in enumerate(train_files)}
        else:
            test_files = list((self.preprocess_dir / 'test').glob('*/*.npz'))
            self.idx2file = {i: f for i, f in enumerate(test_files)}
        
        with open(self.preprocess_dir/split /'annotation.json', 'r') as f:
            self.all_ann_dict = json.load(f)

        self.spect_ids = list(self.idx2file.keys())
        self.interpolation = args.interpolation

    def __len__(self):
        return len(self.spect_ids)

    def __getitem__(self, idx):
        spect_path = self.idx2file[idx]
        # print(spect_path)
        file_id = spect_path.parent.name
        spect_split_id = int(spect_path.stem)
        spect = np.load(spect_path)['arr_0']
        if self.spect_nchan == 3:
            spect = np.stack([spect, spect, spect], axis=-1) # [H, W, C]
        elif self.spect_nchan == 1:
            spect = spect[:, :, np.newaxis]
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
        masks = self._get_gt_masks(spect.shape[:2], contours, interp=self.interpolation)
        combined_gt_mask = utils.combine_masks(masks)

        if not self.args.skeleton:
            combined_gt_mask = self._expand_mask(combined_gt_mask)
        else:
            # combined_gt_mask = self._expand_mask(combined_gt_mask)
            combined_gt_mask = self._skeletonize_mask(combined_gt_mask)
            pass
        
        if self.args.crop:
            spect = spect[-self.crop_top: -self.crop_bottom+1]
            combined_gt_mask = combined_gt_mask[-self.crop_top: -self.crop_bottom+1]

        if self.args.skeleton and self.args.graph:
            multi_edge = False
            sknw_graph = sknw.build_sknw(combined_gt_mask, multi=multi_edge)
            # keypoint_mask = sknw_graph_2_key_mask(sknw_graph, combined_gt_mask.shape, radius=3)

            simplified_graph = simplify_graph(sknw_graph, tolerance=0.5, multi_edge=multi_edge)
            combined_gt_mask = graph_to_mask(simplified_graph, combined_gt_mask.shape, width=2)
            keypoint_mask = graph_to_keymask(simplified_graph, combined_gt_mask.shape, radius=2)

        spect= spect.transpose(-1, 0, 1) # [C, H, W]
        combined_gt_mask = combined_gt_mask[np.newaxis, :, :]
        
        data =  {
                "spect": spect, 
                "contour_mask": combined_gt_mask, 
        }

        if self.args.use_prompt:
            if self.args.sample_points is None:
                points_prompt = None
            elif self.args.sample_points == "random":
                points_prompt = utils.sample_points_random(masks, self.args.num_pos_points, self.args.num_neg_points)
            elif self.args.sample_points == "box":
                points_prompt = utils.sample_points_box(masks, self.args.num_pos_points, self.args.num_neg_points, self.args.box_pad)

            data["points_prompt"]= points_prompt

        if self.args.skeleton and self.args.graph:
            data["keypoint_mask"] = keypoint_mask

        return data

    def _get_gt_masks(self, shape, contours, interp=None):
        """"Get binary mask from each contour"""
        masks = []
        for i, contour in enumerate(contours):
            ma= np.zeros(shape)
            
            x, y = contour[:, 0], contour[:, 1]
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > y_range:
                new_x = np.arange(int(x_min), int(x_max)+1)
                x_order = np.argsort(x)
                x = x[x_order]
                y = y[x_order]

                if interp == "linear":
                    new_y = np.interp(new_x, x, y)
                elif interp == "polynomial":
                    poly_interp = Polynomial.fit(x, y, 3)
                    new_y = poly_interp(new_x)
                elif interp == "spline":
                    splline_interp = splrep(x, y, k=3)
                    new_y = splev(new_x, splline_interp)
                else:
                    raise ValueError("Interpolation method not supported")
            else:
                new_y = np.arange(int(y_min), int(y_max)+1)
                y_order = np.argsort(y)
                x = x[y_order]
                y = y[y_order]
                if interp == "linear":
                    new_x = np.interp(new_y, y, x)
                elif interp == "polynomial":
                    poly_interp = Polynomial.fit(y, x, 3)
                    new_x = poly_interp(new_y)
                elif interp == "spline":
                    splline_interp = splrep(y, x, k=3)
                    new_x = splev(new_y, splline_interp)
                else:
                    raise ValueError("Interpolation method not supported")
            
            new_x = np.maximum(0, np.minimum(new_x, shape[1]-1)).astype(int)
            new_y = np.maximum(0, np.minimum(new_y, shape[0]-1)).astype(int)
  
            for y, x in zip(new_y, new_x):
                ma[y, x] = 1
            masks.append(ma)
        return masks

    def _expand_mask(self, mask, kernel_size=3):
        """dilate the mask"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

    def _skeletonize_mask(self, mask):
        """skeletonize the binary mask"""
        mask = mask.astype(np.uint8)
        skeleton = skeletonize(mask)
        return skeleton.astype(np.uint8)    

    def _skeleton2graph(self, skeleton):
        """convert skeleton to graph"""
        multi_edge = True
        graph = sknw.build_sknw(skeleton, multi=multi_edge)

    def _get_dataset_meta(self, train_anno_dir = 'train_puli', test_anno_dir = 'test_puli'):

        train_anno_dir = self.data_dir / train_anno_dir
        test_anno_dir = self.data_dir / test_anno_dir
        train_anno_files = list(train_anno_dir.glob('*.bin'))
        test_anno_files = list(test_anno_dir.glob('*.bin'))
        meta = {
            'train':  [f.stem for f in train_anno_files if f.stat().st_size > 0],
            'test': [f.stem for f in test_anno_files if f.stat().st_size > 0]
        }
        return meta

    def preprocess(self, split = 'train'):
        all_ann_dict = {}
        for stem in self.meta[split]:
            ann_dict = self._wave2spect(stem, data_split=split, empty=True)
            all_ann_dict.update(ann_dict)

        split_dir = self.preprocess_dir / split
        with open(split_dir/'annotation.json', 'w') as f:
            json.dump(all_ann_dict, f) # {{file: {split_id: [contour]}}
        

    def _load_annotation(self, stem, anno_dir):
        """Read the .bin file and obtain annotations of each contour"""
        bin_file = self.data_dir / anno_dir / f'{stem}.bin'
        data_format = 'dd'  # 2 double-precision [time(s), frequency(Hz)]
        with open(bin_file, 'rb') as f:
            bytes = f.read()
            total_bytes_num = len(bytes)
            if total_bytes_num==0:
                print(f'{bin_file}: EOF')
                return
            cur = 0
            annos = []
            while True:
                # iterater over each contour
                num_point = struct.unpack('>i', bytes[cur: cur+4])[0]
                format_str = f'>{num_point * data_format}'
                point_bytes_num = struct.calcsize(format_str)
                cur += 4
                data = struct.unpack(f'{format_str}', bytes[cur:cur+point_bytes_num])
                num_dim = 2
                data = np.array(data).reshape(-1, num_dim)
                annos.append(data)
                cur += point_bytes_num
                if cur >= total_bytes_num:
                    break
            print(f'{bin_file} has {len(annos)} contours')
            return annos


    def _load_audio(self, stem, audio_dir='audio'):
        """Load audio data and get the signal info"""
        audio_dir = self.data_dir / audio_dir
        audio_file = audio_dir / f'{stem}.wav'
        info = torchaudio.info(audio_file)
        waveform, _ = torchaudio.load(audio_file)
        return waveform, info                 

    def _wave2spect(self, stem, data_split='train', empty=False):
        """process one audio file to spectrogram images and get annotations"""
        # spcet
        waveform, info = self._load_audio(stem)
        if self.args.n_fft is None:
            assert self.args.frame_ms and self.args.hop_ms
            frame_ms = self.args.frame_ms
            hop_ms = self.args.hop_ms
            n_fft = int(frame_ms /1000 * info.sample_rate)
            hop_length = int(hop_ms /1000 * info.sample_rate)
        else:
            n_fft = self.args.n_fft
            hop_length = self.args.hop_length
            hop_ms  = hop_length / info.sample_rate * 1000
            frame_ms  = n_fft / info.sample_rate * 1000
            
        spec = F.spectrogram(waveform[0], pad = 0, window = torch.hamming_window(n_fft), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=2, normalized=False)
        spec_db = F.amplitude_to_DB(spec, multiplier=10, amin = 1e-10, db_multiplier=0)
        spec_db = torch.flip(spec_db, [0])
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())  # normalize to [0, 1]
        height, width = spec_db.shape

        # annotations
        ann_dict = {}
        anno_dir = data_split + '_puli'
        annos = self._load_annotation(stem, anno_dir)

        def get_anns(contour, span_id):
            if span_id not in ann_dict:
                ann_dict[span_id] = {'contours':[], 'bboxes':[], 'masks':[]}
            xcoord, ycoord = contour[:, 0], contour[:, 1]
            bbox = [xcoord.min(), ycoord.min(), xcoord.max(), ycoord.max()]
            # mask_vertics = contour.astype(np.int32)
            ann_dict[span_id]['contours'].append(contour.tolist())
            ann_dict[span_id]['bboxes'].append(bbox)
            # ann_dict[span_id]['masks'].append(mask_vertics.tolist())

        for ann in tqdm(annos, desc='process annotation'):
            ann = np.array(ann)
            # ann = np.sort(ann, axis=0)
            sorted_indices = np.argsort(ann[:, 0])
            ann = ann[sorted_indices]
            ann[:, 0] = ann[:, 0] * 1000 # ms
            min_time, max_time = ann[:, 0].min(), ann[:, 0].max()
            start_span_id = int(min_time // self.args.split_ms)
            end_span_id = int(max_time // self.args.split_ms)
            span_start_ms = start_span_id * self.args.split_ms
            rela_time = ann[:, 0] - span_start_ms
            
            if start_span_id == end_span_id:
                # non-cut
                x = rela_time / hop_ms
                y = ann[:, 1] / (info.sample_rate/ n_fft)  # top-down frequency
                y_bottom = height - y # bottom-up frequency
                contour = np.stack([x, y_bottom], axis=-1)
                get_anns(contour, start_span_id)
            else:
                # cut
                # in
                in_mask = rela_time<self.args.split_ms
                x = rela_time[in_mask] / hop_ms 
                y = ann[:, 1][in_mask] / (info.sample_rate/ n_fft)  # top-down frequency
                y_bottom = height - y # bottom-up frequency
                contour = np.stack([x, y_bottom], axis=-1)
                get_anns(contour, start_span_id)
                # out
                out_mask = rela_time>=self.args.split_ms
                x = (rela_time[out_mask] - self.args.split_ms) / hop_ms 
                y = ann[:, 1][out_mask] / (info.sample_rate/ n_fft)
                y_bottom = height - y
                contour = np.stack([x, y_bottom], axis=-1)
                get_anns(contour, end_span_id)


     
        # save to sepectrogram images
        split_pix = int(self.args.split_ms // hop_ms)
        num_split = spec_db.shape[1] // split_pix # throw away the last part
        print(f'{stem} has {num_split} splits')
        spec_db_li = torch.split(spec_db, split_pix, dim=1)
        all_span_ids = list(range(len(spec_db_li)))
        if spec_db_li[-1].shape[1] < split_pix:
            ann_dict.pop(len(spec_db_li)-1, None)
        ann_span_ids = list(ann_dict.keys())

        spec_dir = self.preprocess_dir / f'spectrogram/{stem}'
        split_dir = self.preprocess_dir / f'{data_split}/{stem}'
        if empty and spec_dir.exists():
            shutil.rmtree(spec_dir)
        if empty and split_dir.exists():
            shutil.rmtree(split_dir)
        spec_dir.mkdir(parents= True,exist_ok=True)
        split_dir.mkdir(parents= True,exist_ok=True)
        self._save_spect(spec_db_li, ann_span_ids, split_dir,  gt=True, normalized=True, ann_dict=ann_dict)
        self._save_spect(spec_db_li, all_span_ids, spec_dir, gt=True, ann_dict=ann_dict, )
        
        # save annotation
        return {stem: ann_dict}


    def _save_spect(self, spec_db_li, span_ids, save_dir, normalized=False, gt=False, ann_dict=None):
        for i in tqdm(span_ids, desc='save spectrogram'):
            sub_spec = spec_db_li[i]
            img_path = save_dir / f'{str(i).zfill(5)}.png'
            # save image
            image_array = (sub_spec.numpy()*255).astype(np.uint8)
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

        self.lmdb_path = self.preprocess_dir / f'patch_{split}.lmdb'
        self.balanced_lmdb_path = self.preprocess_dir / f'balanced_patch_{split}.lmdb'
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
                masks = self._get_gt_masks(spect.shape[:2], contours, interp=self.interpolation)
                gt_mask = utils.combine_masks(masks)
                expand_gt_mask = self._expand_mask(gt_mask)

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
            if not self.args.balanced_cached:
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


if __name__ == "__main__":
    args = tyro.cli(config.Args)
    print(args)
    # WhistleDataset
    # train_set = WhistleDataset(args, 'train')
    # test_set = WhistleDataset(args, 'test')
    # print("testing data done")
    # print(len(train_set), len(test_set)) 
    # data = train_set[0]
    # print(data[0].shape, data[1].shape, )

    # WhistlePatch
    train_set = WhistlePatch(args, 'train')
    # test_set = WhistlePatch(args, 'test')
    img, mask = train_set[0]
    print(img.shape, mask.shape)
    # visualization.visualize_array(img, mask=mask, random_colors=False, filename='patch')
    # visualization.visualize_array(img, random_colors=False, filename='patch2')