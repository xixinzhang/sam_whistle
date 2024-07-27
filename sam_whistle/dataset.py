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
from .config import Args
from tqdm import tqdm

# from datasets.tools import ResizeAndPad

# def load_datasets(args, img_size=1024):
    
#     transform = ResizeAndPad(img_size)
#     train_dataset = WhistleDataset(args, split='train', transform=transform)
#     test_dataset = WhistleDataset(args, split='test', transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_soft)
#     test_loader = DataLoader(test_dataset, batch_size=args.val_batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

#     return train_loader, test_loader

# def collate_fn(batch):
#     images, bboxes, masks = zip(*batch)
#     images = torch.stack(images)
#     return images, bboxes, masks

# def collate_fn_soft(batch):
#     images_soft, images, bboxes, masks = zip(*batch)
#     images = torch.stack(images)
#     # images_origin = np.stack(images_origin)
#     images_soft = torch.stack(images_soft)
#     return images_soft, images, bboxes, masks

class WhistleDataset(Dataset):
    def __init__(self, args:Args, split='train', transform=None):
        self.args = args
        self.data_dir = Path(args.path)
        self.preprocess_dir = self.data_dir / 'preprocessed'       
        self.meta = self._get_dataset_meta()
        self.transform = transform
        if args.preprocess:
            # self._wave2spect("Qx-Dd-SC03-TAT09-060516-211350", "train", empty=True)
            # self._wave2spect("palmyra092007FS192-070924-205730", "test", empty=True)
            self.preprocess(split)
        else:
            for stem in self.meta['train']:
                if not (self.preprocess_dir / f'train/{stem}').exists():
                    raise FileNotFoundError(f'{stem} not found in train')
        
        self.split = split
        if split == 'train':
            train_files = list((self.preprocess_dir / 'train').glob('*/*.npz'))
            self.idx2file = {i: f for i, f in enumerate(train_files)}
        else:
            test_files = list((self.preprocess_dir / 'test').glob('*/*.npz'))
            self.idx2file = {i: f for i, f in enumerate(test_files)}
        
        with open(self.preprocess_dir/split /'annotation.json', 'r') as f:
            self.all_ann_dict = json.load(f)
        self.spect_ids = list(self.idx2file.keys())

    def __len__(self):
        return len(self.spect_ids)

    def __getitem__(self, idx):
        spect_path = self.idx2file[idx]
        file_id = spect_path.parent.name
        spect_split_id = int(spect_path.stem)
        spect = np.load(spect_path)['arr_0']
        spect = np.stack([spect, spect, spect], axis=-1)
        ann_dict = self.all_ann_dict[file_id]
        anns = ann_dict[str(spect_split_id)] # {'contours', 'bboxes', 'masks'}
        bboxes = []
        masks = []
        contours = []
        height, width, _ = spect.shape
        for contour, bbox, ma in zip(anns['contours'], anns['bboxes'], anns['masks']):
            contours.append(contour)
            bboxes.append(bbox)
            # mask
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y in ma:
                if x>=0 and x<width and y>=0 and y<height:
                    mask[y, x] = 1
            masks.append(mask)

        # spect, masks, bboxes = self.transform(spect, masks, np.array(bboxes))
        bboxes = np.stack(bboxes, axis=0) # [num_obj, 4]
        masks = np.stack(masks, axis=0) # [num_obj, height, width]
        contours = [np.array(contour) for contour in contours]
        return spect, bboxes, masks, contours

        # if self.split == 'train':
        #     return spect, spect, torch.tensor(bboxes), torch.tensor(masks).float()
        # else:
        #     return spect, torch.tensor(bboxes), torch.tensor(masks).float()
    


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
            mask_vertics = ann.astype(np.int32)
            ann_dict[span_id]['contours'].append(contour.tolist())
            ann_dict[span_id]['bboxes'].append(bbox)
            ann_dict[span_id]['masks'].append(mask_vertics.tolist())

        for ann in tqdm(annos, desc='process annotation'):
            ann = np.array(ann)
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
        self._save_spect(spec_db_li, ann_span_ids, split_dir, normalized=True, )
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)
    dataset = WhistleDataset(args, 'train')
    print(len(dataset))
    data = dataset[0]
    print(data[0].shape, data[1].shape, data[2].shape)

