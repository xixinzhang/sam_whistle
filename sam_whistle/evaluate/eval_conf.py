import pickle
from typing import Union
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
import numpy as np
import tyro
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from tqdm import tqdm


from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch, custom_collate_fn
from sam_whistle.config import SAMConfig, DWConfig, FCNSpectConfig, FCNEncoderConfig
from sam_whistle import utils
from sam_whistle.utils.visualize import visualize

@torch.no_grad()
def evaluate_sam_prediction(cfg: SAMConfig, load=False, model: SAM_whistle = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = SAM_whistle(cfg)
        model.to(cfg.device)
        # Load model weights
        if not cfg.freeze_img_encoder:
            model.img_encoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'img_encoder.pth'), weights_only = True))
        if not cfg.freeze_mask_decoder:
            model.decoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'decoder.pth'),  weights_only = True))
        if not cfg.freeze_prompt_encoder:
            model.sam_model.prompt_encoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'prompt_encoder.pth'), weights_only = True))
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistleDataset(cfg, 'test', transform=cfg.spect_cfg.transform)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)

        if cfg.loss_fn == "mse":
            loss_fn = nn.MSELoss()
        elif cfg.loss_fn == "dice":
            loss_fn = DiceLoss()
        elif cfg.loss_fn == "bce_logits":
            loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        assert model is not None and testloader is not None and loss_fn is not None

    model.eval()
    test_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(tqdm(testloader)):
        spect, gt_mask = data['img'], data['mask']
        spect = spect.to(cfg.device)
        gt_mask = gt_mask.to(cfg.device)

        pred_mask= model(spect)
        loss = loss_fn(pred_mask, gt_mask)
        test_losses.append(loss.item())

        if load:
            all_gts.append(gt_mask)
            all_preds.append(pred_mask)

        if visualize_eval:
            # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
            spect = spect.permute(0, 2, 3, 1)
            visualize(spect, gt_mask, pred_mask, output_path, str(i)+ visualize_name)

    test_loss = np.mean(test_losses)
    if load:
        all_gts = torch.cat(all_gts, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        return test_loss, all_gts, all_preds,
    else:
        return test_loss


@torch.no_grad()
def evaluate_deep_prediction(cfg: DWConfig, load=False, model: SAM_whistle = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = Detection_ResNet_BN2(cfg.width)
        model.to(cfg.device)
        model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'model.pth'), map_location=cfg.device, weights_only = True))
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistlePatch(cfg, 'test', transform=cfg.spect_cfg.transform)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        loss_fn  = Charbonnier_loss()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
    
    model.eval()
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(tqdm(testloader)):
        img, mask = data['img'], data['mask']
        img = img.to(cfg.device)
        mask = mask.to(cfg.device)
        pred_mask = model(img)
        batch_loss = loss_fn(pred_mask, mask)
        batch_losses.append(batch_loss.item())

        if load:
            all_gts.append(mask)
            all_preds.append(pred_mask)

    test_loss = np.mean(batch_losses)
    if load:
        all_gts = torch.cat(all_gts, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        return test_loss, all_gts, all_preds,
    else:
        return test_loss
    
@torch.no_grad()
def evaluate_fcn_spect_prediction(cfg: FCNSpectConfig, load=False, model: FCN_Spect = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = FCN_Spect(cfg)
        model.to(cfg.device)
        model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'model.pth'), map_location=cfg.device, weights_only = True))
        loss_fn  = Charbonnier_loss()
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistleDataset(cfg, 'test',spect_nchan=1, transform=cfg.spect_cfg.transform)
        print(testset.meta)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)

        model.init_patch_ls(testset[0]['img'].shape[-2:])
        model.order_pick_patch()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
    
    model.eval()
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(tqdm(testloader)):
        img, mask = data['img'], data['mask']
        img = img.to(cfg.device)
        mask = mask.to(cfg.device)

        pred_mask = model(img)
        batch_loss = loss_fn(pred_mask, mask)
        batch_losses.append(batch_loss.item())
        
        if load:
            all_gts.append(mask)
            all_preds.append(pred_mask)

    batch_num = model.patch_num / cfg.dw_batch
    test_loss = np.sum(batch_losses) / (batch_num*len(testloader))
    if load:
        all_gts = torch.cat(all_gts, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        return test_loss, all_gts, all_preds,
    else:
        return test_loss
    
@torch.no_grad()
def evaluate_fcn_encoder_prediction(cfg: FCNEncoderConfig, load=False, model: FCN_Spect = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = FCN_encoder(cfg)
        model.to(cfg.device)
        model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'model.pth')))
        loss_fn = DiceLoss()

        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistleDataset(cfg, 'test',spect_nchan=1, transform=cfg.spect_cfg.transform)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
        print(f"Test set size: {len(testset)}")
        model.init_patch_ls()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
        
    model.eval() # may not needed as batch size is 1, batch norm is unstable
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(tqdm(testloader)):
        img, mask = data['img'], data['mask']
        img = img.to(cfg.device)
        mask = mask.to(cfg.device)
        pred_mask = model(img)
        test_loss = loss_fn(pred_mask, mask)
        batch_losses.append(test_loss.item())
        
        if load:
            all_gts.append(mask)
            all_preds.append(pred_mask)

    test_loss = np.mean(batch_losses)
    if load:
        all_gts = torch.cat(all_gts, dim=0).cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        return test_loss, all_gts, all_preds,
    else:
        return test_loss


def evaluate_conf_map(cfg: Union[SAMConfig, DWConfig], eval_fn, model_name = 'SAM', visualize_eval=False, visualize_name='', min_thre = 0, max_thre=1, pr_name='pr_curve'):
    test_loss, all_gts, all_preds = eval_fn(cfg, load= True, visualize_eval = visualize_eval, visualize_name=visualize_name)
    eval_res = utils.eval_conf_map(all_gts, all_preds, model_name, min_thre, max_thre)
    print(f"Test Loss: {test_loss:.3f}")
    print(f"Precision: {eval_res.precision:.3f}, Recall: {eval_res.recall:.3f}, F1: {eval_res.f1:.3f}, Threshold: {eval_res.threshold:.3f}")
    
    with open(os.path.join(cfg.log_dir, f'{model_name}_results.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)
    
    utils.plot_pr_curve([eval_res], cfg.log_dir, figname=f'{pr_name}')



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_multiple', action = 'store_true', help='Evaluate a single model')
    parser.add_argument('--model', type=str, default='sam', help='Model to evaluate')
    parser.add_argument('--visual', action = 'store_true', help='Visualize the predictions')
    parser.add_argument('--visual_name', type=str, default='', help='Name of the visualized file')
    parser.add_argument('--min_thre', type=float, default=0.01, help='Minimum threshold for filtering')
    parser.add_argument('--max_thre', type=float, default=0.99, help='Maximum threshold for filtering')
    args, remaining = parser.parse_known_args()
    if not args.eval_multiple:
        if args.model == 'sam':
            cfg = tyro.cli(SAMConfig, args=remaining)
            evaluate_conf_map(cfg, eval_fn=evaluate_sam_prediction, model_name=args.model, visualize_eval=args.visual, visualize_name=args.visual_name, min_thre=args.min_thre, max_thre=args.max_thre)
        elif args.model == 'deep':
            cfg = tyro.cli(DWConfig, args=remaining)
            evaluate_conf_map(cfg, eval_fn= evaluate_deep_prediction, model_name=args.model, visualize_eval=args.visual, visualize_name=args.visual_name, min_thre=args.min_thre, max_thre=args.max_thre)
        elif args.model == 'fcn_spect':
            cfg = tyro.cli(FCNSpectConfig, args=remaining)
            evaluate_conf_map(cfg, eval_fn= evaluate_fcn_spect_prediction, model_name=args.model, visualize_eval=args.visual, visualize_name=args.visual_name, min_thre=args.min_thre, max_thre=args.max_thre)
        elif args.model=='fcn_encoder':
            cfg = tyro.cli(FCNEncoderConfig, args=remaining)
            evaluate_conf_map(cfg, eval_fn= evaluate_fcn_encoder_prediction, model_name=args.model, visualize_eval=args.visual, visualize_name=args.visual_name, min_thre=args.min_thre, max_thre=args.max_thre)
        else:
            raise ValueError(f"Model {args.model} not found")
        
    else:
        eval_results = [
            'logs/11-23-2024_15-19-19-sam/sam_results.pkl',
            'logs/11-23-2024_15-27-33-deep_whistle/deep_results.pkl',
            'logs/11-23-2024_15-39-59-fcn_spect/fcn_spect_results.pkl',
            'logs/11-24-2024_03-02-50-fcn_encoder_imbalance/fcn_encoder_results.pkl'
        ]
        eval_res_li = [pickle.load(open(res_file, 'rb')) for res_file in eval_results]
        utils.plot_pr_curve(eval_res_li, 'logs')