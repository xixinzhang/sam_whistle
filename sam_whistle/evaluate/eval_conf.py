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


from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch, custom_collate_fn
from sam_whistle.config import SAMConfig, DWConfig, FCNSpectConfig, FCNEncoderConfig
from sam_whistle import utils
from sam_whistle.utils.visualize import visualize

@dataclass
class EvalResults:
    model_name: str
    precision: float  # Average precision
    recall: float    # At optimal threshold
    f1: float
    threshold: float  # Optimal threshold
    # For plotting PR curve
    precisions: list[float]  
    recalls: list[float]
    thresholds: list[float]


@torch.no_grad()
def evaluate_sam_prediction(cfg: SAMConfig, load=False, model: SAM_whistle = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = SAM_whistle(cfg)
        model.to(cfg.device)
        # Load model weights
        if not cfg.freeze_img_encoder:
            model.img_encoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'img_encoder.pth')))
        if not cfg.freeze_mask_decoder:
            model.decoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'decoder.pth')))
        if not cfg.freeze_prompt_encoder:
            model.sam_model.prompt_encoder.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'prompt_encoder.pth')))
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistleDataset(cfg, 'test')
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
    for i, data in enumerate(testloader):
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
        model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'model_dw.pth')))
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistlePatch(cfg, 'test',)
        testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        loss_fn  = Charbonnier_loss()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
    
    model.eval()
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(testloader):
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
        model = FCN_Spect(cfg.width)
        model.to(cfg.device)
        model.load_state_dict(torch.load(os.path.join(cfg.log_dir, 'model_dw.pth')))
        loss_fn  = Charbonnier_loss()
        
        output_path = os.path.join(cfg.log_dir, 'predictions')
        if not os.path.exists(output_path) and visualize_eval:
            os.makedirs(output_path)
        
        testset = WhistleDataset(cfg, 'test',spect_nchan=1)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)

        model.init_patch_ls(testset[0]['img'].shape[-2:])
        model.order_pick_patch()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
    
    model.eval()
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(testloader):
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
        
        testset = WhistleDataset(cfg, 'test',spect_nchan=1)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=custom_collate_fn)
        print(f"Test set size: {len(testset)}")
        model.init_patch_ls()
    else:
        assert model is not None and testloader is not None and loss_fn is not None
        
    model.eval() # may not needed as batch size is 1, batch norm is unstable
    batch_losses = []
    all_gts = []
    all_preds = []
    for i, data in enumerate(testloader):
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

def evaluate_conf_map(cfg: Union[SAMConfig, DWConfig], eval_fn, model_name = 'SAM', visualize_eval=False, visualize_name='', min_thre = 0, max_thre=1):
    test_loss, all_gts, all_preds = eval_fn(cfg, load= True, visualize_eval = visualize_eval, visualize_name=visualize_name)
    eval_res = utils.evaluate_model(all_gts, all_preds, model_name, min_thre, max_thre)
    print(f"Test Loss: {test_loss:.3f}")
    print(f"Precision: {eval_res.precision:.3f}, Recall: {eval_res.recall:.3f}, F1: {eval_res.f1:.3f}, Threshold: {eval_res.threshold:.3f}")
    
    with open(os.path.join(cfg.log_dir, f'{model_name}_results.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)
    
    plt.figure(figsize=(10, 8))
    plt.plot(eval_res.recalls, eval_res.precisions, 
            label=f'{eval_res.model_name} (F1={eval_res.f1:.3f})')
    plt.scatter(eval_res.recall, eval_res.precision, zorder=5)

    precision_grid, recall_grid = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    f1_grid = 2 * (precision_grid * recall_grid) / (precision_grid + recall_grid)
    f1_contour = plt.contour(recall_grid, precision_grid, f1_grid, levels=np.linspace(0.1, 0.9, 9), colors='green', linestyles='dashed')
    plt.clabel(f1_contour, fmt='%.2f', inline=True, fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc = 'lower left')
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(os.path.join(cfg.log_dir, 'precision_recall_curve.png'))
    plt.close()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_single', action = 'store_true', help='Evaluate a single model')
    parser.add_argument('--model', type=str, default='sam_whistle', help='Model to evaluate')
    parser.add_argument('--visual', action = 'store_true', help='Visualize the predictions')
    parser.add_argument('--visual_name', type=str, default='', help='Name of the visualized file')
    parser.add_argument('--min_thre', type=float, default=0.05, help='Minimum threshold for filtering')
    parser.add_argument('--max_thre', type=float, default=0.95, help='Maximum threshold for filtering')
    cfg, remaining = parser.parse_known_args()
    if cfg.eval_single:

        if cfg.model == 'sam':
            cfg = tyro.cli(SAMConfig, args=remaining)
            evaluate_conf_map(cfg,eval_fn=evaluate_sam_prediction, model_name='sam_whistle', visualize_eval=cfg.visual, visualize_name=cfg.visual_name, min_thre=cfg.min_thre, max_thre=cfg.max_thre)
        elif cfg.model == 'deep':
            cfg = tyro.cli(DWConfig, args=remaining)
            evaluate_conf_map(cfg, eval_fn= evaluate_deep_prediction,model_name='deep_whistle', visualize_eval=cfg.visual, visualize_name=cfg.visual_name, min_thre=cfg.min_thre, max_thre=cfg.max_thre)
        # elif args.model=='fcn_spect':
        #     precision, recall, thresholds= evaluate_fcn_spect(args)
        #     precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'fcn_spect', save_path=args.log_dir)
        #     model_names = ['FCN_Spect']
        # elif args.model=='fcn_encoder':
        #     precision, recall, thresholds= evaluate_fcn_encoder(args)
        #     precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'fcn_encoder', save_path=args.log_dir)
        #     model_names = ['FCN_Encoder']
        # else:
        #     raise ValueError(f"Model {args.model} not found")
        
        # if not isinstance(precision_range, list):
        #     precision_range = [precision_range]
        #     recall_range = [recall_range]
        #     thresholds_range = [thresholds_range]
        #     f1_range = [f1_range]
        # plot_precision_recall(precision_range, recall_range, thresholds_range, f1_range, model_names, save_path=args.log_dir)
    # else:
    #     evaluations =[
    #         "/home/asher/Desktop/projects/sam_whistle/logs/10-06-2024_15-20-41_pu",
    #         "/home/asher/Desktop/projects/sam_whistle/logs/10-06-2024_14-10-33",
    #         "/home/asher/Desktop/projects/sam_whistle/logs/10-12-2024_23-04-06-fcn_spect",
    #         "/home/asher/Desktop/projects/sam_whistle/logs/10-13-2024_16-16-07-fcn_encoder"
    #     ]
    #     precision_range = []
    #     recall_range = []
    #     thresholds_range = []
    #     f1_range = []
    #     model_names = []

    #     for eval in evaluations:
    #         eval_file = os.path.join(eval, 'evaluation_results.pkl')
    #         with open(eval_file, 'rb') as f:
    #             data = pickle.load(f)
    #             precision_range.append(data['precision_range'])
    #             recall_range.append(data['recall_range'])
    #             thresholds_range.append(data['thresholds_range'])
    #             f1_range.append(data['f1_range'])
    #             model_names.append(data['model'])

    #     plot_precision_recall(precision_range, recall_range, thresholds_range, f1_range, model_names, save_path="outputs")
