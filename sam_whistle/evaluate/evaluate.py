import pickle
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
import numpy as np
import tyro
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass, asdict


from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch, custom_collate_fn
from sam_whistle import config
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
def evaluate_sam_prediction(cfg: config.SAMConfig, load=False, model: SAM_whistle = None, testloader: DataLoader = None, loss_fn: nn.Module=None, visualize_eval=False, visualize_name=''):
    if load:
        model = SAM_whistle(cfg,)
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
        
        testset = WhistleDataset(cfg, 'test',model.sam_model.image_encoder.img_size)
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
        spect, gt_mask = data['spect'], data['gt_mask']
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

def evaluate_sam(args: config.SAMConfig, visualize_eval=False, visualize_name=''):
    test_loss, all_gts, all_preds = evaluate_sam_prediction(args, load= True, visualize_eval = visualize_eval, visualize_name=visualize_name)
    print(f"Test Loss: {test_loss}")
    eval_res: utils.EvalResults = utils.evaluate_model(all_gts, all_preds, "SAM")
    print(f"Precision: {eval_res.precision}, Recall: {eval_res.recall}, F1: {eval_res.f1}, Threshold: {eval_res.threshold}")
    with open(os.path.join(args.log_dir, 'eval_sam_results.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)
    
    plt.figure(figsize=(10, 8))
    plt.plot(eval_res.recalls, eval_res.precisions, 
            label=f'{eval_res.model_name} (AP={eval_res.precision:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.log_dir, 'pr_curves.png'))
    plt.close()


@torch.no_grad()
def evaluate_pu(args):
    print("------------Evaluating model------------")
    model = Detection_ResNet_BN2(args.pu_width)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'model_pu.pth')))
    output_path = os.path.join(args.log_dir, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    # trainset = WhistleDataset(args, 'train', model.sam_model.image_encoder.img_size)
    # trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), drop_last=True)
    testset = WhistlePatch(args, 'test',)
    testloader = DataLoader(testset, batch_size=args.pu_batch_size, shuffle=False, num_workers=args.num_workers,)
    print(f"Test set size: {len(testset)}")
    
    loss_fn  = Charbonnier_loss()
    
    model.eval()
    test_losses = []
    gt_masks = []
    pred_masks = []
    for i, data in enumerate(tqdm(testloader)):
        with torch.no_grad():
            img, gt_mask, meta = data
            img = img.to(args.device)
            gt_mask = gt_mask.to(args.device)
            pred_mask = model(img)
            test_loss = loss_fn(pred_mask, gt_mask)
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)

            if args.visualize_eval:
                # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
                visualize(img, gt_mask, pred_mask, output_path, i)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")
    gt_masks = torch.cat(gt_masks, dim=0).flatten().cpu().numpy()
    pred_masks = torch.cat(pred_masks, dim=0).flatten().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_masks, pred_masks)

    return precision, recall, thresholds


def evaluate_fcn_spect(args):
    print("------------Evaluating model------------")
    model = FCN_Spect(args)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'model.pth')))
    output_path = os.path.join(args.log_dir, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    testset = WhistleDataset(args, 'test',spec_nchan=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers,)
    print(f"Test set size: {len(testset)}")
    loss_fn  = Charbonnier_loss()
    
    model.eval()
    model.init_patch_ls(testset[0][0].shape[-2:])
    test_losses = []
    gt_masks = []
    pred_masks = []
    for i, data in enumerate(tqdm(testloader)):
        with torch.no_grad():
            img, gt_mask= data
            img = img.to(args.device)
            gt_mask = gt_mask.to(args.device)
            model.order_pick_patch()
            pred_mask, gt_mask = model(img, gt_mask)
            test_loss = loss_fn(pred_mask, gt_mask)
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)

            if args.visualize_eval:
                # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
                visualize(img, gt_mask, pred_mask, output_path, i)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")
    gt_masks = torch.cat(gt_masks, dim=0).flatten().cpu().numpy()
    pred_masks = torch.cat(pred_masks, dim=0).flatten().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_masks, pred_masks)

    return precision, recall, thresholds

def evaluate_fcn_encoder(args):
    print("------------Evaluating model------------")
    model = FCN_encoder(args)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'model.pth')))
    output_path = os.path.join(args.log_dir, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    testset = WhistleDataset(args, 'test',spec_nchan=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers,)
    print(f"Test set size: {len(testset)}")
    loss_fn = DiceLoss()
    
    # model.eval()
    model.init_patch_ls()
    test_losses = []
    gt_masks = []
    pred_masks = []
    for i, data in enumerate(tqdm(testloader)):
        with torch.no_grad():
            img, gt_mask= data
            img = img.to(args.device)
            gt_mask = gt_mask.to(args.device)
            pred_mask = model(img)
            test_loss = loss_fn(pred_mask, gt_mask)
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)

            if args.visualize_eval:
                # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
                visualize(img, gt_mask, pred_mask, output_path, i)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")
    gt_masks = torch.cat(gt_masks, dim=0).flatten().cpu().numpy()
    pred_masks = torch.cat(pred_masks, dim=0).flatten().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_masks, pred_masks)

    return precision, recall, thresholds


def filter_precision_recall(precision, recall, thresholds, model_name, min_threshold=0, max_threshold=1, save_path=None):
    precision = precision[:-1]
    recall = recall[:-1]

    # filter
    valid_indices = (precision > 0) & (recall > 0)
    precision_filtered = precision[valid_indices]
    recall_filtered = recall[valid_indices]
    thresholds_filtered = thresholds[valid_indices]
    f1_filtered =  2 * (precision_filtered * recall_filtered) / (precision_filtered + recall_filtered)
    

    range_indices = (thresholds >= min_threshold) & (thresholds <= max_threshold)

    # Further filtering based on the limits for precision and recall
    precision_range = precision_filtered[range_indices]
    recall_range = recall_filtered[range_indices]
    f1_range = f1_filtered[range_indices]
    thresholds_range = thresholds_filtered[range_indices]
    print(f"Best F1 score: {np.max(f1_range)}")
    if save_path is not None:
        # Save precision_range, recall_range, thresholds_range, f1_range in a file
        if save_path is not None:
            data = {
                'model': model_name,
                'precision_range': precision_range,
                'recall_range': recall_range,
                'thresholds_range': thresholds_range,
                'f1_range': f1_range
            }
            file_path = os.path.join(save_path, 'evaluation_results.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

    return precision_range, recall_range, thresholds_range, f1_range


def plot_precision_recall(precision, recall, thresholds, f1_score, model_names, save_path=None):
    plt.figure(figsize=(8, 6))
    # Optional: Plot F1 score iso-contours
    precision_grid, recall_grid = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    f1_grid = 2 * (precision_grid * recall_grid) / (precision_grid + recall_grid)
    fi_contour = plt.contour(recall_grid, precision_grid, f1_grid, levels=np.linspace(0.1, 0.9, 9), colors='green', linestyles='dashed')
    
    for r, p, f, model in zip(recall, precision, f1_score, model_names):
        plt.plot(r, p)
        best_idx = np.argmax(f)
        plt.scatter(r[best_idx], p[best_idx], label=f'Best F1: {model}', zorder=5)
    
    plt.clabel(fi_contour, fmt='%.2f', inline=True, fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend(loc = 'lower left')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(os.path.join(save_path, 'precision_recall_curve.png'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sam', help='Model to evaluate')
    parser.add_argument('--eval_single', action = 'store_true', help='Evaluate a single model')
    parser.add_argument('--visual', action = 'store_true', help='Visualize the predictions')
    parser.add_argument('--visual_name', type=str, default='', help='Name of the visualized file')
    args, remaining = parser.parse_known_args()
    if args.eval_single:

        if args.model == 'sam':
            cfg = tyro.cli(config.SAMConfig, args=remaining)
            evaluate_sam(cfg, visualize_eval=args.visual, visualize_name=args.visual_name)
        # if args.model == 'sam':
        #     precision, recall, thresholds= evaluate_sam(args)
        #     precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'sam', save_path=args.log_dir)
        #     model_names = ['SAM']
        # elif args.model=='pu':
        #     precision, recall, thresholds= evaluate_pu(args)
        #     precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'pu', save_path=args.log_dir)
        #     model_names = ['PU']
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
    else:
        evaluations =[
            "/home/asher/Desktop/projects/sam_whistle/logs/10-06-2024_15-20-41_pu",
            "/home/asher/Desktop/projects/sam_whistle/logs/10-06-2024_14-10-33",
            "/home/asher/Desktop/projects/sam_whistle/logs/10-12-2024_23-04-06-fcn_spect",
            "/home/asher/Desktop/projects/sam_whistle/logs/10-13-2024_16-16-07-fcn_encoder"
        ]
        precision_range = []
        recall_range = []
        thresholds_range = []
        f1_range = []
        model_names = []

        for eval in evaluations:
            eval_file = os.path.join(eval, 'evaluation_results.pkl')
            with open(eval_file, 'rb') as f:
                data = pickle.load(f)
                precision_range.append(data['precision_range'])
                recall_range.append(data['recall_range'])
                thresholds_range.append(data['thresholds_range'])
                f1_range.append(data['f1_range'])
                model_names.append(data['model'])

        plot_precision_recall(precision_range, recall_range, thresholds_range, f1_range, model_names, save_path="outputs")
