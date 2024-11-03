import pickle
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import tyro
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from sam_whistle.model import SAM_whistle, Detection_ResNet_BN2, FCN_Spect, FCN_encoder
from sam_whistle.model.loss import Charbonnier_loss, DiceLoss
from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch
from sam_whistle.config import Args
from sam_whistle import utils
from sam_whistle.visualization import visualize

@torch.no_grad()
def evaluate_sam(args: Args):
    print("------------Evaluating model------------")
    model = SAM_whistle(args,)
    model.to(args.device)
    if not args.freeze_img_encoder:
        model.img_encoder.load_state_dict(torch.load(os.path.join(args.save_path, 'img_encoder.pth')))
    if not args.freeze_mask_decoder:
        model.decoder.load_state_dict(torch.load(os.path.join(args.save_path, 'decoder.pth')))
    if not args.freeze_prompt_encoder:
        model.sam_model.prompt_encoder.load_state_dict(torch.load(os.path.join(args.save_path, 'prompt_encoder.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    testset = WhistleDataset(args, 'test',model.sam_model.image_encoder.img_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_workers,)
    print(f"Test set size: {len(testset)}")

    model.eval()
    test_losses = []
    gt_masks = []
    pred_masks = []
    for i, data in enumerate(tqdm(testloader)):
        with torch.no_grad():
            spect, gt_mask= data['spect'], data['contour_mask']
            loss, pred_mask, low_mask, _ = model(data)
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)
            if args.visualize_eval:
                # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
                spect = spect.permute(0, 2, 3, 1)
                visualize(spect, gt_mask, pred_mask, output_path, str(i)+"_cropped")
            test_losses.append(loss.item())

    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")
    gt_masks = torch.cat(gt_masks, dim=0).flatten().cpu().numpy()
    pred_masks = torch.cat(pred_masks, dim=0).flatten().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_masks, pred_masks)

    return precision, recall, thresholds


@torch.no_grad()
def evaluate_pu(args: Args):
    print("------------Evaluating model------------")
    model = Detection_ResNet_BN2(args.pu_width)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_pu.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
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


def evaluate_fcn_spect(args: Args):
    print("------------Evaluating model------------")
    model = FCN_Spect(args)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    testset = WhistleDataset(args, 'test',spect_nchan=1)
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

def evaluate_fcn_encoder(args: Args):
    print("------------Evaluating model------------")
    model = FCN_encoder(args)
    model.to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    testset = WhistleDataset(args, 'test',spect_nchan=1)
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
    args = tyro.cli(Args)
    args.evaluate = True
    if args.single_eval:
        if args.model == 'sam':
            precision, recall, thresholds= evaluate_sam(args)
            precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'sam', save_path=args.save_path)
            model_names = ['SAM']
        elif args.model=='pu':
            precision, recall, thresholds= evaluate_pu(args)
            precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'pu', save_path=args.save_path)
            model_names = ['PU']
        elif args.model=='fcn_spect':
            precision, recall, thresholds= evaluate_fcn_spect(args)
            precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'fcn_spect', save_path=args.save_path)
            model_names = ['FCN_Spect']
        elif args.model=='fcn_encoder':
            precision, recall, thresholds= evaluate_fcn_encoder(args)
            precision_range, recall_range, thresholds_range, f1_range= filter_precision_recall(precision, recall, thresholds, 'fcn_encoder', save_path=args.save_path)
            model_names = ['FCN_Encoder']
        else:
            raise ValueError(f"Model {args.model} not found")
        
        if not isinstance(precision_range, list):
            precision_range = [precision_range]
            recall_range = [recall_range]
            thresholds_range = [thresholds_range]
            f1_range = [f1_range]
        plot_precision_recall(precision_range, recall_range, thresholds_range, f1_range, model_names, save_path=args.save_path)
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
