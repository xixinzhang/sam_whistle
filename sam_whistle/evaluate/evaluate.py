import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import tyro
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from sam_whistle.model.model import SAM_whistle
from sam_whistle.datasets.dataset import WhistleDataset
from sam_whistle.config import Args
from sam_whistle import utils
from sam_whistle.visualization import visualize

@torch.no_grad()
def evaluate(args: Args):
    print("------------Evaluating model------------")
    model = SAM_whistle(args,)
    model.to(args.device)
    model.decoder.load_state_dict(torch.load(os.path.join(args.save_path, 'decoder.pth')))
    output_path = os.path.join(args.save_path, 'predictions')
    if not os.path.exists(output_path) and args.visualize_eval:
        os.makedirs(output_path)
    
    trainset = WhistleDataset(args, 'train', model.sam_model.image_encoder.img_size)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), drop_last=True)
    testset = WhistleDataset(args, 'test',model.sam_model.image_encoder.img_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=os.cpu_count(),)
    print(f"Train set size: {len(trainset)}, Test set size: {len(testset)}")

    model.eval()
    test_losses = []
    gt_masks = []
    pred_masks = []
    for i, data in enumerate(tqdm(testloader)):
        with torch.no_grad():
            spect, _, _, gt_mask, points= data 
            loss, pred_mask, low_mask = model(data)
            gt_masks.append(gt_mask)
            pred_masks.append(pred_mask)
            if args.visualize_eval:
                # utils.visualize_array(low_mask.cpu().numpy(), output_path, i, 'low_res')
                visualize(spect, gt_mask, points, pred_mask, output_path, i)
            test_losses.append(loss.item())

    test_loss = np.mean(test_losses)
    print(f"Test Loss: {test_loss}")
    gt_masks = torch.cat(gt_masks, dim=0).flatten().cpu().numpy()
    pred_masks = torch.cat(pred_masks, dim=0).flatten().cpu().numpy()
    precision, recall, threshold = precision_recall_curve(gt_masks, pred_masks)
    
    return precision, recall, threshold


def plot_precision_recall(precision, recall, threshold, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision,  )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    # Optional: Plot F1 score iso-contours
    precision_grid, recall_grid = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    f1_score = 2 * (precision_grid * recall_grid) / (precision_grid + recall_grid)
    contour = plt.contour(recall_grid, precision_grid, f1_score, levels=np.linspace(0.1, 0.9, 9), colors='green', linestyles='dashed')
    plt.clabel(contour, fmt='%.2f', inline=True, fontsize=10)
    plt.savefig(os.path.join(save_path, 'precision_recall_curve.png'))

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.evaluate = True
    precision, recall, threshold = evaluate(args)
    plot_precision_recall(precision, recall, threshold, save_path=args.save_path)
