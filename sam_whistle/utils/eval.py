import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from dataclasses import dataclass, asdict
from typing import Dict, List
import os

@dataclass
class PR_Results:
    model_name: str
    precision: float  
    recall: float   
    f1: float
    threshold: float
    # For plotting PR curve
    precisions: List[float]  
    recalls: List[float]
    thresholds: List[float]


def plot_pr_curve(eval_res_li:list[PR_Results], fig_dir:str, figname='pr_curve.png'):
    
    plt.figure(figsize=(10, 8))
    precision_grid, recall_grid = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    f1_grid = 2 * (precision_grid * recall_grid) / (precision_grid + recall_grid)
    f1_contour = plt.contour(recall_grid, precision_grid, f1_grid, levels=np.linspace(0.1, 0.9, 9), colors='green', linestyles='dashed')

    legend_handles1 = []
    legend_handles2 = []
    legend_labels1 = []
    legend_labels2 = []
    for eval_res in eval_res_li:
        line,  = plt.plot(eval_res.recalls, eval_res.precisions, )
        plt.scatter(eval_res.recall, eval_res.precision, zorder=5)
        legend_handles1.append(Line2D([0], [0], color= line.get_color(), lw=2))
        legend_handles2.append(Line2D([], [], marker='o', color = line.get_color(), lw=0, linestyle='None', markersize=8))

        legend_labels1.append(eval_res.model_name)
        legend_labels2.append(f'F1: {eval_res.f1:.3f}, Thre: {eval_res.threshold:.3f}')
        
    legend_handles = legend_handles1 + legend_handles2
    legend_labels = legend_labels1 + legend_labels2
    
    plt.clabel(f1_contour, fmt='%.2f', inline=True, fontsize=10)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    # for i in range(0, len(legend_handles), 2):
    #     plt.legend(handles = legend_handles[i:i+2], labels=legend_labels[i:i+2], loc = 'lower left', fontsize=10, ncol=2, frameon=False)
    plt.legend(handles = legend_handles, labels=legend_labels, loc = 'lower left', fontsize=10, ncols=2)
    plt.grid(True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(os.path.join(fig_dir, figname))
    plt.close()


def eval_conf_map(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  model_name: str,
                  min_thre = 0, 
                  max_thre=1) -> PR_Results:
    """
    Evaluate segmentation model performance.
    
    Args:
        y_true: Ground truth masks (N, H, W) or flattened
        y_pred: Predicted probabilities (N, H, W) or flattened
        model_name: Name of the model for recording
    """
    # Flatten if needed
    if y_true.ndim > 1:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    range_idx = np.where((thresholds >= min_thre) & (thresholds <= max_thre))
    precisions = precisions[range_idx]
    recalls = recalls[range_idx]
    thresholds = thresholds[range_idx]

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    final_f1 = f1_score(y_true, y_pred_binary)
    
    return PR_Results(
        model_name=model_name,
        precision=precisions[optimal_idx],
        recall=recalls[optimal_idx],
        f1=final_f1,
        threshold=optimal_threshold,
        precisions=precisions.tolist(),
        recalls=recalls.tolist(),
        thresholds=thresholds.tolist()
    )

def eval_tonal_map(pr, model_name):
    precisions, recalls, thresholds = map(np.array, zip(*pr),)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    return PR_Results(
        model_name= model_name,
        precision=precisions[optimal_idx],
        recall=recalls[optimal_idx],
        f1=f1_scores[optimal_idx],
        threshold=optimal_threshold,
        precisions=precisions.tolist(),
        recalls=recalls.tolist(),
        thresholds=thresholds.tolist()
    )


def f1_pr(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)