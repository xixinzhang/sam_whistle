import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from dataclasses import dataclass, asdict
from typing import Dict, List
import os
from matplotlib import rcParams
import seaborn as sns

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


def plot_pr_curve(eval_res_li: list, fig_dir: str, figname='pr_curve.jpg', xlim_min=None, xlim_max=None, ylim_min=None, ylim_max=None, legend = True, colors = None):
    # Set the style to match Nature's guidelines
    plt.style.use('seaborn-v0_8-white')
    
    # Set font to Arial (Nature's preferred font)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    
    # Create figure with high DPI for print quality
    plt.figure(figsize=(5, 5), dpi=300)
    
    # Create color palette
    if colors is None:
        colors = sns.color_palette('husl', n_colors=len(eval_res_li))
    else:
        colors = sns.color_palette(colors, n_colors=len(eval_res_li))
    
    model_name_map = {
        'sam':'SAM-Whistle',
        'deep':'Deep-Whistle',
        'fcn_spect':'FCN-Spect',
        'fcn_encoder':'FCN-Encoder',
        'graph_search':'Graph Search'
    }


    # Create precision-recall grid for F1 score contours
    precision_grid, recall_grid = np.meshgrid(np.linspace(0.01, 1, 100), 
                                            np.linspace(0.01, 1, 100))
    f1_grid = 2 * (precision_grid * recall_grid) / (precision_grid + recall_grid)
    
    # Plot F1 score contours with refined styling
    f1_contour = plt.contour(recall_grid, precision_grid, f1_grid,
                            levels=np.linspace(0.2, 0.9, 8), 
                            colors='gray',
                            linestyles='dashed',
                            linewidths=0.8,
                            alpha=0.6)
    
    # Initialize legend handles and labels
    legend_handles1, legend_handles2 = [], []
    legend_labels1, legend_labels2 = [], []
    
    # Plot PR curves for each model
    for idx, eval_res in enumerate(eval_res_li):
        
        line, = plt.plot(eval_res.recalls, 
                        eval_res.precisions,
                        color=colors[idx],
                        linewidth=1.75,
                        alpha=0.9,
                        zorder=2*idx+1)
        markers = ['o', 's', '^', 'D', 'p', 'h', '8', 'v'] 

        # Plot optimal point with refined styling
        plt.scatter(eval_res.recall,
                   eval_res.precision,
                #    color=colors[idx],
                    facecolor=colors[idx],
                    edgecolor='black',
                   marker=markers[idx],    
                   s=30,
                   linewidth= 0.4,
                   zorder=5)
        
        # Create legend entries
        legend_handles1.append(Line2D([0], [0], color=colors[idx], lw=2))
        legend_handles2.append(Line2D([], [], 
                                    marker=markers[idx % len(markers)],
                                    color=colors[idx],
                                    lw=0,
                                    markersize=5,
                                    markeredgecolor='black',
                                    markeredgewidth=0.3))
        
        if legend:
            legend_labels1.append(model_name_map[eval_res.model_name])
            legend_labels2.append(f'Optimal F1: {eval_res.f1:.3f} ({r'$\tau$'}={eval_res.threshold:.3f})')
    
    # Combine legend handles and labels
    legend_handles = legend_handles1 + legend_handles2
    legend_labels = legend_labels1 + legend_labels2
    
    # Add contour labels with refined styling
    plt.clabel(f1_contour,
               fmt='%.1f',
               inline=True,
               fontsize=8,
               colors='dimgray')
    
    # Customize axes
    plt.xlabel('Recall', fontsize=12, labelpad=10)
    plt.ylabel('Precision', fontsize=12, labelpad=10)
    # plt.title('Precision-Recall Curves', fontsize=14, pad=15)
    
    # Add legend with refined styling
    if legend:
        plt.legend(handles=legend_handles,
                labels=legend_labels,
                loc='lower left',
                fontsize=9,
                ncol=2,
                frameon=True,
                fancybox=False,
                edgecolor='black',
                bbox_to_anchor=(0, 0),
                framealpha=0.5,
                columnspacing=0.5)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    xlim_min = xlim_min if xlim_min is not None else 0
    xlim_max = xlim_max if xlim_max is not None else 1.02
    ylim_min = ylim_min if ylim_min is not None else 0
    ylim_max = ylim_max if ylim_max is not None else 1.02
    
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    
    # Adjust tick spacing based on the range
    x_range = xlim_max - xlim_min
    y_range = ylim_max - ylim_min
    
    plt.xticks(np.arange(np.floor(xlim_min*10)/10, xlim_max+0.01, 0.1), fontsize=10)
    plt.yticks(np.arange(np.floor(ylim_min*10)/10, ylim_max+0.01, 0.1), fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    
    # Save in high quality PDF (vector format preferred for publication)
    plt.savefig(os.path.join(fig_dir, figname),
                dpi=300,
                bbox_inches='tight',)
    plt.close()

def plot_grouped_metrics(eval_res_li: list, fig_dir: str, figname='grouped_metrics.jpg', 
                         colors=None, width=0.15):
    """
    Plot grouped bar chart comparing models for each metric (Precision, Recall, F1).
    
    Parameters:
    eval_res_li (list): List of evaluation results containing lists of precisions, recalls, and thresholds
    fig_dir (str): Directory to save the figure
    figname (str): Name of the output figure file
    colors (list): List of colors for different models
    width (float): Width of each bar
    """
    # Set the style to match Nature's guidelines
    plt.style.use('seaborn-v0_8-white')
    
    # Set font to Arial (Nature's preferred font)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Create figure with high DPI for print quality
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Create color palette for different models
    if colors is None:
        colors = sns.color_palette('husl', n_colors=len(eval_res_li))
    else:
        colors = sns.color_palette(colors, n_colors=len(eval_res_li))
    
    model_name_map = {
        'sam':'SAM-Whistle',
        'deep':'Deep-Whistle',
        'fcn_spect':'FCN-Spect',
        'fcn_encoder':'FCN-Encoder',
        'graph_search':'Graph Search'
    }

    # Prepare data
    models = [model_name_map[res.model_name] for res in eval_res_li]
    n_models = len(models)
    metrics = ['Precision', 'Recall', 'F1']
    n_metrics = len(metrics)
    
    # Calculate metrics for each model
    metric_values = {
        'Precision': {'means': [], 'stds': []},
        'Recall': {'means': [], 'stds': []},
        'F1': {'means': [], 'stds': []}
    }
    
    for res in eval_res_li:
        # Calculate mean values
        precision_mean = np.mean(res.precisions)
        recall_mean = np.mean(res.recalls)
        f1_scores = 2 * (np.array(res.precisions) * np.array(res.recalls)) / (np.array(res.precisions) + np.array(res.recalls) + 1e-10)
        f1_mean = np.mean(f1_scores)
        
        # Calculate standard deviations
        precision_std = np.std(res.precisions)
        recall_std = np.std(res.recalls)
        f1_std = np.std(f1_scores)
        
        # Store values
        metric_values['Precision']['means'].append(precision_mean.item())
        metric_values['Recall']['means'].append(recall_mean.item())
        metric_values['F1']['means'].append(f1_mean.item())
        
        metric_values['Precision']['stds'].append(precision_std.item())
        metric_values['Recall']['stds'].append(recall_std.item())
        metric_values['F1']['stds'].append(f1_std.item())
    
    print(metric_values)
    # Calculate x positions for bars
    x = np.arange(n_metrics)
    
    # Plot bars for each model
    bars = []
    for idx, model in enumerate(models):
        # Calculate bar positions for this model
        bar_positions = x - ((n_models-1)/2 - idx) * width
        
        # Get means and stds for this model
        means = [metric_values[m]['means'][idx] for m in metrics]
        stds = [metric_values[m]['stds'][idx] for m in metrics]
        
        # Plot bars
        bar = plt.bar(bar_positions, means, width, label=model,
                     color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add error bars
        plt.errorbar(bar_positions, means, yerr=stds, fmt='none',
                    color='black', capsize=8, capthick=0.8, linewidth=1)
        
        
        # for i, rect in enumerate(bar):
        #     height = means[i]
        #     std = stds[i]
        #     # Position text above error bar with small offset
        #     y_pos = height + std + 0.02
        #     plt.text(rect.get_x() + rect.get_width()/2., y_pos,
        #              f'{height:.2f}{r"$\pm$"}{std:.2f}',
        #              ha='center', va='bottom',
        #              fontsize=10)

        bars.append(bar)
    
    # # Customize axes
    # plt.xlabel('Metrics', fontsize=12, labelpad=10)
    # plt.ylabel('Score', fontsize=12, labelpad=10)
    
    # Set x-axis ticks
    plt.xticks(x, metrics, rotation=0, fontsize=15)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)


    # Customize y-axis
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
    
    # Add legend with refined styling
    plt.legend(loc='upper center',
              bbox_to_anchor=(0.5, 1.2),
              fontsize=12,
              frameon=True,
              fancybox=False,
              edgecolor='black',
              framealpha=0.8,
              ncol= n_models)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(fig_dir, figname),
                dpi=300,
                bbox_inches='tight')
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