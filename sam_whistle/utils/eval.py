import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List

@dataclass
class EvalResults:
    model_name: str
    precision: float  # Average precision
    recall: float    # At optimal threshold
    f1: float
    threshold: float  # Optimal threshold
    # For plotting PR curve
    precisions: List[float]  
    recalls: List[float]
    thresholds: List[float]
    

def evaluate_model(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  model_name: str) -> EvalResults:
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
    
    # Calculate PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find optimal threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Get binary predictions at optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    # Calculate final metrics
    final_f1 = f1_score(y_true, y_pred_binary)
    
    return EvalResults(
        model_name=model_name,
        precision=precisions[optimal_idx],
        recall=recalls[optimal_idx],
        f1=final_f1,
        threshold=optimal_threshold,
        precisions=precisions.tolist(),
        recalls=recalls.tolist(),
        thresholds=thresholds.tolist()
    )