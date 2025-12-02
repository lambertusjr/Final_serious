import numpy as np
from Helper_functions import calculate_metrics

def test_calculate_metrics():
    # Case 1: Binary classification (0 and 1)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0])
    # Probabilities for class 1 (or class 0? Let's assume prob of class 1 for now as standard)
    # But wait, our code handles 2D or 1D.
    # Let's provide 2D probs.
    # [prob_class_0, prob_class_1]
    y_prob = np.array([
        [0.9, 0.1], # True 0, Pred 0
        [0.2, 0.8], # True 1, Pred 1
        [0.8, 0.2], # True 0, Pred 0
        [0.6, 0.4], # True 1, Pred 0 (Miss)
        [0.95, 0.05], # True 0, Pred 0
        [0.3, 0.7], # True 1, Pred 1
        [0.4, 0.6], # True 0, Pred 1 (False Alarm)
        [0.85, 0.15] # True 0, Pred 0
    ])
    
    print("Testing calculate_metrics with 2D probabilities...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert 'prauc' in metrics
    assert 'kappa' in metrics
    assert metrics['roc_auc'] != -1
    
    # Case 2: 1D probabilities (prob of class 1 usually)
    y_prob_1d = y_prob[:, 1]
    print("\nTesting calculate_metrics with 1D probabilities...")
    metrics_1d = calculate_metrics(y_true, y_pred, y_prob_1d)
    for k, v in metrics_1d.items():
        print(f"{k}: {v}")
        
    assert metrics_1d['roc_auc'] == metrics['roc_auc']
    
    print("\nVerification passed!")

if __name__ == "__main__":
    test_calculate_metrics()
