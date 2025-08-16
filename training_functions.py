"""
What I need in the generic training function:
best_f1 = -1
best_f1_model_wts = None

"""

def train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs):
    best_f1 = -1
    best_f1_model_wts = None
    metrics = {
        'precision_weighted': [],
        'precision_illicit': [],
        'recall_weighted': [],
        'recall_illicit': [],
        'f1_weighted': [],
        'f1_illicit': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = model_wrapper.train_step(data, train_perf_eval)
        val_loss, val_metrics = model_wrapper.evaluate(data, val_perf_eval)
        
        metrics['precision_weighted'].append(val_metrics['precision_weighted'])
        metrics['precision_illicit'].append(val_metrics['precision_illicit'])
        metrics['recall_weighted'].append(val_metrics['recall_weighted'])
        metrics['recall_illicit'].append(val_metrics['recall_illicit'])
        metrics['f1_weighted'].append(val_metrics['f1_weighted'])
        metrics['f1_illicit'].append(val_metrics['f1_illicit'])
        
        best_f1, best_model_wts = update_best_weights(model_wrapper.model, best_f1, val_metrics['f1_illicit'])
    return metrics, best_model_wts
import copy
def update_best_weights(model, best_f1, current_f1):
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_model_wts = copy.deepcopy(model.state_dict())
    return best_f1, best_f1_model_wts