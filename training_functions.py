"""
What I need in the generic training function:
best_f1 = -1
best_f1_model_wts = None

"""
import torch
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from models import ModelWrapper

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
        #print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = model_wrapper.train_step(data, train_perf_eval)
        val_loss, val_metrics = model_wrapper.evaluate(data, val_perf_eval)
        #print(f'Validation F1 Score: {val_metrics["f1_illicit"]:.4f}')
        
        metrics['precision_weighted'].append(val_metrics['precision_weighted'])
        metrics['precision_illicit'].append(val_metrics['precision_illicit'])
        metrics['recall_weighted'].append(val_metrics['recall_weighted'])
        metrics['recall_illicit'].append(val_metrics['recall_illicit'])
        metrics['f1_weighted'].append(val_metrics['f1_weighted'])
        metrics['f1_illicit'].append(val_metrics['f1_illicit'])
        
        best_f1, best_model_wts = update_best_weights(model_wrapper.model, best_f1, val_metrics['f1_illicit'], best_f1_model_wts)
    return metrics, best_model_wts, best_f1
import copy
def update_best_weights(model, best_f1, current_f1, best_f1_model_wts):
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_model_wts = copy.deepcopy(model.state_dict())
    return best_f1, best_f1_model_wts

def train_and_test(model_wrapper, data, train_perf_eval, val_perf_eval, test_perf_eval, num_epochs=200):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper.model.to(device)
    data = data.to(device)
    #Use training and validation data to train the model
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    test_perf_eval = test_perf_eval.to(device)
    
    metrics, best_model_wts, best_f1 = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs)
    
    model_wrapper.model.load_state_dict(best_model_wts)
    test_loss, test_metrics = model_wrapper.evaluate(data, test_perf_eval)
    
    return test_metrics, best_f1
    