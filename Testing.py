import optuna
import numpy as np
from models import MLP, GCN, GAT, GIN, ModelWrapper
import torch
from Helper_functions import FocalLoss, calculate_metrics

models = ['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN']
model_parameters = {
    'MLP': [],
    'SVM': [],
    'XGB': [],
    'RF': [],
    'GCN': [],
    'GAT': [],
    'GIN': []}

def objective(trial, model, data, train_perf_eval, val_perf_eval):
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    gamma_focal = trial.suggest_float('gamma_focal', 2, 5)
    
    if model == 'MLP':
        from models import MLP
        model_instance = MLP(num_node_features=data.num_node_features, num_classes=data.num_classes, hidden_units=hidden_units)
    elif model == 'SVM':
        from sklearn.svm import SVC
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        degree = trial.suggest_int('degree', 2, 5)
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        model_instance = SVC(kernel=kernel, C=C, class_weight='balanced', degree=degree)
    elif model == 'XGB':
        from xgboost import XGBClassifier
        max_depth = trial.suggest_int('max_depth', 3, 10)
        Gamma_XGB = trial.suggest_float('Gamma_XGB', 0, 5)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        model_instance = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=9.25,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=0.7,
            gamma=Gamma_XGB
        )
    elif model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        model_instance = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model == 'GCN':
        from models import GCN
        model_instance = GCN(num_node_features=data.num_node_features, num_classes=data.num_classes, hidden_units=hidden_units)
    elif model == 'GAT':
        from models import GAT
        num_heads = trial.suggest_int('num_heads', 1, 8)
        model_instance = GAT(num_node_features=data.num_node_features, num_classes=data.num_classes, hidden_units=hidden_units, num_heads=num_heads)
    elif model == 'GIN':
        from models import GIN
        model_instance = GIN(num_node_features=data.num_node_features, num_classes=data.num_classes, hidden_units=hidden_units)
    
    criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
    
    
    
    
    
def run_optimization(models, data, train_perf_eval, val_perf_eval):
    for model_name in models:
            study = optuna.create_study(direction='maximize',
                                study_name= f'{model_name}_optimization',
                                storage='sqlite:///optimization_results.db',
                                load_if_exists=True)
            study.optimize(lambda trial: objective(trial, model_name, data, train_perf_eval, val_perf_eval), n_trials=200)
            print(f"Best hyperparameters for {model_name}:", study.best_params)
            model_parameters[model_name].append(study.best_params)