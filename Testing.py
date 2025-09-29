import optuna
import numpy as np
from models import MLP, GCN, GAT, GIN, ModelWrapper
from encoding import XGBEncoder, make_xgbe_embeddings
import torch
import torch.nn as nn
from Helper_functions import FocalLoss, calculate_metrics, run_trial_with_cleanup
from training_functions import train_and_validate, train_and_test

import torch
models = ['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN']


def objective(trial, model, data, train_perf_eval, val_perf_eval, train_mask, val_mask):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    alpha = trial.suggest_float('alpha', 0.1, 1.0)
    gamma_focal = trial.suggest_float('gamma_focal', 2, 5)
    criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
    #Define model and hyperparameters for the trials per selected model.
    if model == 'MLP':
        from models import MLP
        model_instance = MLP(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units)
    elif model == 'SVM':
        from sklearn.svm import SVC
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        degree = trial.suggest_int('degree', 2, 5)
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        model_instance = SVC(kernel=kernel, C=C, class_weight='balanced', degree=degree)
    elif model == 'XGB':
        from xgboost import XGBClassifier
        max_depth = trial.suggest_int('max_depth', 5, 15)
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
        max_depth = trial.suggest_int('max_depth', 5, 15)
        model_instance = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model == 'GCN':
        from models import GCN
        model_instance = GCN(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units)
    elif model == 'GAT':
        from models import GAT
        num_heads = trial.suggest_int('num_heads', 1, 8)
        model_instance = GAT(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units, num_heads=num_heads)
    elif model == 'GIN':
        from models import GIN
        model_instance = GIN(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units)
    elif model == 'XGBe+GIN':
            # ---- XGBEncoder hyperparameters ----
            embedding_dim   = trial.suggest_categorical("xgbe_embedding_dim", [64, 96, 128, 192, 256])
            n_estimators    = trial.suggest_int("xgbe_n_estimators", 100, 600, step=50)
            max_depth       = trial.suggest_int("xgbe_max_depth", 3, 10)
            learning_rate   = trial.suggest_float("xgbe_learning_rate", 1e-3, 2e-1, log=True)
            subsample       = trial.suggest_float("xgbe_subsample", 0.5, 1.0)
            colsample_bt    = trial.suggest_float("xgbe_colsample_bytree", 0.5, 1.0)
            reg_lambda      = trial.suggest_float("xgbe_reg_lambda", 0.0, 10.0)
            reg_alpha       = trial.suggest_float("xgbe_reg_alpha", 0.0, 5.0)
            tree_method     = trial.suggest_categorical("xgbe_tree_method", ["hist"])  # add "gpu_hist" if available

            # ---- GIN hyperparameters ----
            gin_hidden      = trial.suggest_int("gin_hidden_units", 64, 256, step=32)
            #gin_layers      = trial.suggest_int("gin_layers", 2, 5)
            #dropout         = trial.suggest_float("gin_dropout", 0.0, 0.5)
            lr              = trial.suggest_float("gin_lr", 1e-4, 5e-2, log=True)
            weight_decay    = trial.suggest_float("gin_weight_decay", 1e-6, 1e-2, log=True)
            
            #Setting model instance
            from models import GIN
            model_instance = GIN(num_node_features=embedding_dim, num_classes=2, hidden_units=gin_hidden)
        
#Section where training and evaulation happens

    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN'] # models that use ModelWrapper
    if model in wrapper_models:
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
        model_wrapper.model.to(device)

    data = data.to(device)
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    
    
    if model in wrapper_models:
        metrics, best_model_wts, best_f1 = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs=200)
        return best_f1
    elif model == 'SVM':
        model_instance.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
        pred = model_instance.predict(data.x[val_perf_eval].cpu().numpy())
        metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), pred)
        return metrics['f1_illicit']
    elif model == 'XGB':
        model_instance.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
        pred = model_instance.predict(data.x[val_perf_eval].cpu().numpy())
        metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), pred)
        return metrics['f1_illicit']
    elif model == 'RF':
        model_instance.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
        pred = model_instance.predict(data.x[val_perf_eval].cpu().numpy())
        metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), pred)
        return metrics['f1_illicit']
    elif model == 'XGBe+GIN':
        xgbe = XGBEncoder(
            num_node_features=data.num_features,
            embedding_dim=embedding_dim,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bt,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=42,
            tree_method=tree_method,
            device=("cuda" if torch.cuda.is_available() else "cpu")
        )
        emb_data = make_xgbe_embeddings(xgbe, data, train_perf_eval, val_perf_eval, None)  # test not needed for val objective
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
        metrics, best_model_wts, best_f1 = train_and_validate(model_wrapper, emb_data, train_perf_eval, val_perf_eval, num_epochs=200)
        if best_f1 is None:
            exit("Best F1 is None, something went wrong during training.")
        return best_f1
    

from training_functions import train_and_test_NMW_models

def run_optimization(models, data, train_perf_eval, val_perf_eval, test_perf_eval, train_mask, val_mask):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_parameters = {
        'MLP': [],
        'SVM': [],
        'XGB': [],
        'RF': [],
        'GCN': [],
        'GAT': [],
        'GIN': [],
        'XGBe+GIN': [],
        'GINe+XGB': [],
        'LSTMe+GINe+XGB': [],
        'XGBe+LSTMe+GIN': []
    }
    testing_results = {
        'MLP': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'SVM': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'XGB': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'RF': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'GCN': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'GAT': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'GIN': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'XGBe+GIN': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'GINe+XGB': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'LSTMe+GINe+XGB': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        },
        'XGBe+LSTMe+GIN': {
            'precision_weighted': [],
            'precision_illicit': [],
            'recall_weighted': [],
            'recall_illicit': [],
            'f1_weighted': [],
            'f1_illicit': []
        }
    }
    for model_name in models:
            study = optuna.create_study(direction='maximize',
                                study_name= f'{model_name}_optimization',
                                storage='sqlite:///optimization_results.db',
                                load_if_exists=True)
            study.optimize(
                lambda trial: run_trial_with_cleanup(
                    objective, model_name, trial, model_name, data, train_perf_eval, val_perf_eval, train_mask, val_mask
                ),
                n_trials=200
            )

            print(f"Best hyperparameters for {model_name}:", study.best_params)
            model_parameters[model_name].append(study.best_params)
            #Assign hyperparameters to model for testing
            params_for_model = study.best_params
            
            hidden_units = params_for_model.get("hidden_units", 64)
            learning_rate = params_for_model.get("learning_rate", 0.045)
            weight_decay = params_for_model.get("weight_decay", 0.0001)
            alpha = params_for_model.get("alpha", 0.5)
            gamma_focal = params_for_model.get("gamma_focal", 2.0)
            for _ in range(30):
                match model_name:
                    case "MLP":
                        MLP_model = MLP(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
                        optimizer = torch.optim.Adam(MLP_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
                        model_wrapper = ModelWrapper(model=MLP_model, optimizer=optimizer, criterion=criterion)
                        
                        test_metrics, best_f1 = train_and_test(model_wrapper=model_wrapper,
                                                               data=data,
                                                               train_perf_eval=train_perf_eval,
                                                               val_perf_eval=val_perf_eval,
                                                               test_perf_eval=test_perf_eval)
                        testing_results[model_name]['precision_weighted'].append(test_metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(test_metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(test_metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(test_metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(test_metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(test_metrics['f1_illicit'])
                    case "SVM" | "RF" | "XGB":
                        metrics = train_and_test_NMW_models(model_name=model_name,
                                                            data=data,
                                                            train_perf_eval=train_perf_eval,
                                                            val_perf_eval=val_perf_eval,
                                                            test_perf_eval=test_perf_eval,
                                                            params_for_model=params_for_model)
                        testing_results[model_name]['precision_weighted'].append(metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(metrics['f1_illicit'])
                    case "GCN":
                        GCN_model = GCN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
                        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
                        model_wrapper = ModelWrapper(model=GCN_model, optimizer=optimizer, criterion=criterion)
                        
                        test_metrics, best_f1 = train_and_test(model_wrapper=model_wrapper,
                                                               data=data,
                                                               train_perf_eval=train_perf_eval,
                                                               val_perf_eval=val_perf_eval,
                                                               test_perf_eval=test_perf_eval)
                        testing_results[model_name]['precision_weighted'].append(test_metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(test_metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(test_metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(test_metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(test_metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(test_metrics['f1_illicit'])
                    case "GAT":
                        num_heads = params_for_model.get("num_heads", 4)
                        GAT_model = GAT(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units, num_heads=num_heads).to(device)
                        optimizer = torch.optim.Adam(GAT_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
                        model_wrapper = ModelWrapper(model=GAT_model, optimizer=optimizer, criterion=criterion)

                        test_metrics, best_f1 = train_and_test(model_wrapper=model_wrapper,
                                                               data=data,
                                                               train_perf_eval=train_perf_eval,
                                                               val_perf_eval=val_perf_eval,
                                                               test_perf_eval=test_perf_eval)
                        testing_results[model_name]['precision_weighted'].append(test_metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(test_metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(test_metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(test_metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(test_metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(test_metrics['f1_illicit'])
                    case "GIN":
                        GIN_model = GIN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to(device)
                        optimizer = torch.optim.Adam(GIN_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        criterion = FocalLoss(alpha=alpha, gamma=gamma_focal)
                        model_wrapper = ModelWrapper(model=GIN_model, optimizer=optimizer, criterion=criterion)
                        
                        test_metrics, best_f1 = train_and_test(model_wrapper=model_wrapper,
                                                               data=data,
                                                               train_perf_eval=train_perf_eval,
                                                               val_perf_eval=val_perf_eval,
                                                               test_perf_eval=test_perf_eval)
                        testing_results[model_name]['precision_weighted'].append(test_metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(test_metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(test_metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(test_metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(test_metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(test_metrics['f1_illicit'])
                    case "XGBe+GIN":
                        embedding_dim = params_for_model.get("xgbe_embedding_dim", 128)
                        n_estimators = params_for_model.get("xgbe_n_estimators", 300)
                        max_depth = params_for_model.get("xgbe_max_depth", 6)
                        learning_rate = params_for_model.get("xgbe_learning_rate", 0.05)
                        subsample = params_for_model.get("xgbe_subsample", 0.8)
                        colsample_bt = params_for_model.get("xgbe_colsample_bytree", 0.8)
                        reg_lambda = params_for_model.get("xgbe_reg_lambda", 1.0)
                        reg_alpha = params_for_model.get("xgbe_reg_alpha", 0.0)
                        tree_method = params_for_model.get("xgbe_tree_method", "hist")
                        gin_hidden = params_for_model.get("gin_hidden_units", 128)
                        lr = params_for_model.get("gin_lr", 0.01)
                        weight_decay = params_for_model.get("gin_weight_decay", 0.0001)
                        xgbe = XGBEncoder(
                            num_node_features=data.num_features,
                            embedding_dim=embedding_dim,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bt,
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha,
                            random_state=42,
                            tree_method=tree_method,
                            device=("cuda" if torch.cuda.is_available() else "cpu")
                        )
                        emb_data = make_xgbe_embeddings(xgbe, data, train_perf_eval, val_perf_eval, None)  # test not needed for val objective
                        model_instance = GIN(num_node_features=embedding_dim, num_classes=2, hidden_units=gin_hidden)
                        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr, weight_decay=weight_decay)
                        model_wrapper = ModelWrapper(model_instance, optimizer, criterion)
                        test_metrics, best_f1 = train_and_test(model_wrapper=model_wrapper,
                                                               data=emb_data,
                                                               train_perf_eval=train_perf_eval,
                                                               val_perf_eval=val_perf_eval,
                                                               test_perf_eval=test_perf_eval)
                        testing_results[model_name]['precision_weighted'].append(test_metrics['precision_weighted'])
                        testing_results[model_name]['precision_illicit'].append(test_metrics['precision_illicit'])
                        testing_results[model_name]['recall_weighted'].append(test_metrics['recall_weighted'])
                        testing_results[model_name]['recall_illicit'].append(test_metrics['recall_illicit'])
                        testing_results[model_name]['f1_weighted'].append(test_metrics['f1_weighted'])
                        testing_results[model_name]['f1_illicit'].append(test_metrics['f1_illicit'])
                    #Next adding other model

    return model_parameters, testing_results
            #need to start writing introduction
            #Start with excel sheet on places to publish paper
            
            
            