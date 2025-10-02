# Final restart of my code
#notes:


seeded_run = True
prototyping = False
MLP_prototype = False
svm_prototype = False
XGB_prototype = False
RF_prototype = False
parameter_tuning = False
validation_runs = False
Full_run = True
num_epochs = 20
#%% Setup
# Detecting system
import platform
pc = platform.system()

import os
if pc == 'Darwin':
    os.chdir("/Users/lambertusvanzyl/Desktop/Final_serious")
else:
    os.chdir("/Users/Lambertus/Desktop/Final_serious")

#Setting seed
import torch
import numpy as np
if seeded_run == True:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
else: 
    seed = np.random.SeedSequence().entropy
    
#Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx

from functools import partial

#Importing PyTorch libraries
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool

#Importing sklearn libraries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import sys
#%% Importing custom libraries
from Reading_files import readfiles
from pre_processing import elliptic_pre_processing, create_data_object, create_elliptic_masks
from debugging import print_tensor_info
from models import GCN, ModelWrapper, MLP
from training_functions import train_and_validate
from Helper_functions import FocalLoss, calculate_metrics
from analysis import results_to_long_df, summarise_long_df, formatted_wide_table, produce_tables, boxplots_by_metric, bar_means_with_ci
#%% Getting data ready for models
features_df, classes_df, edgelist_df = readfiles(pc)
features_df, classes_df, edgelist_df, known_nodes = elliptic_pre_processing(features_df, classes_df, edgelist_df)
data = create_data_object(features_df, classes_df, edgelist_df)
data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

train_mask, val_mask, test_mask, train_perf_eval, val_perf_eval, test_perf_eval = create_elliptic_masks(data)

print_tensor_info(
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    train_perf_eval=train_perf_eval,
    val_perf_eval=val_perf_eval,
    test_perf_eval=test_perf_eval
)
#%% Testing if the model runs
if prototyping == True:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GCN(num_node_features=data.num_features, num_classes=2, hidden_units=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    data = data.to(device)
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    model_wrapper = ModelWrapper(model, optimizer, criterion)
    #Beginning training
    metrics, best_model_wts = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if MLP_prototype == True:
    model = MLP(num_node_features=data.num_features, num_classes=2, hidden_units=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model_wrapper = ModelWrapper(model, optimizer, criterion)
    
    data = data.to(device)
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    metrics, best_model_wts, best_f1 = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs)

if svm_prototype == True: #Toets ander kernel functions, regularisation parameters en gamma values
    from sklearn import svm
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(data.x[train_perf_eval], data.y[train_perf_eval])
    y_pred = clf.predict(data.x[val_perf_eval])
    val_metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), y_pred)
    
if XGB_prototype == True:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=9.25, learning_rate=0.1, max_depth=6, n_estimators=100, colsample_bytree=0.7)
    xgb_model.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
    y_pred = xgb_model.predict(data.x[val_perf_eval].cpu().numpy())
    val_metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), y_pred)
    
if RF_prototype == True:
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
    y_pred = rf_model.predict(data.x[val_perf_eval].cpu().numpy())
    val_metrics = calculate_metrics(data.y[val_perf_eval].cpu().numpy(), y_pred)
    
#%% Optuna

def objective(trial, data, train_perf_eval, val_perf_eval):
    #Setting hyperparameters for optuna runs
    hidden_units = trial.suggest_categorical("hidden_units", [16, 32, 48, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    alpha = trial.suggest_float("alpha", 0.2, 0.7)
    gamma = trial.suggest_float("gamma", 2.0, 5.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    
    model = GCN(num_node_features=data.num_features, num_classes=2, hidden_units=hidden_units).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    model_wrapper = ModelWrapper(model, optimizer, criterion)
    
    metrics, best_model_wts, best_f1 = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs)
    
    
    return best_f1

if parameter_tuning == True:
    import optuna
    import numpy as np
    #criterion = nn.CrossEntropyLoss()
    
    study = optuna.create_study(
        direction="maximize",
        study_name="GCN Hyperparameter Optimization",
        storage="sqlite:///db.sqlite3",
        load_if_exists=True)
    
    study.optimize(lambda trial: objective(trial, data, train_perf_eval, val_perf_eval), n_trials=200)
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)
    
#%% Validation runs

def run_multiple_experiments(model_class, data, train_mask, val_mask, test_mask,
                             criterion, params, num_epochs, num_runs=30):
    all_results = {"val_metrics": [],
                   "test_metrics": []
                   }

    device = data.x.device
    
    for run in range(num_runs):  # ensures reproducibility but still varied across runs
        print(f"Run {run + 1}/{num_runs}")
        model = model_class(
            num_node_features=data.num_features,
            num_classes=2,
            hidden_units=params["hidden_units"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        wrapper = ModelWrapper(model, optimizer, criterion)

        history, best_model_wts, best_f1 = train_and_validate(wrapper, data, train_mask, val_mask, num_epochs)
        # Evaluate on test set at the end
        test_loss, test_metrics = wrapper.evaluate(data, test_mask)

        all_results["val_metrics"].append(history["f1_illicit"][-1])
        all_results["test_metrics"].append(test_metrics)

    return all_results

#Moving variables to cuda device
train_perf_eval = train_perf_eval.to(data.x.device)
val_perf_eval = val_perf_eval.to(data.x.device)
test_perf_eval = test_perf_eval.to(data.x.device)

if validation_runs == True:
    all_results = run_multiple_experiments(model_class=GCN,
                                           data=data,
                                           train_mask=train_perf_eval,
                                           val_mask = val_perf_eval,
                                           test_mask= test_perf_eval,
                                           criterion=FocalLoss(gamma=2.5, alpha=0.5, reduction='mean'),
                                           params={"hidden_units": 128, "lr": 0.045},
                                           num_epochs=200,
                                           num_runs=30)
    
def summarize_and_visualize_results(all_results):
    """
    Summarizes and visualizes the results stored in all_results.
    Expects all_results to be a dict with keys 'val_metrics' and 'test_metrics'.
    """
    import matplotlib.pyplot as plt

    # Convert lists to numpy arrays for easier manipulation
    val_f1_scores = np.array(all_results["val_metrics"])
    test_metrics = all_results["test_metrics"]

    # Extract F1, Precision, Recall for test set
    test_f1 = np.array([m["f1_illicit"] for m in test_metrics])
    test_precision = np.array([m["precision_illicit"] for m in test_metrics])
    test_recall = np.array([m["recall_illicit"] for m in test_metrics])

    print("Validation F1: Mean = {:.4f}, Std = {:.4f}".format(val_f1_scores.mean(), val_f1_scores.std()))
    print("Test F1:        Mean = {:.4f}, Std = {:.4f}".format(test_f1.mean(), test_f1.std()))
    print("Test Precision: Mean = {:.4f}, Std = {:.4f}".format(test_precision.mean(), test_precision.std()))
    print("Test Recall:    Mean = {:.4f}, Std = {:.4f}".format(test_recall.mean(), test_recall.std()))

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.boxplot(val_f1_scores)
    plt.title("Validation F1")
    plt.ylabel("F1 Score")

    plt.subplot(1, 3, 2)
    plt.boxplot(test_f1)
    plt.title("Test F1")

    plt.subplot(1, 3, 3)
    plt.boxplot([test_precision, test_recall], labels=["Precision", "Recall"])
    plt.title("Test Precision & Recall")

    plt.tight_layout()
    plt.show()

    # Example usage after validation runs:
    # summarize_and_visualize_results(all_results)
    
#summarize_and_visualize_results(all_results)
# %%Final parameter optimisation and testing code

if Full_run == True:
    from Testing import run_optimization

    model_parameters, testing_results = run_optimization(
                                                models=['GCN', 'GAT', 'GIN', 'MLP', 'SVM', 'XGB', 'RF', 'XGBe+GIN', 'GINe+XGB'],   # Add or remove models as needed
                                                data=data,
                                                train_perf_eval=train_perf_eval,
                                                val_perf_eval=val_perf_eval,
                                                test_perf_eval=test_perf_eval,
                                                train_mask=train_mask,
                                                val_mask=val_mask
                                                )

#Save results from optimization
import pickle

def save_testing_results_pickle(results, path="testing_results.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
save_testing_results_pickle(testing_results, "testing_results.pkl")

#%% Loading in saved results
def load_testing_results_pickle(path="testing_results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
testing_results = load_testing_results_pickle("testing_results.pkl")
#%% Analysing results
df_long, df_summary, df_wide = produce_tables(testing_results)
#Boxplots by metric
boxplots_by_metric(df_long)
# Bar means with CI
bar_means_with_ci(df_summary, metric="f1_illicit")

# %% LSTM temporary pipeline

from LSTM_pipeline import run_lstm_embeddings_xgb

result = run_lstm_embeddings_xgb(
    features_df=features_df,
    data=data,
    train_perf_eval=train_perf_eval,
    val_perf_eval=val_perf_eval,
    test_perf_eval=test_perf_eval,
    T=49,
    lstm_hidden_dim=128,
    projection_dim=256,
    warmstart_epochs=20,          # set 0 to skip
    xgb_params={"tree_method": "hist"}  # use 'gpu_hist' if available
)

print("VAL:", result["val_metrics"])
print("TEST:", result["test_metrics"])
# embeddings available as result["embeddings"]