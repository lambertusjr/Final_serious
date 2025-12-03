# Final restart of my code
#notes:


seeded_run = False
prototyping = False
MLP_prototype = True
svm_prototype = True
XGB_prototype = True
RF_prototype = True
parameter_tuning = False
validation_runs = False
elliptic_dataset = False
IBM_dataset = True
#Select IBM dataset type/size
dataset_type_size = 'HISMALL'  # Options: 'HISMALL', 'HIMEDIUM', 'LISMALL', 'LIMEDIUM'
Full_run = True
delete_existing_studies = True
num_epochs = 200
early_stop_patience = 60
early_stop_min_delta = 1e-4
early_stop_logging = True


#%% Quick functions
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
#import seaborn as sns
#import matplotlib.pyplot as plt

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
#import torch_geometric_temporal
#from torch_geometric_temporal.nn.recurrent import EvolveGCNH, EvolveGCNO

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
from Helper_functions import FocalLoss, calculate_metrics, balanced_class_weights
from analysis import results_to_long_df, summarise_long_df, formatted_wide_table, produce_tables, boxplots_by_metric, bar_means_with_ci
from encoding import DGTEncoder
#%% Getting data ready for models
from pre_processing import make_ibm_masks
if elliptic_dataset == True:
    features_df, classes_df, edgelist_df = readfiles(pc)
    features_df, classes_df, edgelist_df, known_nodes = elliptic_pre_processing(features_df, classes_df, edgelist_df)
    data = create_data_object(features_df, classes_df, edgelist_df)
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

    train_mask, val_mask, test_mask, train_perf_eval, val_perf_eval, test_perf_eval = create_elliptic_masks(data)
    
if IBM_dataset == True:
    from stock_IBM_process import AMLtoGraph
    if pc =='Darwin':
        dataset = AMLtoGraph(root='/Users/lambertusvanzyl/Desktop/Final_serious/data', dataset_type_size=dataset_type_size)
        data: Data = dataset[0]
    else:
        dataset = AMLtoGraph(root='/Users/Lambertus/Desktop/Final_serious/data', dataset_type_size=dataset_type_size)
        data: Data = dataset[0]
    data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    data.y = data.y.long()
    
    train_mask, val_mask, test_mask, train_perf_eval, val_perf_eval, test_perf_eval = make_ibm_masks(data)
    
#%% Feature engineering


# print_tensor_info(
#     train_mask=train_mask,
#     val_mask=val_mask,
#     test_mask=test_mask,
#     train_perf_eval=train_perf_eval,
#     val_perf_eval=val_perf_eval,
#     test_perf_eval=test_perf_eval
# )



#summarize_and_visualize_results(all_results)
# %%Final parameter optimisation and testing code
#variable for dataset used in optimization
if elliptic_dataset == True:
    data_for_optimization = 'elliptic'
elif IBM_dataset == True:
    if dataset_type_size == 'HISMALL':
        data_for_optimization = 'IBM_HISMALL'
    elif dataset_type_size == 'HIMEDIUM':
        data_for_optimization = 'IBM_HIMEDIUM'
    elif dataset_type_size == 'LISMALL':
        data_for_optimization = 'IBM_LISMALL'
    elif dataset_type_size == 'LIMEDIUM':
        data_for_optimization = 'IBM_LIMEDIUM'
        
        
if Full_run == True:
    from Testing import run_optimization

    model_parameters, testing_results = run_optimization(
                                                models=['GAT', 'GIN' ],# 'XGB', 'RF','GCN', 'GAT', 'GIN', 'XGBe+GIN', 'GINe+XGB'    # Add or remove models as needed
                                                data=data,
                                                train_perf_eval=train_perf_eval,
                                                val_perf_eval=val_perf_eval,
                                                test_perf_eval=test_perf_eval,
                                                train_mask=train_mask,
                                                val_mask=val_mask,
                                                data_for_optimization=data_for_optimization,
                                                delete_existing_studies=delete_existing_studies
                                                )

#Save results from optimization
import pickle

def save_testing_results_pickle(results, path=f"testing_results_{data_for_optimization}.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
save_testing_results_pickle(testing_results, "testing_results.pkl")

def save_testing_results_csv(results, path=f"{data_for_optimization}_testing_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(f"csv_results/{data_for_optimization}_testing_results.csv", index=False)

save_testing_results_csv(testing_results)

#%% Loading in saved results
import pickle
def load_testing_results_pickle(path=f"{data_for_optimization}_testing_results.pkl"):
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
"""
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
    xgb_params={"tree_method": "hist"},  # use 'gpu_hist' if available
    warmstart_patience=5,
    warmstart_min_delta=1e-3,
    warmstart_log_early_stop=early_stop_logging
)

print("VAL:", result["val_metrics"])
print("TEST:", result["test_metrics"])
# embeddings available as result["embeddings"]
"""

#%% Testing if DGT works
DGT_prototype = True

if DGT_prototype:
    dgt_device = torch.device("cpu")

    class DGTClassifier(nn.Module):
        def __init__(
            self,
            num_node_features: int,
            num_classes: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_layers: int = 3,
            heads: int = 4,
            dropout: float = 0.2,
            spd_K: int = 4
        ):
            super().__init__()
            self.encoder = DGTEncoder(
                num_node_features=num_node_features,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                spd_K=spd_K
            ).to(dgt_device)
            self.head = nn.Linear(embedding_dim, num_classes).to(dgt_device)

        def forward(self, graph: Data) -> torch.Tensor:
            z = self.encoder(graph)
            return self.head(z)

    dgt_data = Data(
        x=data.x.detach().cpu(),
        edge_index=data.edge_index.detach().cpu(),
        y=data.y.detach().cpu(),
        num_nodes=data.num_nodes
    )

    dgt_train_mask = train_perf_eval.detach().cpu()
    dgt_val_mask = val_perf_eval.detach().cpu()
    dgt_test_mask = test_perf_eval.detach().cpu()

    dgt_alpha = balanced_class_weights(dgt_data.y[dgt_train_mask])
    dgt_model = DGTClassifier(num_node_features=dgt_data.num_features, num_classes=2).to(dgt_device)
    dgt_optimizer = torch.optim.Adam(dgt_model.parameters(), lr=5e-3, weight_decay=5e-4)
    dgt_criterion = FocalLoss(alpha=dgt_alpha, gamma=2.5, reduction='mean')
    dgt_wrapper = ModelWrapper(dgt_model, dgt_optimizer, dgt_criterion, use_amp=False)

    dgt_metrics, dgt_best_wts, dgt_best_f1 = train_and_validate(
        dgt_wrapper,
        dgt_data,
        dgt_train_mask,
        dgt_val_mask,
        num_epochs,
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        log_early_stop=early_stop_logging
    )

    if dgt_best_wts is not None:
        dgt_wrapper.model.load_state_dict(dgt_best_wts)

    _, dgt_test_metrics = dgt_wrapper.evaluate(dgt_data, dgt_test_mask)

    dgt_last_val_f1 = float('nan')
    if dgt_metrics['f1_illicit']:
        dgt_last_val_f1 = dgt_metrics['f1_illicit'][-1]

    print(f"DGT validation F1 (last epoch): {dgt_last_val_f1:.4f}")
    print(f"DGT best validation F1: {dgt_best_f1:.4f}")
    print(f"DGT test metrics: {dgt_test_metrics}")

    with torch.no_grad():
        dgt_embeddings = dgt_wrapper.model.encoder(dgt_data)

    dgt_embeddings = dgt_embeddings.to(data.x.device)
    dgt_embedded_data = Data(
        x=dgt_embeddings,
        edge_index=data.edge_index,
        y=data.y,
        num_nodes=data.num_nodes
    ).to(data.x.device)

    dgt_embedded_train_mask = dgt_train_mask.to(data.x.device)
    dgt_embedded_val_mask = dgt_val_mask.to(data.x.device)
    dgt_embedded_test_mask = dgt_test_mask.to(data.x.device)

    dgt_embedded_data.train_mask = dgt_embedded_train_mask
    dgt_embedded_data.val_mask = dgt_embedded_val_mask
    dgt_embedded_data.test_mask = dgt_embedded_test_mask

    dgt_prototype_results = {
        "history": dgt_metrics,
        "best_val_f1": dgt_best_f1,
        "test_metrics": dgt_test_metrics,
        "embedded_data": dgt_embedded_data
    }

# %%EvolveGCN temporary pipeline
