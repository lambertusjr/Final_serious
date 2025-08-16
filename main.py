# Final restart of my code
#notes:


seeded_run = True
prototyping = True
parameter_tuning = False
num_epochs = 200
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
from models import GCN, ModelWrapper
from training_functions import train_and_validate
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    data = data.to(device)
    model_wrapper = ModelWrapper(model, optimizer, criterion)
    #Beginning training
    metrics, best_model_wts = train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs)
    
# %%
