# Final restart of my code
#notes:

#%% Setup
seeded_run = True
parameter_tuning = False
num_epochs = 200
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

#%% Getting data ready for models
features_df, classes_df, edgelist_df = readfiles(pc)
features_df, classes_df, edgelist_df, known_nodes = elliptic_pre_processing(features_df, classes_df, edgelist_df)
data = create_data_object(features_df, edgelist_df, classes_df)
data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

train_mask, val_mask, test_mask, train_perf_eval, val_perf_eval, test_perf_eval = create_elliptic_masks(data)
# %%
