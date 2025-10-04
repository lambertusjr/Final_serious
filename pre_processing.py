def elliptic_pre_processing(features_df, classes_df, edgelist_df):
    #Geting known nodes
    known_nodes = classes_df != 'unknown'
    known_nodes = known_nodes['class'].values
    #Remap class labels
    class_mapping = {
        'unknown': -1,
        '1': 0,
        '2': 1
    }
    classes_df['class'] = classes_df['class'].map(class_mapping)
    classes_df['class'] = classes_df['class'].astype(int) 
    
    for new_id, old_id in enumerate(features_df['txId']):
        class_mapping[old_id] = new_id
    
    classes_df['txId'] = classes_df['txId'].map(class_mapping)
    edgelist_df['txId1'] = edgelist_df['txId1'].map(class_mapping)
    edgelist_df['txId2'] = edgelist_df['txId2'].map(class_mapping) 
    
    return features_df, classes_df, edgelist_df, known_nodes
import torch
from torch_geometric.data import Data
def create_data_object(features_df, classes_df, edgelist_df):
    #Converting dataframes to tensors
    features_tensor = torch.tensor(features_df.drop(columns=['txId']).values, dtype=torch.float)
    edgelist_tensor = torch.tensor(edgelist_df.values.T, dtype=torch.long)
    classes_tensor = torch.tensor(classes_df['class'].values, dtype=torch.long)
    
    data = Data(x = features_tensor, edge_index=edgelist_tensor, y=classes_tensor)
    
    return data

def create_elliptic_masks(data):
    #Getting unique time steps
    time_steps = data.x[:, 0].unique()
    #Creating masks depending on time steps
    #Training 0-30
    #Validation 31-40
    #Testing 41-49
    train_mask = (data.x[:, 0] <= 30) & (data.x[:, 0] >= 1)
    val_mask = (data.x[:, 0] <= 40) & (data.x[:, 0] >= 31)
    test_mask = (data.x[:, 0] <= 49) & (data.x[:, 0] >= 41)
    
    #Creating performance evaluation masks for each data split
    
    #Training mask
    train_perf_eval = train_mask & (data.y != -1)
    #Validation mask
    val_perf_eval = val_mask & (data.y != -1)
    #Testing mask
    test_perf_eval = test_mask & (data.y != -1)
    
    return train_mask, val_mask, test_mask, train_perf_eval, val_perf_eval, test_perf_eval

def make_ibm_masks(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    torch.manual_seed(seed)
    idx = torch.randperm(data.num_nodes, device=data.x.device)
    n_train = int(train_ratio * len(idx))
    n_val = int(val_ratio * len(idx))

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=idx.device)
    val_mask   = train_mask.clone()
    test_mask  = train_mask.clone()

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    known = data.y != -1  # keep convention if you label unknowns as -1
    return (
        train_mask,
        val_mask,
        test_mask,
        train_mask & known,
        val_mask & known,
        test_mask & known,
    )
