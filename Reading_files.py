import pandas as pd
def readfiles(pc):
    if pc =='Darwin':
        elliptic_txs_features = pd.read_csv('/Users/lambertusvanzyl/Library/CloudStorage/OneDrive-StellenboschUniversity/Masters/Code repository/Elliptic_dataset/elliptic_txs_features.csv', header=None)
        elliptic_txs_features.columns = ['txId']+['time_step'] + [f'V{i}' for i in range(1, 166)]
        elliptic_txs_classes  = pd.read_csv('/Users/lambertusvanzyl/Library/CloudStorage/OneDrive-StellenboschUniversity/Masters/Code repository/Elliptic_dataset/elliptic_txs_classes.csv')
        elliptic_txs_edgelist  = pd.read_csv('/Users/lambertusvanzyl/Library/CloudStorage/OneDrive-StellenboschUniversity/Masters/Code repository/Elliptic_dataset/elliptic_txs_edgelist.csv')
    else:
        elliptic_txs_features = pd.read_csv('/Users/Lambertus/OneDrive - Stellenbosch University/Masters/Code repository/Elliptic_dataset/elliptic_txs_features.csv', header=None)
        elliptic_txs_features.columns = ['txId']+['time_step'] + [f'V{i}' for i in range(1, 166)]
        elliptic_txs_classes  = pd.read_csv('/Users/Lambertus/OneDrive - Stellenbosch University/Masters/Code repository/Elliptic_dataset/elliptic_txs_classes.csv')
        elliptic_txs_edgelist  = pd.read_csv('/Users/Lambertus/OneDrive - Stellenbosch University/Masters/Code repository/Elliptic_dataset/elliptic_txs_edgelist.csv')
        
    
    return elliptic_txs_features, elliptic_txs_classes, elliptic_txs_edgelist