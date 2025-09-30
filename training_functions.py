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
from xgboost import XGBClassifier
from Helper_functions import calculate_metrics
from models import ModelWrapper, MLP

def train_and_validate(model_wrapper, data, train_perf_eval, val_perf_eval, num_epochs, best_f1 = -1, best_f1_model_wts = None):
    # Device alignment guard (fail fast with clear message)
    mdl_dev = next(model_wrapper.model.parameters()).device
    if not (data.x.device == mdl_dev and train_perf_eval.device == mdl_dev and val_perf_eval.device == mdl_dev):
        raise RuntimeError(
            f"Device mismatch: model={mdl_dev}, data.x={data.x.device}, "
            f"train_mask={train_perf_eval.device}, val_mask={val_perf_eval.device}"
        )

    
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

        best_f1, best_f1_model_wts = update_best_weights(model_wrapper.model, best_f1, val_metrics['f1_illicit'], best_f1_model_wts)
    return metrics, best_f1_model_wts, best_f1
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

def train_and_test_NMW_models(model_name, data, train_perf_eval, val_perf_eval, test_perf_eval, params_for_model):
    match model_name:
        case "SVM":
            C = params_for_model.get("C", 1.0)
            degree = params_for_model.get("degree", 3)
            kernel = params_for_model.get("kernel", 'rbf')
            svm_model = SVC(kernel=kernel, C=C, class_weight='balanced', degree=degree)
            train_perf_eval = train_perf_eval | val_perf_eval #Combining to increase available training data
            svm_model.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
            pred = svm_model.predict(data.x[test_perf_eval].cpu().numpy())
            metrics = calculate_metrics(data.y[test_perf_eval].cpu().numpy(), pred)
            return metrics
        case "RF":
            n_estimators = params_for_model.get("n_estimators", 100)
            max_depth = params_for_model.get("max_depth", 10)
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')
            train_perf_eval = train_perf_eval | val_perf_eval #Combining to increase available training data
            rf_model.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
            pred = rf_model.predict(data.x[test_perf_eval].cpu().numpy())
            metrics = calculate_metrics(data.y[test_perf_eval].cpu().numpy(), pred)
            return metrics
        case "XGB":
            from xgboost import XGBClassifier
            max_depth = params_for_model.get("max_depth", 10)
            n_estimators = params_for_model.get("n_estimators", 100)
            xgb_model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, scale_pos_weight=9.25)
            train_perf_eval = train_perf_eval | val_perf_eval #Combining to increase available training data
            xgb_model.fit(data.x[train_perf_eval].cpu().numpy(), data.y[train_perf_eval].cpu().numpy())
            pred = xgb_model.predict(data.x[test_perf_eval].cpu().numpy())
            metrics = calculate_metrics(data.y[test_perf_eval].cpu().numpy(), pred)
            return metrics
from torch_geometric.data import Data

#GIN encoder with XGB classifier model
def train_and_test_GINeXGB(data: Data, train_perf_eval, val_perf_eval, test_perf_eval, params_for_model):
    from encoding import pre_train_GIN_encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    train_perf_eval = train_perf_eval.to(device)
    val_perf_eval = val_perf_eval.to(device)
    test_perf_eval = test_perf_eval.to(device)
    
    #Extracting hyperparameters for GIN encoder pre-training from optuna trials
    hidden_units = params_for_model.get("hidden_units", 64)
    lr_encoder = params_for_model.get("gin_lr", 0.01)
    wd_encoder = params_for_model.get("gin_weight_decay", 0.0001)
    lr_head = params_for_model.get("xgb_lr", 0.1)
    wd_head = params_for_model.get("xgb_weight_decay", 0.0)
    epochs = params_for_model.get("epochs", 100)
    embedding_dim = params_for_model.get("embedding_dim", 128)
    
    #initialize and pre-train GIN encoder
    encoder, best_encoder_state = pre_train_GIN_encoder(
        data=data,
        train_perf_eval = train_perf_eval,
        val_perf_eval = val_perf_eval,
        hidden_units=hidden_units,
        lr_encoder=lr_encoder,
        wd_encoder=wd_encoder,
        lr_head=lr_head,
        wd_head=wd_head,
        epochs=epochs,
        embedding_dim=embedding_dim
    )
    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder.embed(data).to(device)
        
    #Applying mask to embeddings and labels
    x_train = embeddings[train_perf_eval | val_perf_eval].cpu().numpy() #Combining to increase available training data
    y_train = data.y[train_perf_eval | val_perf_eval].cpu().numpy()
    x_test = embeddings[test_perf_eval].cpu().numpy()
    y_test = data.y[test_perf_eval].cpu().numpy()
    
    #Getting XGBoost classifier hyperparameters
    n_estimators = params_for_model.get("xgb_n_estimators", 300)
    max_depth = params_for_model.get("xgb_max_depth", 6)
    learning_rate_XGB = params_for_model.get("xgb_learning_rate", 0.05)
    subsample = params_for_model.get("xgb_subsample", 0.8)
    colsample_bt = params_for_model.get("xgb_colsample_bytree", 0.8)
    reg_lambda = params_for_model.get("xgb_reg_lambda", 1.0)
    reg_alpha = params_for_model.get("xgb_reg_alpha", 0.0)
    #Fit XGBoost model
    xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=9.25,
            learning_rate=learning_rate_XGB,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bt,
            subsample=subsample,
            tree_method = 'hist',
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            device=("cuda" if torch.cuda.is_available() else "cpu")
            )
    #Fit model to embedding data
    xgb_model.fit(x_train, y_train)
    
    #Generate predictions
    test_pred = xgb_model.predict(x_test)
    test_metrics = calculate_metrics(y_test, test_pred)
    return test_metrics
