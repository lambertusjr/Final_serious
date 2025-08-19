"""
What I need in the generic training function:
best_f1 = -1
best_f1_model_wts = None

"""
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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

def train_svm(data, train_mask, val_mask, kernel="rbf", C=1.0):
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask.cpu().numpy()], y[train_mask.cpu().numpy()]
    X_val, y_val = X[val_mask.cpu().numpy()], y[val_mask.cpu().numpy()]

    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    return model, val_score

def train_random_forest(data, train_mask, val_mask, n_estimators=100, max_depth=None):
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask.cpu().numpy()], y[train_mask.cpu().numpy()]
    X_val, y_val = X[val_mask.cpu().numpy()], y[val_mask.cpu().numpy()]

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    return model, val_score

def train_decision_tree(data, train_mask, val_mask, max_depth=None):
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask.cpu().numpy()], y[train_mask.cpu().numpy()]
    X_val, y_val = X[val_mask.cpu().numpy()], y[val_mask.cpu().numpy()]

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    return model, val_score

def train_logistic_regression(data, train_mask, val_mask):
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    X_train, y_train = X[train_mask.cpu().numpy()], y[train_mask.cpu().numpy()]
    X_val, y_val = X[val_mask.cpu().numpy()], y[val_mask.cpu().numpy()]

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    val_score = model.score(X_val, y_val)
    return model, val_score