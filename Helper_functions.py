from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def calculate_metrics(y_true, y_pred):
    precision_score_weighted = precision_score(y_true, y_pred, average='weighted')
    precision_score_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary')
    
    recall_score_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_score_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary')
    
    f1_score_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_score_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary')
    
    metrics = {
        'precision_weighted': precision_score_weighted,
        'precision_illicit': precision_score_illicit,
        'recall_weighted': recall_score_weighted,
        'recall_illicit': recall_score_illicit,
        'f1_weighted': f1_score_weighted,
        'f1_illicit': f1_score_illicit
    }
    return metrics