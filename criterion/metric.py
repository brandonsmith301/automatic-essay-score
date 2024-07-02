import numpy as np
from sklearn.metrics import cohen_kappa_score

def get_score(y_trues, y_preds):
    y_true = y_trues.flatten()
    y_pred = np.round(y_preds.flatten()) 
    
    return cohen_kappa_score(
        y_true, 
        y_pred, 
        weights='quadratic'
    )