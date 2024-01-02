
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import joblib
def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)

def REP(pred, true):
    """Compute the mean relative percentage error between prediction and true values."""
    # Avoid division by zero by adding a small constant
    epsilon = 1e-10
    scaler = joblib.load('./data/target_scaler.pkl')
    #转换为np
    pred = pred.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(true)
    relative_errors = np.abs((pred - true) / (true + epsilon))* 100
    return np.mean(relative_errors)