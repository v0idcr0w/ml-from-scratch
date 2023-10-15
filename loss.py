import numpy as np
from collections.abc import Callable

def MSE(y: np.ndarray, yhat: np.ndarray) -> float: 
    """Computes the mean squared error loss. 

    Args:
        y (np.ndarray): true values 
        yhat (np.ndarray): predictions 
    """ 
    return np.sum((y - yhat)**2) / len(y) 
  
def MAE(y: np.ndarray, yhat: np.ndarray) -> float: 
    """Computes the mean absolute error loss. 

    Args:
        y (np.ndarray): true values 
        yhat (np.ndarray): predictions 
    """
    return np.sum( np.abs(y - yhat) ) / len(y)