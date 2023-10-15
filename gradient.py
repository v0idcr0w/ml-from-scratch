import numpy as np

def mse_grad(X: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]: 
    """
    Computes the gradient of the MSE loss function with respect to model parameters  
    Args:
      X (ndarray): input matrix with N observations  
      y (ndarray): true labels 
      yhat (ndarray): current model predictions 
    Returns:
      dL_dw (ndarray): gradient with respect to the weights 
      dL_db (scalar): gradient with respect to the bias 
     """
    N = X.shape[0]
    
    diff = yhat - y 
    dL_dw = 2/N * diff @ X 
    dL_db = 2/N * np.sum( diff ) 
    
    return dL_dw, dL_db  
  
  
  
def mae_grad(X: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]: 
    """
    Computes the gradient of the MAE loss function with respect to model parameters  
    Args:
      X (ndarray): input matrix with N observations (rows) and features
      y (ndarray): true labels 
      yhat (ndarray): current model predictions 
    Returns:
      dL_dw (ndarray): gradient with respect to the weights 
      dL_db (scalar): gradient with respect to the bias 
     """
    N = X.shape[0]
    
    diff = y - yhat
    sign_switch = np.zeros(N)
    sign_switch[diff>0] = -1 
    sign_switch[diff<=0] = 1 
    
    dL_dw = 1/N * sign_switch @ X
    dL_db = 1/N * np.sum( sign_switch ) 
    
    return dL_dw, dL_db  