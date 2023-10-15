import numpy as np
from collections.abc import Callable

def gradient_descent(X: np.ndarray, y: np.ndarray, w_in: np.ndarray | float, b_in: float, pred_fn: Callable, loss_fn: Callable, grad_fn: Callable, learning_rate: float, num_iters: int = 1000, tol: float = 1e-7, verbose=False) -> tuple[np.ndarray, float, list]: 
    
    w, b = w_in, b_in
    loss_history = []
   
    for i in range(num_iters): 
      # current predictions with the current weights 
      y_pred = pred_fn(X, w, b)
      # compute loss
      loss = loss_fn(y, y_pred)
      # compute gradients 
      dL_dw, dL_db = grad_fn(X, y, y_pred)
      
      # update parameters 
      w = w - learning_rate * dL_dw 
      b = b - learning_rate * dL_db 
      
      loss_history.append(loss)
        
      if i % 10 == 0 and verbose:
        print(f"Iteration {i}:\tLoss = {loss_history[-1]:.2f}")
        print(f"Weights: {w}")
        print(f"Bias: {b}")

      if i > 2 and abs(loss_history[-1] - loss_history[-2])/loss_history[-2] < tol: 
        break 
        
    
    return w, b, loss_history