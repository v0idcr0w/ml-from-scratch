import numpy as np 
import gradient, loss, solver

class LinearRegression: 
  def __init__(self, solver='gd', loss='mse', learning_rate=0.1, num_iters: int = 1000, tol: float = 1e-7, verbose=False):
    self.learning_rate = learning_rate 
    self.solver = solver 
    self.loss = loss 
    self.num_iters = num_iters 
    self.verbose = verbose 
    self.tol = tol 
    
  def train(self, X_train: np.ndarray, y_train: np.ndarray):
    """Trains a linear regression model. 

    Args:
        X_train (np.ndarray): array containing features (columns) and observations (rows) 
        y_train (np.ndarray): array containing true labels 
        verbose (bool): prints out weights and bias every 10 iterations
    """
    # Initialize random weights based on the number of features 
    self.num_features = X_train.shape[1]
    self.weights = np.random.random(self.num_features)
    self.bias = np.random.random(1) 
    
    # Set gradients according to loss 
    if self.loss == 'mse':
      loss_fn = loss.MSE 
      grad_fn = gradient.mse_grad 
    
    if self.loss == 'mae':
      loss_fn = loss.MAE
      grad_fn = gradient.mae_grad
      
    if self.solver == 'gd': 
      self.weights, self.bias, train_history = solver.gradient_descent(X_train, y_train, self.weights, self.bias, pred_fn=self.predict, loss_fn=loss_fn, grad_fn=grad_fn, learning_rate=self.learning_rate, num_iters=self.num_iters, tol=self.tol, verbose=self.verbose)
      self.history = train_history 
  
  def predict(self, X: np.ndarray, weights=None, bias=None) -> np.ndarray:
    """Computes the current model predictions
    
    Args:
        X (np.ndarray): inputs to make the prediction on

    Returns:
        np.ndarray: array with the model prediction for each row in X
    """
    if weights is None:
      weights = self.weights
    if bias is None:
      bias = self.bias 
      
    return X @ weights.T + bias 

# Testing     
if __name__ == '__main__':   
  X_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]).reshape(-1,1)
  y_train = np.array([250, 300, 480,  430,   630, 730,])
  lr = LinearRegression(learning_rate=0.1, loss='mae', num_iters=1000, verbose=True)
  lr.train(X_train, y_train)
  print(lr.predict(np.array([1.0, 1.2, 2.0]).reshape(-1,1)))
