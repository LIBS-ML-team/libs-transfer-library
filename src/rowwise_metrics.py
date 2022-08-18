import numpy as np

def rowwise_cosine(y_true, y_pred):
  """
  https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
  """
  return - np.einsum('ij,ij->i', y_true, y_pred) / (
              np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    )
  
def rowwise_mse(y_true, y_pred):
  return np.square(np.subtract(y_true, y_pred)).mean(1)

def rowwise_rmse(y_true, y_pred):
  return np.sqrt(rowwise_mse(y_true, y_pred))
  
def rowwise_se(y_true, y_pred):
  return np.sum(np.square(np.subtract(y_true, y_pred)), axis=1)
  
def rowwise_euclid(y_true, y_pred):
  return np.sqrt(rowwise_se(y_true, y_pred))
