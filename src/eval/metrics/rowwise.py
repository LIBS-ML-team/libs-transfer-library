import numpy as np

def rowwise_cosine(y_true, y_pred):
  """
  https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
  """
  return 1 - np.einsum('ij,ij->i', y_true, y_pred) / (
              np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    )
  
def rowwise_mse(y_true, y_pred):
  return np.square(np.subtract(y_true, y_pred)).mean(1)

def rowwise_rmse(y_true, y_pred):
  return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean(1))
