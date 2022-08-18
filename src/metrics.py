import numpy as np

def rse(y_true, y_pred):
  return np.sum(np.square(np.subtract(y_true, y_pred)))                        \
    / np.sum(np.square(np.subtract(y_true, np.mean(y_true, axis=0))))
