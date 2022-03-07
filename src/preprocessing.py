import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter
from sklearn.utils import resample
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.pipeline import Pipeline
from scipy.integrate import trapz


class LabelCropp(BaseEstimator, TransformerMixin):
  """
  TODO add nonnumeric labels
  """

  def __init__(self, label_from=None, label_to=None, labels=None):
    # TODO default values!!!
    self.label_from = label_from
    self.label_to = label_to
    self.labels = labels


  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self


  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    if self.labels is None:
      labels = np.arange(self.n_features_in_)
    else:
      labels = self.labels
      
    if not np.issubdtype(labels.dtype, np.number):
      raise ValueError('Only implemented for numeric values')
    if self.label_from > np.maximum(labels) or self.label_to < np.minimum(labels):
      warnings.warn('Labels out of range! Skipping!')
      return X

    idx_from, idx_to = np.argmax(labels >= self.label_from),  len(labels) - np.argmax((labels <= self.label_to)[::-1]) - 1
    return X[:, idx_from: idx_to]
    
    
    
class Cover(BaseEstimator, TransformerMixin):
  """
  """

  def __init__(self, label_from=None, label_to=None, labels=None):
    self.label_from = label_from
    self.label_to = label_to
    self.labels = labels


  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self


  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    if self.labels is None:
      labels = np.arange(self.n_features_in_)
    else:
      labels = self.labels
      
    if not np.issubdtype(labels.dtype, np.number):
      raise ValueError('Only implemented for numeric values')
    if self.label_from > np.maximum(labels) or self.label_to < np.minimum(labels):
      warnings.warn('Labels out of range! Skipping!')
      return X

    idx_from, idx_to = np.argmax(labels >= self.label_from),  len(labels) - np.argmax((labels <= self.label_to)[::-1]) - 1
    X_new = np.array(X, copy=True)
    X_new[:, idx_from: idx_to] = 0
    return X_new




class Cropp(BaseEstimator, TransformerMixin):

  def __init__(self, idx_from=2500, idx_to=33500):
    self.idx_from = idx_from
    self.idx_to = idx_to

  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self

  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    return X[:, self.idx_from: self.idx_to]
    
    

class DownSampler(BaseEstimator, TransformerMixin):
  def __init__(self, samples_to_combine=3, strategy='mean'):
    self.N = samples_to_combine
    self.strategy = strategy
    

  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self
    

  def transform(self, X, *args, **kwargs):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    if self.strategy == 'mean':
      # adds additional value for X % N == 0
      return np.pad(X, [(0, 0), (0, self.N - X.shape[1] % self.N if X.shape[1] % self.N else 0)], mode='edge').reshape((X.shape[0], X.shape[1] // self.N + int(X.shape[1] % self.N > 0), self.N)).mean(axis=2)
    else:
      raise NotImplementedError('Strategy not found!')
      
      

class SavGol(BaseEstimator, TransformerMixin):
  def __init__(self, window_length=5, polyorder=3, deriv=2, mode='constant', delta=3):
    self.window_length = window_length
    self.polyorder = polyorder
    self.deriv = deriv
    self.mode = mode
    self.delta = delta
    

  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self
    

  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    f = lambda x: savgol_filter(
        x,
        window_length=self.window_length,
        polyorder=self.polyorder,
        deriv=self.deriv,
        mode=self.mode,
        delta=self.delta
        )
    return np.apply_along_axis(f, 1, X)
    
    

class Normalizer(BaseEstimator, TransformerMixin):
  def __init__(self, norm='max'):
    self.norm = norm
    
  
  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self
    

  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    return normalize(X, norm=self.norm)
    
    

class ERFReducer(BaseEstimator, TransformerMixin):
  def __init__(self, n_estimators=256, random_state=42):
    self.n_estimators = n_estimators
    self.random_state = random_state
    
  
  def fit(self, X, y):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    clf = ExtraTreesClassifier(n_estimators = self.n_estimators, n_jobs=-1, random_state=self.random_state)
    selector = SelectFromModel(clf, prefit=False).fit(X, y)
    self.filter_ = pd.Series(selector.get_support(), dtype=bool)
    return self
    

  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    return X[:, self.filter_.values[:X.shape[1]]]
    
    
    
class Integrate(BaseEstimator, TransformerMixin):
  def __init__(self, func=trapz, *args, **kwargs):
    self.func = func
    self.args = args
    self.kwargs = kwargs

  def fit(self, X):
    return self

  def transform(self, X):
    return pd.DataFrame(self.func(X, *self.args, **self.kwargs))
    
    
    
def match_wavelengths(original: pd.DataFrame, other: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """
  Match the dimension of each spectra in the <other> dataset to the <original>.
  Both datasets need to have column names set to the corresponding wavelengths.
  Finds the intersection, interpolates the <other> dataset to match, fills rest with zeros.
  """
  if len(original.columns) < len(other.columns):
      print('[WARNING] <original> has smaller dimension than <other>!')
  print('[WARNING] Experimental!')
  # find the intersection
  try:
    start = np.where(original.columns <= other.columns[0])[0][-1]
  except:
    start = 0
  try:
    end = np.where(original.columns >= other.columns[-1])[0][0]
  except:
    end = -1
  intersection = original.columns[start:end]
  # interpolate <other> to match
  new = other.apply(lambda x: np.interp(intersection, xp=other.columns, fp=x), axis=1, result_type="expand").values
  # fill with zeros
  new = pd.DataFrame(np.pad(new, ((0, 0), (start, len(original.columns) - end)), 'constant', constant_values=0).astype(np.float32))
  new.columns = original.columns
  return original, new
