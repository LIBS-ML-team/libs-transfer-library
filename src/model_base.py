import os
from typing import Union, Optional
import datetime

import numpy as np
import pandas as pd
import dill as pickle
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import keras

KERAS_MODELS = [tf.keras.Model, keras.Model, tf.estimator.Estimator]

class SaveMixin:
    
  def save(self, path: Union[str, Path]) -> None:
    if not isinstance(path, Path):
        path = Path(path)
    if not (path / "params").exists():
        os.makedirs(path / "params")
    # save keras models
    keras_attrs = {}
    for attr in vars(self).keys():
      if any((isinstance(getattr(self, attr), keras_model) for keras_model in KERAS_MODELS)):
        keras_attrs[attr] = getattr(self, attr)
        os.makedirs(path / attr)
        getattr(self, attr).save(path / attr)
        setattr(self, attr, None)
    # TODO?
    self.history_ = None
    self.keras_attrs = list(keras_attrs.keys())
    pickle.dump(self, open(path / "params" / "params.pkl", 'wb'))
    delattr(self, 'keras_attrs')
    # add back keras models
    for key, val in keras_attrs.items():
      setattr(self, attr, val)
        
        

class NNClassifier(BaseEstimator, ClassifierMixin, SaveMixin):

    def __init__(self, tensorboard: Optional[str] = None):
      self.tensorboard = tensorboard
      

    def fit(self, X, y=None, X_val=None, y_val=None, n_iter=50, verbose=0, callbacks=[], batch_size=None):
      tf.keras.backend.clear_session()
      # Check that X and y have correct shape
      X, y = check_X_y(X, y)
      if X_val is not None:
          X_val, y_val = check_X_y(X_val, y_val)
      # Store the classes seen during fit
      self.classes_ = np.unique(y)
      # Store number of features
      self.n_features_in_ = X.shape[1]
      self.n_features_out_ = y.shape[1]

      # one hot encoding
      y = OneHotEncoder(categories=[self.classes_], sparse=False).fit_transform(y.reshape(-1, 1))
      if y_val is not None:
        y_val = OneHotEncoder(categories=[self.classes_], sparse=False).fit_transform(y_val.reshape(-1, 1))

      # Initialize architecture
      self.build()

      if verbose > 0:
        print(self.model_.summary())

      # Add TensorBoard
      if self.tensorboard is not None:
        self.log_dir_ = './tb_logs/' + self.tensorboard
        callbacks += [TensorBoard(log_dir=self.log_dir_, histogram_freq=1)]

      # Train the model
      if X_val is None or y_val is None:
        history_obj = self.model_.fit(X, y, epochs=n_iter, verbose=verbose, callbacks=callbacks, batch_size=batch_size)
      else:
        history_obj = self.model_.fit(X, y, epochs=n_iter, verbose=verbose, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=batch_size)
        
      # Set history
      self.history_ = history_obj
      return self
      

    def build(self):
      raise NotImplemented
      

    def predict(self, X):
      check_is_fitted(self)

      X = check_array(X)

      return self.classes_[self.model_.predict(X).argmax(axis=1)]
      
      
class NNRegressor(BaseEstimator, RegressorMixin, SaveMixin):

    def __init__(self, tensorboard: Optional[str] = None):
      self.tensorboard = tensorboard
      

    def fit(self, X, y=None, X_val=None, y_val=None, n_iter=50, verbose=0, callbacks=[], batch_size=None):
      tf.keras.backend.clear_session()
      X = check_array(X)
      y = check_array(y)
      # Store number of features
      self.n_features_in_ = X.shape[1]
      self.n_features_out_ = y.shape[1]

      # Initialize architecture
      self.build()

      if verbose > 0:
        print(self.model_.summary())

      # Add TensorBoard
      if self.tensorboard:
        self.log_dir_ = './tb_logs/' + self.tensorboard
        callbacks += [TensorBoard(log_dir=self.log_dir_, histogram_freq=1)]

      # Train the model
      if X_val is None or y_val is None:
        history_obj = self.model_.fit(X, y, epochs=n_iter, verbose=verbose, callbacks=callbacks, batch_size=batch_size)
      else:
        history_obj = self.model_.fit(X, y, epochs=n_iter, verbose=verbose, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=batch_size)
        
      # Set history
      self.history_ = history_obj
      return self
      

    def build(self):
      raise NotImplemented
      

    def predict(self, X):
      check_is_fitted(self)

      X = check_array(X)

      return self.model_.predict(X)
      
      
      
def load_model(path: Union[str, Path], *args) -> Union[NNClassifier, NNRegressor]:
    if not isinstance(path, Path):
        path = Path(path)
    model = pickle.load(open(path / "params" / "params.pkl", 'rb'))
    for attr in model.keras_attrs:
      setattr(model, attr, tf_load_model(path / attr))
    delattr(model, 'keras_attrs')
    return model
