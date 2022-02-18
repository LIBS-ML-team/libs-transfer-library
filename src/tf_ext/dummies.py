from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

class Average_Dummy(BaseEstimator, RegressorMixin):

  def __init__(self, *args, **kwargs):
      pass
    

  def fit(self, X, y, *args, **kwargs):
    X = check_array(X)
    y = check_array(y)
    self.prediction_ = np.mean(X, axis=0)
    return self
    

  def predict(self, X, *args, **kwargs):
    check_array(X)
    return np.tile(baseline.prediction_, (s1.shape[0], 1))
