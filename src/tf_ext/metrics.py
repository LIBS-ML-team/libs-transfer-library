import tensorflow as tf

def tensor_sparsity(vector):
  return tf.divide(
      tf.subtract(
          tf.sqrt(
              tf.cast(
                  tf.size(vector), float)),
                  tf.scalar_mul(
                      tf.math.reciprocal(tf.norm(vector)),
                      tf.reduce_sum(tf.abs(vector))
                      )
                  ),
                   tf.subtract(tf.sqrt(tf.cast(tf.size(vector), float)), 1)
                   )
  

def make_val_scorer(metric, X_test, y_test):
  def test_scorer(estimator, *args, **kwargs):
    return metric(y_test, estimator.predict(X_test))
  return test_scorer

def make_baselined_scorer_from_loss(loss, baseline_loss):
  def score(y_true, y_pred):
      return 1 - loss(y_true, y_pred) / baseline_loss
  return score
