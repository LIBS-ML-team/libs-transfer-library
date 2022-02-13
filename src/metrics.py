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
  

def make_test_scorer(metric, X_test, y_test):
  def test_scorer(estimator, *args, **kwargs):
    return metric(y_test, estimator.predict(X_test))
  return test_scorer
