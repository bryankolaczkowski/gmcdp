from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def lrsact(x, alpha=0.4):
  """
  2-sided 'leaky-rectified' linear activation
  scales x by alpha*x whenever |x| > (1-alpha)
  """
  v  = 1.0 - alpha
  b  = v * v
  # leaky-rectify positive values
  c = tf.math.greater(x, v)
  r = tf.where(c, alpha*x+b, x)
  # leaky-rectify negative values
  c = tf.math.less(r, -v)
  r = tf.where(c, alpha*r-b, r)
  return r
