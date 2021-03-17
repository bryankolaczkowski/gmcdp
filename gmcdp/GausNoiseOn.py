from tensorflow.python.keras import backend as K
from tensorflow.python.ops   import array_ops
from tensorflow.keras.layers import GaussianNoise

### implements Gaussian noise generation that is always on;
### during training and generation

class GausNoiseOn(GaussianNoise):

  def __init__(self, stddev, **kwargs):
    super(GausNoiseOn, self).__init__(stddev, **kwargs)
    return

  def call(self, inputs, training=None):
    return inputs * K.random_normal(shape=array_ops.shape(inputs),
                                    mean=0.,
                                    stddev=self.stddev,
                                    dtype=inputs.dtype)
