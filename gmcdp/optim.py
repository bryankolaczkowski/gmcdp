from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf


class GanOptimizer(Optimizer):
  """
  implements a generator,discriminator optimizer pair
  """
  def __init__(self,
               gen_optimizer='sgd',
               dis_optimizer='sgd',
               **kwargs):
    super(GanOptimizer, self).__init__(name='GanOptimizer', **kwargs)
    self.gen_optimizer = tf.keras.optimizers.get(gen_optimizer)
    self.dis_optimizer = tf.keras.optimizers.get(dis_optimizer)
    return

  """
  def __setattr__(self, name, value):
    super(GanOptimizer, self).__setattr__(name, value)
    if name == 'learning_rate' or name == 'lr':
      self.gen_optimizer.__setattr__(name, value)
      self.dis_optimizer.__setattr__(name, value * self.dis_lr_mult)
    return
  """

  def apply_gradients(self, grads_and_vars,
                      name=None, experimental_aggregate_gradients=True):
    raise NotImplementedError('GAN optimizer should call '
                              'apply_generator_gradients and '
                              'apply_discriminator_gradients instead')

  def apply_generator_gradients(self, grads_and_vars):
    return self.gen_optimizer.apply_gradients(grads_and_vars)

  def apply_discriminator_gradients(self, grads_and_vars):
    return self.dis_optimizer.apply_gradients(grads_and_vars)

  def get_config(self):
    config = super(GanOptimizer, self).get_config()
    config.update({
      'gen_optimizer' : tf.keras.optimizers.serialize(self.gen_optimizer),
      'dis_optimizer' : tf.keras.optimizers.serialize(self.dis_optimizer),
    })
    return config
