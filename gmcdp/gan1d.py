from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras import Model, optimizers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf

from generator1d     import MCGen
from discriminator1d import MCDis


class WassersteinLoss(Loss):
  """
  implements wasserstein loss function

  'earth mover' distance from:
    https://arxiv.org/pdf/1701.07875.pdf
    https://arxiv.org/pdf/1704.00028.pdf
  """
  def __init__(self):
    super(WassersteinLoss, self).__init__(name='wasserstein_loss')
    return

  def call(self, y_true, y_pred):
    return K.mean(y_true * y_pred)


class GanOptimizer(Optimizer):
  """
  implements a generator,discriminator optimizer pair
  """
  def __init__(self,
               gen_optimizer='adam',
               dis_optimizer='adam',
               **kwargs):
    super(GanOptimizer, self).__init__(name='GanOptimizer', **kwargs)
    self.gen_optimizer = optimizers.get(gen_optimizer)
    self.dis_optimizer = optimizers.get(dis_optimizer)
    return

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


class MCGan(Model):
  """
  microbial community generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               dis_updates=2,
               **kwargs):
    super(MCGan, self).__init__(**kwargs)
    self.genr = generator
    self.disr = discriminator
    self.dis_updates = dis_updates
    # create exponential moving-average scale for path length regularization
    self.pl_scale = tf.Variable(0.0, trainable=False)
    return

  def compile(self,
              optimizer=GanOptimizer(),
              loss=WassersteinLoss(),
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    super(MCGan, self).compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
    return

  def call(self, inputs):
    x = self.genr(inputs)
    x = self.disr(x)
    return x

  def _path_len_reg_loss(self, losses):
    """
    calculates path-length regularization loss across batches and layers
    """
    # get average of current path length regularization losses
    mean_loss = tf.reduce_mean(losses)
    # update exponential moving average scale
    new_scale = self.pl_scale + 0.01 * (mean_loss - self.pl_scale)
    self.pl_scale.assign_sub(new_scale)
    # calculate 2 * (mean_loss - self.pl_scale)^2
    current_loss = tf.math.square(mean_loss - self.pl_scale) * 2.0
    return current_loss

  def _img_div_loss(self, real_data, fake_data):
    """
    returns high loss when fake_data is very similar to real_data
    """
    real_imgs = real_data[0]
    fake_imgs = fake_data[0]
    rrmsd = tf.math.rsqrt(tf.math.reduce_mean(
                            tf.math.square(fake_imgs - real_imgs)
                         ) + 1.0e-8 )
    return rrmsd

  def train_step(self, real_data):
    """
    single training step
    """
    bs = tf.shape(real_data[0])[0]
    pones =  tf.ones((bs,1))
    nones = -tf.ones((bs,1))

    for _ in range(self.dis_updates):
      # train discriminator using real data
      with tf.GradientTape() as tape:
        preds   = self.disr(real_data)
        disr_rl = self.compiled_loss(nones, preds)
      grds = tape.gradient(disr_rl, self.disr.trainable_weights)
      self.optimizer.apply_discriminator_gradients(zip(grds,
                                                   self.disr.trainable_weights))

      # train discriminator and generator using fake data
      with tf.GradientTape(persistent=True) as tape:
        fake_data = self.genr(real_data)
        preds     = self.disr(fake_data)
        disr_fl   = self.compiled_loss(pones, preds)
      grds = tape.gradient(disr_fl, self.disr.trainable_weights)
      self.optimizer.apply_discriminator_gradients(zip(grds,
                                                   self.disr.trainable_weights))

    # train generator
    with tf.GradientTape() as tape:
      fake_data = self.genr(real_data)
      # calculate path length regularization loss
      genr_plr_loss = self._path_len_reg_loss(self.genr.losses)
      # calculate image diversity loss
      genr_div_loss = self._img_div_loss(real_data, fake_data)
      # calculate discriminator-induced loss
      preds         = self.disr(fake_data)
      genr_dis_loss = self.compiled_loss(nones, preds)
      genr_loss = genr_dis_loss + genr_plr_loss + genr_div_loss
    grds = tape.gradient(genr_loss, self.genr.trainable_weights)
    self.optimizer.apply_generator_gradients(zip(grds,
                                             self.genr.trainable_weights))

    return {'disr_rl'   : disr_rl,
            'disr_fl'   : disr_fl,
            'genr_loss' : genr_loss,
            'genr_bse'  : genr_dis_loss,
            'genr_pth'  : genr_plr_loss,
            'genr_div'  : genr_div_loss,
           }

  def get_config(self):
    config = super(MCGanBase, self).get_config()
    config.update({
      'generator'     : tf.keras.layers.serialize(self.genr),
      'discriminator' : tf.keras.layers.serialize(self.disr),
    })
    return config


if __name__ == '__main__':
  """
  module example test
  """
  from generator1d import MCGen
  from discriminator1d import MCDis

  batchsize = 8
  ndata     = 100
  labelshp  = (ndata,1)
  datashp   = (ndata,1024,1)

  # create random data and labels
  datas  = tf.random.normal(shape=datashp)
  labels = tf.math.rint(tf.random.uniform(shape=labelshp))

  # package into dataset
  data = tf.data.Dataset.from_tensor_slices((datas, labels))
  data = data.shuffle(ndata).batch(batchsize)

  # create gan
  gan = MCGan(generator=MCGen(), discriminator=MCDis())
  gan.compile()

  # fit gan
  gan.fit(data, epochs=10)
