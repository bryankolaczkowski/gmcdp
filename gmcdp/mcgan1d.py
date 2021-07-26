from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import Model
import tensorflow as tf

from .losss   import WassersteinLoss
from .optim   import GanOptimizer
from .mcgen1d import CondGen1D
from .mcdis1d import CondDis1D


class CondGan1D(Model):
  """
  microbial community generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               **kwargs):
    super(CondGan1D, self).__init__(**kwargs)
    self.genr = generator
    self.disr = discriminator
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
    super(CondGan1D, self).compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics,
                               loss_weights=loss_weights,
                               weighted_metrics=weighted_metrics,
                               run_eagerly=run_eagerly,
                               steps_per_execution=steps_per_execution,
                               **kwargs)
    return

  def call(self, inputs, training=None):
    """
    inputs should be labels
    """
    gdta,lbls = self.genr(inputs, training=training)  # generate data, labels
    dsr_score = self.disr((gdta,
                           self.genr(inputs, training=training)[0],
                           inputs), training=training)
    return ((gdta, lbls), dsr_score)

  def augment_data(self, dta):
    """
    data augmenation function
    """
    """
    ## noisify 10% random data entries
    mask = tf.cast(tf.random.categorical(tf.math.log([[0.9, 0.1]]),
                                         tf.math.reduce_prod(tf.shape(dta))),
                                         tf.float32)
    mask = tf.reshape(mask, shape=tf.shape(dta))
    nois = tf.random.normal(mean=0.0, stddev=0.5, shape=tf.shape(dta)) * mask
    dta  = dta + nois
    ## obliterate 10% random data entries
    mask = tf.cast(tf.random.categorical(tf.math.log([[0.1, 0.9]]),
                                         tf.math.reduce_prod(tf.shape(dta))),
                                         tf.float32)
    mask = tf.reshape(mask, shape=tf.shape(dta))
    dta  = dta * mask
    """
    return dta

  def _calc_loss(self, qry_data, gnr_data, lbls, y, training=None):
    """
    calculates appropriate loss function
    """
    y_hat = self.disr((qry_data, gnr_data, lbls), training=training)
    return self.compiled_loss(y, y_hat)

  def test_step(self, inputs):
    """
    single validation step
    """
    bs    = tf.shape(inputs[0])[0]  # batch size
    pones =  tf.ones((bs,1))        # positive labels
    nones = -tf.ones((bs,1))        # negative labels
    data  = inputs[0]               # input data
    lbls  = inputs[1]               # input labels
    # discriminator loss on real data
    disr_rl = self._calc_loss(qry_data=data,
                              gnr_data=self.genr(lbls, training=False)[0],
                              lbls=lbls,
                              y=nones,
                              training=False)
    # discriminator loss on fake data
    disr_fk = self._calc_loss(qry_data=self.genr(lbls, training=False)[0],
                              gnr_data=self.genr(lbls, training=False)[0],
                              lbls=lbls,
                              y=pones,
                              training=False)
    # generator loss
    genr_ls = self._calc_loss(qry_data=self.genr(lbls, training=False)[0],
                              gnr_data=self.genr(lbls, training=False)[0],
                              lbls=lbls,
                              y=nones,
                              training=False)
    return {'disr_rl' : disr_rl,
            'disr_fk' : disr_fk,
            'genr_ls' : genr_ls,}

  def train_step(self, inputs):
    """
    single training step; inputs are (data,labels)
    """
    bs = tf.shape(inputs[0])[0]

    # labels
    pones =  tf.ones((bs,1))
    nones = -tf.ones((bs,1))

    # split data and labels
    data = inputs[0]
    lbls = inputs[1]

    # train discriminator using real data
    with tf.GradientTape() as tape:
      disr_rl = self._calc_loss(qry_data=self.augment_data(data),
                                gnr_data=self.genr(lbls, training=False)[0],
                                lbls=lbls,
                                y=nones,
                                training=True)
    grds = tape.gradient(disr_rl, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train discriminator using fake data
    with tf.GradientTape() as tape:
      disr_fk = self._calc_loss(\
                qry_data=self.augment_data(self.genr(lbls, training=False)[0]),
                gnr_data=self.genr(lbls, training=False)[0],
                lbls=lbls,
                y=pones,
                training=True)
    grds = tape.gradient(disr_fk, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train generator
    with tf.GradientTape() as tape:
      genr_ls = self._calc_loss(\
                qry_data=self.augment_data(self.genr(lbls, training=True)[0]),
                gnr_data=self.genr(lbls, training=False)[0],
                lbls=lbls,
                y=nones,
                training=False)
    grds = tape.gradient(genr_ls, self.genr.trainable_weights)
    self.optimizer.apply_generator_gradients(zip(grds,
                                             self.genr.trainable_weights))

    return {'disr_rl' : disr_rl,
            'disr_fk' : disr_fk,
            'genr_ls' : genr_ls,
            'genr_lr' : self.optimizer.gen_optimizer.lr(\
                                  self.optimizer.gen_optimizer.iterations),
            'disr_lr' : self.optimizer.dis_optimizer.lr(\
                                  self.optimizer.dis_optimizer.iterations),}

  def get_config(self):
    config = super(CondGan1D, self).get_config()
    config.update({
      'generator'     : tf.keras.layers.serialize(self.genr),
      'discriminator' : tf.keras.layers.serialize(self.disr),
    })
    return config
