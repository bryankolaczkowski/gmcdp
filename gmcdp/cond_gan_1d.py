from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras import Model, optimizers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf

from cond_generator_1d     import CondGen1D
from cond_discriminator_1d import CondDis1D


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


class CondGan1D(Model):
  """
  microbial community generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               **kwargs):
    super(CondGan1D, self).__init__(**kwargs)
    self.genr     = generator
    self.disr     = discriminator
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
            'genr_ls' : genr_ls,}

  def get_config(self):
    config = super(CondGan1D, self).get_config()
    config.update({
      'generator'     : tf.keras.layers.serialize(self.genr),
      'discriminator' : tf.keras.layers.serialize(self.disr),
    })
    return config


if __name__ == '__main__':
  """
  module example test
  """
  import sys
  sys.path.append("../tests")
  import test_data_generator

  ### DATA INTAKE ##############################################################

  ## training data ##
  ndata     = 16344
  #ndata     = 262144
  batchsize = 128

  # generate training simulated data and labels
  dtas,lbls = test_data_generator.gen_dataset(ndata, plot=False)
  print(dtas,lbls)
  lbl_shp = tf.shape(lbls)
  dta_shp = tf.shape(dtas)

  # package data into dataset
  data = tf.data.Dataset.from_tensor_slices((dtas, lbls))
  data = data.shuffle(ndata).batch(batchsize)

  ## validation data ##
  val_ndata = 128

  # generate validation simulated data and labels
  val_dtas,val_lbls = test_data_generator.gen_dataset(val_ndata, plot=False)
  val_data = tf.data.Dataset.from_tensor_slices((val_dtas, val_lbls))
  val_data = val_data.batch(val_ndata)

  ### MODEL BUILD ##############################################################

  # create a little 'generator model'
  gen = CondGen1D((lbl_shp[1],), dta_shp[1])
  gen.summary(positions=[0.4, 0.7, 0.8, 1.0])

  # create a little 'discriminator model'
  dis = CondDis1D(dta_shp[1], lbl_shp[1])
  dis.summary(positions=[0.4, 0.7, 0.8, 1.0])

  # create optimizer
  glr  = 1e-5
  dlr  = glr * 0.1
  gopt = tf.keras.optimizers.SGD(learning_rate=glr, momentum=0.9, nesterov=True)
  dopt = tf.keras.optimizers.SGD(learning_rate=dlr, momentum=0.9, nesterov=True)
  opt  = GanOptimizer(gopt, dopt)

  # create gan
  gan = CondGan1D(generator=gen, discriminator=dis)
  gan.compile(optimizer=opt)

  ### generated data image callback ###
  import io
  import numpy as np
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  class PlotCallback(tf.keras.callbacks.Callback):
    """
    plot generated data
    """
    def __init__(self, log_dir='logs'):
      self.writer = tf.summary.create_file_writer(log_dir + '/gen')
      return

    def plot_data(self, data):
      x = np.arange(0,tf.shape(data)[1],1)
      fig = plt.figure()
      ax  = fig.add_subplot(111)
      y   = data.numpy().transpose()
      #y.sort(axis=0)
      #y = np.flip(y, axis=0)
      ax.plot(x, y, 'o', markersize=2, alpha=0.5)
      ax.set_ylim([-5,+5])
      return fig

    def plot_to_image(self, plot):
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close(plot)
      buf.seek(0)
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      image = tf.expand_dims(image, 0)
      return image

    def on_epoch_end(self, epoch, logs=None):
      # generate 10 example datas
      ((dta,lbl),scr) = self.model(lbls[0:10])
      fig = self.plot_data(dta)
      img = self.plot_to_image(fig)
      with self.writer.as_default():
        tf.summary.image('GenData', img, step=epoch)
      return

  # fit gan
  gan.fit(data,
          epochs=10000,
          verbose=1,
          validation_data=val_data,
          callbacks=[tf.keras.callbacks.TensorBoard(),
                     PlotCallback()])
