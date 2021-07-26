from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


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
