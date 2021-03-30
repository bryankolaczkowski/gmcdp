#!/usr/bin/env python3

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy
import scipy.stats
import tensorflow        as tf
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ntaxa = 256    # number of taxa in 'relative abundance' data

rng = numpy.random.default_rng()

def gen_data(perturb_up, nsamp):
  # generate raw log-normal distributed data
  abundance_means = rng.lognormal(10, .01, ntaxa)
  abundance_stdvs = abundance_means / 1000
  rawdata = rng.normal(abundance_means, abundance_stdvs, (nsamp,ntaxa))

  # sort data by column means, in decreasing order
  col_means = numpy.mean(rawdata, axis=0, keepdims=True)
  sort_idxs = numpy.argsort(col_means.min(axis=0))[::-1]
  sort_data = rawdata[:,sort_idxs]

  # perturb some taxa
  npert = 10
  ptaxa = numpy.arange(100,120)
  pvalu = 0.996
  if perturb_up:
    ptaxa = numpy.arange(140,160)
    pvalu = 1.004

  for idx in range(nsamp):
    pidxs = numpy.random.choice(ptaxa,npert)
    for idx2 in pidxs:
      sort_data[idx,idx2] *= pvalu

  # normalize data using CLR (centered log ratio)
  geo_means = scipy.stats.gmean(sort_data, axis=1)
  norm_data = numpy.log(numpy.divide(sort_data, geo_means[:,numpy.newaxis]))
  return norm_data


def test_gen(perturb_up, n_samples, gen):
  idnum = 0
  if perturb_up:
    idnum = 1

  ## get real data
  real_data = gen_data(perturb_up, n_samples)

  ## load model
  mstr = str(gen)
  if gen < 0:
    mstr = ''
  mfname = 'data' + str(idnum) + '.csv.gen.model' + mstr
  generator = tf.keras.models.load_model(mfname)
  generator.summary()

  ## get fake data
  z = tf.random.normal(shape=(n_samples, ntaxa))
  fake_data = generator(z, training=False).numpy().reshape((n_samples,ntaxa))

  return (real_data, fake_data)


def plot_data(real_data, fake_data, axs):
  ## compare real to fake data
  axis = 0
  real_means = numpy.mean(real_data, axis=axis)
  real_stdvs = numpy.std( real_data, axis=axis, ddof=1)
  fake_means = numpy.mean(fake_data, axis=axis)
  fake_stdvs = numpy.std( fake_data, axis=axis, ddof=1)

  ## plot
  min = numpy.minimum(numpy.amin(real_means), numpy.amin(fake_means)) * 1.1
  max = numpy.maximum(numpy.amax(real_means), numpy.amax(fake_means)) * 1.1
  axs[0].set_xlim(min, max)
  axs[0].set_ylim(min, max)
  axs[0].plot([min,max],[min,max], linewidth=0.5, color='gray')
  axs[0].scatter(real_means, fake_means, s=2, alpha=0.8)

  min = 0.0
  max = numpy.maximum(numpy.amax(real_stdvs), numpy.amax(fake_stdvs)) * 1.1
  axs[1].set_xlim(min, max)
  axs[1].set_ylim(min, max)
  axs[1].plot([min,max],[min,max], linewidth=0.5, color='gray')
  axs[1].scatter(real_stdvs, fake_stdvs, s=2, alpha=0.8, color='red')
  axs[0].set_aspect(1)
  axs[1].set_aspect(1)

  return


if __name__ == '__main__':
  gen = int(sys.argv[1])

  n_samples = 1000
  perturb_up = False

  fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)

  real_data,fake_data = test_gen(perturb_up, n_samples, gen)
  plot_data(real_data, fake_data, axs[0])

  perturb_up = True
  #real_data,fake_data = test_gen(perturb_up, n_samples, gen)
  #plot_data(real_data, fake_data, axs[1])

  plt.tight_layout()
  plt.show()



  """
    ### plot example results
    matplotlib.rcParams.update({'font.size':8})

    x = numpy.linspace(1, args.data_dim, num=args.data_dim)
    gens = numpy.arange(0, len(gen_points_list))

    fig,axs = plt.subplots(nrows=3, ncols=n_samples, figsize=(n_samples,3),
                           sharex=True, sharey=True)

    # set up example data logging
    axs[0,0].set_ylabel('real data')
    axs[1,0].set_ylabel('fake data')
    dyn_data = []
    for i in range(n_samples):
      axs[0,i].get_xaxis().set_ticks([])
      axs[0,i].get_yaxis().set_ticks([])
      axs[1,i].get_xaxis().set_ticks([])
      axs[1,i].get_yaxis().set_ticks([])
      ax = axs[0,i]
      y  = real_points_list[i]
      ax.plot(x,y, 'bo', markersize=1)
      dd, = axs[1,i].plot([],[], 'ro', markersize=1)
      dyn_data.append(dd)

    # set up loss logging
    lsax = plt.subplot2grid((3,n_samples), (2,0), colspan=n_samples)
    lsax.set_xlim([0, len(gen_points_list)-1])
    min_loss = numpy.amin([x for x in log_lists.values()])
    max_loss = numpy.amax([x for x in log_lists.values()])
    lsax.set_ylim([min_loss, max_loss])
    lsax.set_ylabel('loss')
    lsax.set_xlabel('generation')
    lsax.get_xaxis().set_ticks([])
    lsax.get_yaxis().set_ticks([])
    for loss in log_lists.keys():
      dd, = lsax.plot([],[], linewidth=0.5, label=loss)
      dyn_data.append(dd)
    lsax.legend(loc='upper left', borderpad=0.2,
                                  labelspacing=0.2,
                                  handlelength=0.5,
                                  handletextpad=0.2)

    plt.tight_layout()

    def update(data_idx):
      # example data
      data = gen_points_list[data_idx]
      for i in range(n_samples):
        y  = data[i,:,:].ravel()
        dyn_data[i].set_data(x,y)
      # losses
      loss_x = gens[:data_idx+1]  ### plot example results
  matplotlib.rcParams.update({'font.size':8})

  x = numpy.linspace(1, args.data_dim, num=args.data_dim)
  gens = numpy.arange(0, len(gen_points_list))

  fig,axs = plt.subplots(nrows=3, ncols=n_samples, figsize=(n_samples,3),
                         sharex=True, sharey=True)

  # set up example data logging
  axs[0,0].set_ylabel('real data')
  axs[1,0].set_ylabel('fake data')
  dyn_data = []
  for i in range(n_samples):
    axs[0,i].get_xaxis().set_ticks([])
    axs[0,i].get_yaxis().set_ticks([])
    axs[1,i].get_xaxis().set_ticks([])
    axs[1,i].get_yaxis().set_ticks([])
    ax = axs[0,i]
    y  = real_points_list[i]
    ax.plot(x,y, 'bo', markersize=1)
    dd, = axs[1,i].plot([],[], 'ro', markersize=1)
    dyn_data.append(dd)

  # set up loss logging
  lsax = plt.subplot2grid((3,n_samples), (2,0), colspan=n_samples)
  lsax.set_xlim([0, len(gen_points_list)-1])
  min_loss = numpy.amin([x for x in log_lists.values()])
  max_loss = numpy.amax([x for x in log_lists.values()])
  lsax.set_ylim([min_loss, max_loss])
  lsax.set_ylabel('loss')
  lsax.set_xlabel('generation')
  lsax.get_xaxis().set_ticks([])
  lsax.get_yaxis().set_ticks([])
  for loss in log_lists.keys():
    dd, = lsax.plot([],[], linewidth=0.5, label=loss)
    dyn_data.append(dd)
  lsax.legend(loc='upper left', borderpad=0.2,
                                labelspacing=0.2,
                                handlelength=0.5,
                                handletextpad=0.2)

  plt.tight_layout()

  def update(data_idx):
    # example data
    data = gen_points_list[data_idx]
    for i in range(n_samples):
      y  = data[i,:,:].ravel()
      dyn_data[i].set_data(x,y)
    # losses
    loss_x = gens[:data_idx+1]
    i = n_samples
    for loss in log_lists.values():
      loss_y = loss[:data_idx+1]
      dyn_data[i].set_data(loss_x, loss_y)
      i += 1
    return dyn_data,

  anim = FuncAnimation(fig, update,
                       frames=numpy.arange(0,len(gen_points_list)),
                       interval=200)
  #plt.show()

  writer = PillowWriter(fps=4)
  gifname = args.file + '.out.gif'
  anim.save(gifname, writer=writer)
      i = n_samples
      for loss in log_lists.values():
        loss_y = loss[:data_idx+1]
        dyn_data[i].set_data(loss_x, loss_y)
        i += 1
      return dyn_data,

    anim = FuncAnimation(fig, update,
                         frames=numpy.arange(0,len(gen_points_list)),
                         interval=200)
    #plt.show()

    writer = PillowWriter(fps=4)
    gifname = args.file + '.out.gif'
    anim.save(gifname, writer=writer)
    """
