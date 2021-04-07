#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import argparse
import distutils.util

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf

rng = numpy.random.default_rng()

def make_microbes(targets, ntaxa, plot=True):
  """
  create random microbial community data
  targets is list of labels
  """
  # target length is number of samples
  nsamp = targets.shape[0]

  # generate raw log-normal distributed data
  abundance_means = rng.lognormal(10, .01, ntaxa)
  abundance_stdvs = abundance_means / 500
  rawdata = rng.normal(abundance_means, abundance_stdvs, (nsamp,ntaxa))

  # sort data by column means, in decreasing order
  col_means = numpy.mean(rawdata, axis=0, keepdims=True)
  sort_idxs = numpy.argsort(col_means.min(axis=0))[::-1]
  sort_data = rawdata[:,sort_idxs]

  # perturb some taxa
  npert = 10
  for idx in numpy.arange(nsamp):
    ptaxa = numpy.arange(100,120)
    pvalu = 0.995
    if targets[idx] == 1:
      ptaxa = numpy.arange(140,160)
      pvalu = 1.005
    pidxs = numpy.random.choice(ptaxa,npert)
    for idx2 in pidxs:
      sort_data[idx,idx2] *= pvalu

  # normalize data using CLR (centered log ratio)
  geo_means = scipy.stats.gmean(sort_data, axis=1)
  norm_data = numpy.log(numpy.divide(sort_data, geo_means[:,numpy.newaxis]))

  # sort each data sample independently
  # this decouples specific taxa across data samples, so columns don't identify
  # specific taxa, anymore. This is a bit weird, but we'll see how it goes.
  # Alternatively, sort only by column means (ie, use norm_data directly); this
  # makes each column correspond to a specific bacterial taxon, but we must
  # then map every microbial community to the SAME taxa, making it difficult
  # to add new data to the study without having to retrain the GAN
  norm_data.sort(axis=1)
  retr_data = numpy.flip(norm_data, axis=1)

  # plot the sorted column means
  if plot:
    sorted_col_means = numpy.sort(col_means[-1])[::-1]
    geo_col_means    = scipy.stats.gmean(sorted_col_means)
    norm_col_means   = numpy.log(numpy.divide(sorted_col_means, geo_col_means))

    x = numpy.linspace(1,ntaxa,num=ntaxa)
    plt.plot(x, norm_col_means, 'o', color='gray')

  return retr_data


def make_data(outfname, ntaxa, nsamples):
  """
  create a random data set
  plot a sample from the data set before writing data to file
  """
  # generate random target labels
  targets  = rng.choice([0,1], size=nsamples)
  microbes = make_microbes(targets, ntaxa, plot=True)

  # plot a random example datasets
  x = numpy.linspace(1,ntaxa,num=ntaxa)
  for i in range(10):
    rnd_idx = numpy.random.choice(nsamples)
    type = 'bo'
    if targets[rnd_idx] == 1:
      type = 'ro'
    plt.plot(x, microbes[rnd_idx,:], type, markersize=1)

  plt.show()

  # write data to file
  with open(outfname, 'w') as outf:
    outf.write('id,target')
    for t in range(ntaxa):
      outf.write(',t{}'.format(t))
    outf.write('\n')
    for idx in range(nsamples):
      trgt = targets[idx]
      taxa = microbes[idx]
      outf.write('{},{},'.format(idx, trgt))
      outf.write(','.join(['{:.4e}'.format(taxon) for taxon in taxa]))
      outf.write('\n')
  return


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


def test_generator(ntaxa, nsamples, generator_file):
  # load generator
  generator = tf.keras.models.load_model(generator_file)
  generator.summary()

  # set labels for data
  lbls_ones = numpy.ones(shape=nsamples)
  lbls_zros = numpy.zeros(shape=nsamples)

  # generate real data
  realdata_ones = make_microbes(lbls_ones, ntaxa, plot=False)
  realdata_zros = make_microbes(lbls_zros, ntaxa, plot=False)

  # generate fake data
  # result from generator should be (data,labels)
  fakedata_ones = generator(lbls_ones, training=False)[0]\
                           .numpy().reshape((nsamples,ntaxa))
  fakedata_zros = generator(lbls_zros, training=False)[0]\
                           .numpy().reshape((nsamples,ntaxa))

  # plot data
  fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
  plot_data(realdata_zros, fakedata_zros, axs[0])
  plot_data(realdata_ones, fakedata_ones, axs[1])

  plt.tight_layout()
  plt.show()
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
                description='generate data and test GAN',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--ntaxa', dest='ntaxa', type=int,
                      help='number of taxa per microbiome', metavar='N')
  parser.add_argument('--nsamples', dest='nsamples', type=int,
                      help='number of microbiome samples', metavar='N')
  parser.add_argument('--testgen', dest='testgen',
                      type=distutils.util.strtobool,
                      help='forego data generation, test GAN',
                      metavar='y|n')
  parser.add_argument('--generator', dest='generator',
                      help='test model from this file', metavar='MDL')

  parser.set_defaults(ntaxa=256,
                      nsamples=2048,
                      testgen=False,
                      generator=None)

  args = parser.parse_args()

  if args.testgen and args.generator==None:
    sys.stderr.write('ERRR: must specify generator to test\n')
    sys.exit(1)

  if not args.testgen:
    make_data('data.csv', args.ntaxa, args.nsamples)

  else:
    test_generator(args.ntaxa, args.nsamples, args.generator)
