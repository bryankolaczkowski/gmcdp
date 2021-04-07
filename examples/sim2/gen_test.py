#!/usr/bin/env python3

import numpy
import scipy.stats
import matplotlib.pyplot as plt

ntaxa = 256    # number of taxa in 'relative abundance' data
nsamp = 2048   # number of samples

rng = numpy.random.default_rng()

def make_microbes(targets):
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
  sorted_col_means = numpy.sort(col_means[-1])[::-1]
  geo_col_means    = scipy.stats.gmean(sorted_col_means)
  norm_col_means   = numpy.log(numpy.divide(sorted_col_means, geo_col_means))

  x = numpy.linspace(1,ntaxa,num=ntaxa)
  plt.plot(x, norm_col_means, 'o', color='gray')

  return retr_data


def make_data(outfname):
  # generate random target labels
  targets  = rng.choice([0,1], size=nsamp)
  microbes = make_microbes(targets)

  # plot a random example datasets
  x = numpy.linspace(1,ntaxa,num=ntaxa)
  for i in range(10):
    rnd_idx = numpy.random.choice(nsamp)
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
    for idx in range(nsamp):
      trgt = targets[idx]
      taxa = microbes[idx]
      outf.write('{},{},'.format(idx, trgt))
      outf.write(','.join(['{:.4e}'.format(taxon) for taxon in taxa]))
      outf.write('\n')
  return


if __name__ == '__main__':
  make_data('data.csv')
