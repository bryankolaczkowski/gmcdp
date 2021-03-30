#!/usr/bin/env python3

import numpy
import scipy.stats
import matplotlib.pyplot as plt

ntaxa = 256    # number of taxa in 'relative abundance' data
nsamp = 2560   # number of samples

rng = numpy.random.default_rng()

def make_data(outfname, perturb_up):
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

  # plot a random example datasets against the sorted column means
  sorted_col_means = numpy.sort(col_means[-1])[::-1]
  geo_col_means    = scipy.stats.gmean(sorted_col_means)
  norm_col_means   = numpy.log(numpy.divide(sorted_col_means, geo_col_means))

  x = numpy.linspace(1,ntaxa,num=ntaxa)
  plt.plot(x, norm_col_means, 'bo')

  for i in range(10):
    rnd_idx = numpy.random.choice(nsamp)
    plt.plot(x, norm_data[rnd_idx,:], 'ro', markersize=2)

  plt.show()

  # write data to file
  numpy.savetxt(outfname, norm_data, fmt='%.4e', delimiter=',')
  return

if __name__ == '__main__':
  make_data('data0.csv', False)
  make_data('data1.csv', True)
