#!/usr/bin/env python3

import numpy
import scipy.stats
import matplotlib.pyplot

ntaxa = 32
nsamp = 10000

rng = numpy.random.default_rng()

abundance_means = rng.lognormal(10, .1, ntaxa)
abundance_stdvs = abundance_means / 100

rawdata = rng.normal(abundance_means, abundance_stdvs, (nsamp,ntaxa))

col_means = numpy.mean(rawdata, axis=0, keepdims=True)
sort_idxs = numpy.argsort(col_means.min(axis=0))[::-1]

sort_data = rawdata[:,sort_idxs]
geo_means = scipy.stats.gmean(sort_data, axis=1)
norm_data = numpy.log(numpy.divide(sort_data, geo_means[:,numpy.newaxis]))

sorted_col_means = numpy.sort(col_means[-1])[::-1]
geo_col_means    = scipy.stats.gmean(sorted_col_means)
norm_col_means   = numpy.log(numpy.divide(sorted_col_means, geo_col_means))

x = numpy.linspace(1,ntaxa,num=ntaxa)
matplotlib.pyplot.plot(x, norm_col_means, 'bo')
for i in range(20):
  matplotlib.pyplot.plot(x, norm_data[i,:], 'ro', markersize=2)
matplotlib.pyplot.show()

numpy.savetxt('data.csv', norm_data, fmt='%.4f', delimiter=',')
