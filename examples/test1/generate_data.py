#!/usr/bin/env python3

import numpy
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
norm_data = sort_data / (numpy.sum(sort_data, axis=1)[0])

sorted_col_means = numpy.sort(col_means[-1])[::-1]
norm_col_means   = sorted_col_means / numpy.sum(sorted_col_means)
matplotlib.pyplot.plot(numpy.linspace(1,ntaxa,num=ntaxa),
                      -numpy.log(norm_col_means), 'bo')
for i in range(10):
  matplotlib.pyplot.plot(numpy.linspace(1,ntaxa,num=ntaxa),
                        -numpy.log(norm_data[i,:]), 'ro', markersize=2)
matplotlib.pyplot.show()

numpy.savetxt('data.csv', norm_data, fmt='%.4f', delimiter=',')
