import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas

if __name__ == '__main__':
  # read data into pandas data frame
  fname = sys.argv[1]
  dataframe = pandas.read_csv(fname)
  # extract data columns
  dtapre = 'DTA'
  data_ids = [ x for x in dataframe.columns if x.find(dtapre) == 0 ]
  ys = dataframe[data_ids].to_numpy(dtype=np.float32)
  # plot data sample head
  ndata = 10
  x = np.linspace(1,ys.shape[-1],num=ys.shape[-1])
  for i in np.arange(ndata):
    plt.plot(x, ys[i,:], 'o', markersize=2)
  plt.ylim([-5,+5])
  plt.show()
