import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas

rng = np.random.default_rng()

def gen_dataset(ndata, ntaxa=256, plot=True):
  """
  generates a full example dataset
  """
  ### map labels to impacted taxa
  assert ntaxa > 200
  lbl1_taxa = ( 20, 30)
  lbl2_taxa = (100,110)
  lbl3_taxa = (150,180)
  lbl4_taxa = ( 60, 80)
  lbl5_taxa = (120,130)
  ### map label values to taxa perturbations
  lbl1_pert = (0.5, 1.5)
  lbl2_pert = (1.5, 0.5)
  lbl3_pert = (0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8)
  lbl4_mean =  5.0
  lbl5_mean = 10.0
  ### label generation
  # generate 3 categorical labels:
  #  2 binary labels
  #    one with equal probablity 0|1
  #    one skewed toward 0
  #  1 8-category label with equal probabilities
  lbl1 = rng.choice([0,1], size=ndata, p=[0.5,0.5])
  lbl2 = rng.choice([0,1], size=ndata, p=[0.9,0.1])
  lbl3 = rng.choice(np.arange(8), size=ndata)
  # convert labels to one-hot tensors for neural nets
  lbl1_oh = tf.one_hot(lbl1, 2, dtype=tf.float32)
  lbl2_oh = tf.one_hot(lbl2, 2, dtype=tf.float32)
  lbl3_oh = tf.one_hot(lbl3, 8, dtype=tf.float32)
  # generate quantitative labels
  lbl4 = rng.normal(lbl4_mean, 1, size=(ndata,1))
  lbl5 = rng.normal(lbl5_mean, 1, size=(ndata,1))
  # concatenate one-hot encoded labels and quantitative labels
  all_lbls_oh = tf.concat([lbl1_oh, lbl2_oh, lbl3_oh, lbl4, lbl5], axis=-1)

  ### data generation
  # generate raw log-normal distributed data with 1-channel
  abundance_means = rng.lognormal(100, 1, (ntaxa,))
  abundance_stdvs = abundance_means / 10
  rawdata = rng.normal(abundance_means, abundance_stdvs, (ndata,ntaxa,))
  # sort data by column means, in decreasing order
  col_means = np.mean(rawdata, axis=0, keepdims=False)
  sort_idxs = np.argsort(col_means, axis=0).squeeze()
  sort_idxs = np.flip(sort_idxs, axis=0)
  sort_data = rawdata[:,sort_idxs]
  ## data perturbation
  for idx in np.arange(ndata):
    # part 1: lbl1
    pvalu = lbl1_pert[lbl1[idx]]
    ptaxa = np.arange(lbl1_taxa[0], lbl1_taxa[1])
    for idx2 in ptaxa:
      sort_data[idx,idx2] *= pvalu
    # part 2: lbl2
    pvalu = lbl2_pert[lbl2[idx]]
    ptaxa = np.arange(lbl2_taxa[0], lbl2_taxa[1])
    for idx2 in ptaxa:
      sort_data[idx,idx2] *= pvalu
    # part 3: lbl3
    pvalu = lbl3_pert[lbl3[idx]]
    ptaxa = np.arange(lbl3_taxa[0], lbl3_taxa[1])
    for idx2 in ptaxa:
      sort_data[idx,idx2] *= pvalu
    # part 4: lbl4
    pvalu = lbl4_mean / (lbl4[idx] + 1e-8)
    ptaxa = np.arange(lbl4_taxa[0], lbl4_taxa[1])
    for idx2 in ptaxa:
      sort_data[idx,idx2] *= pvalu
    # parg 5: lbl5
    pvalu = lbl5_mean / (lbl5[idx] + 1e-8)
    ptaxa = np.arange(lbl5_taxa[0], lbl5_taxa[1])
    for idx2 in ptaxa:
      sort_data[idx,idx2] *= pvalu
  ## normalize data using CLR (centered log ratio)
  geo_means = scipy.stats.gmean(sort_data, axis=1)
  norm_data = np.log(np.divide(sort_data, geo_means[:,np.newaxis]))
  ## sort each data set independently
  norm_data.sort(axis=1)
  norm_data = np.flip(norm_data, axis=1)

  ## expand channels using 'random walk', if needed
  #next_data = norm_data
  #for _ in np.arange(1,channels):
  #  next_data = np.add(next_data, rng.normal(0, 0.1, size=next_data.shape))
  #  next_data.sort(axis=1)
  #  next_data = np.flip(next_data, axis=1)
  #  norm_data = np.append(norm_data, next_data, axis=-1)

  ## plot data (optional)
  if plot:
    x = np.linspace(1,ntaxa,num=ntaxa)
    # plot data sets
    for i in np.arange(ndata):
      plt.plot(x, norm_data[i,:], 'o', markersize=2)
    plt.ylim([-5,+5])
    plt.show()
  ## return (data,labels) tensors
  return (tf.convert_to_tensor(norm_data, dtype=tf.float32), all_lbls_oh)

if __name__ == '__main__':
  dta_prefix = 'DTA'
  lbl_prefix = 'LBL'
  ndata = 16344
  # generate data
  datat,lablt = gen_dataset(ndata, plot=False)
  # convert to numpy
  datan = datat.numpy()
  labln = lablt.numpy()
  print(datan.shape, labln.shape)
  # create column header
  cols = []
  for i in range(datan.shape[-1]):
    cols.append(dta_prefix + str(i))
  for i in range(labln.shape[-1]):
    cols.append(lbl_prefix + str(i))
  # package into dataframe
  dataframe = pandas.DataFrame(np.concatenate([datan, labln], axis=-1),
                               columns=cols)
  print(dataframe.head())
  # write file
  outfname = 'data.csv'
  dataframe.to_csv(outfname, index=False)
  print('wrote {}'.format(outfname))
