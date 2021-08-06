import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from test_data_generator import gen_dataset


# generate synthetic dataset
ndata = 1000
real_d, real_l = gen_dataset(ndata)

# calculate mean and stdev
data = real_d.numpy()
real_means = np.mean(data, axis=0)
real_stdvs = np.std(data, axis=0)
print(real_means.shape, real_stdvs.shape)

# load model
model = tf.keras.models.load_model(sys.argv[1])
model.summary()

# generate fake dataset
fake_d, fake_l = model(real_l)
data = fake_d.numpy()

# calculate mean and stdev
fake_means = np.mean(data, axis=0)
fake_stdvs = np.std(data, axis=0)

# plot means and stdvs scatters
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(real_means, fake_means, marker='o', color='blue', alpha=0.5)
min = -4.0
max = +4.0
ax1.plot([min,max],[min,max], markersize=0, linewidth=1)
ax1.set_xlim((min,max))
ax1.set_ylim((min,max))
ax1.set_xlabel('real data mean')
ax1.set_ylabel('fake data mean')
ax2.scatter(real_stdvs, fake_stdvs, marker='o', color='red', alpha=0.5)
min = 0.0
max = 0.25
ax2.plot([min,max],[min,max], markersize=0, linewidth=1)
ax2.set_xlim((min,max))
ax2.set_ylim((min,max))
ax2.set_xlabel('real data stdev')
ax2.set_ylabel('fake data stdev')

plt.show()
