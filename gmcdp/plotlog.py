import matplotlib.pyplot as plt
import numpy as np
import time

x = np.arange(0,256,1)
y = np.random.normal(size=256)

plt.ion()
fig   = plt.figure()
ax    = fig.add_subplot(111)
data, = ax.plot(x, y, 'o', markersize=2, alpha=0.5)
ax.set_ylim([-5,+5])
ax.set_title('epoch: 0')

with open('log.csv', 'r') as handle:
  iter = 0
  for line in handle:
    iter += 1
    epoch = iter // 10
    ax.set_title('epoch: {}'.format(epoch))
    darr = np.array([float(x) for x in line.split(',')])
    data.set_ydata(darr)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.ioff()
plt.show()
