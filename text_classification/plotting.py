# Somethimes 
import math
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.arange(1,100)
ypoints = list( map(lambda x: math.log10(x),xpoints))
plt.plot(xpoints, ypoints)


plt.plot(xpoints, np.array(ypoints))
plt.show()
