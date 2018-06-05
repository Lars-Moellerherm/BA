import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


x = np.linspace(-4,4,301)
y = np.linspace(-4,4,301)
z=np.linspace(0,5,301)

X,Y = np.meshgrid(x,y)

Z1 = 2*X-Y-8
Z2 = 4*Y+0.5*X-14.5
y2 = 
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(X,Y,Z1)
ax.plot_surface(X,Y,Z2)
ax.plot(x,y)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
