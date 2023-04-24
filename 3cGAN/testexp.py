import numpy as np

x = [0,0]
y = [1,1]
z = [x,y]
z1 = np.asarray(z)
z2 = np.mean(z1,axis=0)
print(z2)