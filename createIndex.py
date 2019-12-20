f = open("indexs20",'w')
V=200000
import numpy as np
for i in range(V):
    x = np.random.randint(1,V)
    f.write("%d "%(x))
f.write("%d "%(-1))
f.close()