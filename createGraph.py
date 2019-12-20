import numpy as np
V = 500000
E = 16
name = "avg_%d_%d.gr"%(V/10000,E)
f = open(name,'w')
f.write("p sp %d %d\n"%(V,V*E))
for i in range(V):
    for j in range(E):
        x = np.random.randint(1,V+1)
        y = np.random.randint(1,1000)
        f.write("a %d %d %d\n"%(i+1,x,y))
# f.write("a")
f.close()