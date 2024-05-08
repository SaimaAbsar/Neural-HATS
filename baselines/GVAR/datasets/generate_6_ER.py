# OCT 2023
# new data generated according to ER-graphs
# for 6 variables only

import numpy as np
import random
#import igraph as ig

seed = 0
random.seed(seed)
np.random.seed(seed)

'''def generate_ER_graph(n,m):
    G_und = ig.Graph.Erdos_Renyi(n=n, m=m, directed=True, loops=False)
    G = np.array(G_und.get_adjacency().data)
    print("adjacency:\n", G)
    return G'''

def generate_timeseries(n, noise_scale):
    # 1D Array as per Gaussian Distribution
    
    a = (np.random.randn(5)).tolist()
    b = (np.random.randn(5)).tolist()
    c = (np.random.randn(5)).tolist()
    d = (np.random.randn(5)).tolist()
    e = (np.random.randn(5)).tolist()
    f = (np.random.randn(5)).tolist()
    
    print(a,b,c,d,e,f)
    
    for i in range(5,n):

        ta = 0.85*np.cos(b[i-1]+1) + 0.65*np.cos(e[i-1]+1) +\
            0.15*np.cos(b[i-2]+1) + 0.95*np.cos(e[i-2]+1) +\
                0.35*np.cos(b[i-3]+1) + 1.05*np.cos(e[i-3]+1) +\
                    0.55*np.cos(b[i-4]+1) + 0.8*np.cos(e[i-4]+1) +\
                        0.75*np.cos(b[i-5]+1) + 0.25*np.cos(e[i-5]+1) + noise_scale*np.random.randn()*10 #noise_scale*e1[i] 
        a.append(ta)
        tb = 0.8*np.cos(c[i-1]+1) + 0.93*np.cos(a[i-1]+1) +\
            0.7*np.cos(c[i-2]+1) + 0.33*np.cos(a[i-2]+1) +\
                0.9*np.cos(c[i-3]+1) + 0.53*np.cos(a[i-3]+1) +\
                    0.08*np.cos(c[i-4]+1) + 0.63*np.cos(a[i-4]+1) +\
                        0.18*np.cos(c[i-5]+1) + 0.39*np.cos(a[i-5]+1) + noise_scale*np.random.randn()*10   #noise_scale*e2[i] 
        b.append(tb) 
        tc = np.random.randn()
        c.append(tc) 

        td = 0.28*np.cos(e[i-1]+1) +\
            0.75*np.cos(e[i-2]+1) +\
                0.59*np.cos(e[i-3]+1) + \
                    0.8*np.cos(e[i-4]+1) +\
                        0.18*np.cos(e[i-5]+1) + noise_scale*np.random.randn()*5 #noise_scale*e3[i]
        d.append(td)

        tf = 0.6*np.cos(d[i-1]+1) +\
            0.21*np.cos(d[i-2]+1) + \
                1*np.cos(d[i-3]+1) + \
                    1.3*np.cos(d[i-4]+1) + \
                        0.81*np.cos(d[i-5]+1) + noise_scale*np.random.randn()*5  #e5[i]#*noise_scale  #np.random.randn(1)#
        f.append(tf) 

        te=  0.8*np.cos(c[i-1]+1) +\
            0.7*np.cos(c[i-2]+1) +\
                0.9*np.cos(c[i-3]+1) + \
                    0.08*np.cos(c[i-4]+1) +\
                        0.18*np.cos(c[i-5]+1) + noise_scale*np.random.randn()*5 #noise_scale*e3[i]
        e.append(te)
    #e = np.random.normal(mean,std,n)
  
    T = [a,b,c,d,e,f]
    T = np.array(T)
    #print(len(T))
    return T.T

  


