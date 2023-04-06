from typing import Tuple
import numpy as np

def create_flat_triangular_mesh(step: float, 
                                smoothness: float = 0,
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a square in [-1, 1] evenly divided in squares of side (step)
    where each square is evenly divided in two triangles. The z coordinate of
    each point is determined by s=(smoothness) and uniformly sampled in 
    [-s/2, s/2]."""
    x = np.arange(start=-1, stop=1, step=step)
    y = np.arange(start=-1, stop=1, step=step)
    N_p = len(x) * len(y)
    p = np.zeros((N_p, 3))
    N_t = (len(x)-1) * (len(y)-1) * 2
    tri = np.zeros((N_t, 3))   
    n_p = 0   
    n_t = 0    
    for i in range(len(x)):    
        for j in range(len(y)):   
            p[n_p, :] = [x[i], y[j], 0] 
            if (i < len(x)-1) and (j < len(y)-1):       
                tri[n_t] = [n_p, n_p+1, n_p+len(y)]
                n_t += 1
                tri[n_t] = [n_p+1, n_p+len(y), n_p+len(y)+1]
                n_t += 1
            n_p += 1   
    p[:,-1] = smoothness * (0.5 - np.random.rand(p.shape[0]))
    return p, tri