from jax import jacfwd, jit
import jax.numpy as jnp
from tqdm import tqdm


# Test, how parallel gradients of two function g1 and g2 are.
# Integrate absolute value of area of parallelogram spanned by g1/norm(g1) and g2/norm(g2)

# 2dimensions: determinant (3d: cross product, nd: using Graam matrices)
def compare_contour_num(g1,g2,xx1,xx2):
    
    dg1 = jacfwd(g1)
    dg2 = jacfwd(g2)
    
    n1 = len(xx1)
    n2 = len(xx2)
       
    @jit
    def area_parallelogram(x):
        
        # evaluate gradients
        dg1v = jnp.array(dg1(x))
        dg2v = jnp.array(dg2(x))
        
        # normalise
        dg1v = dg1v/jnp.linalg.norm(dg1v)
        dg2v = dg2v/jnp.linalg.norm(dg2v)
        
        return jnp.abs(jnp.linalg.det(jnp.array([dg1v,dg2v])))
    
    val = 0.
    
    
    
    for x1 in tqdm(xx1):
        for x2 in xx2:
            
            x = jnp.hstack([x1,x2])
            val = val+ area_parallelogram(x)
            
    return val
            
    

