from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import ops, lax, jacfwd, jacrev, jit, jvp, grad

from functools import partial

@partial(jit, static_argnums=(1,2))	
def Leapfrog(z,h,f):
## classical Leapfrog scheme for force field f
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

	dim = int(len(z)/2)

	#z[dim:] = z[dim:]+h/2*f(z[:dim])
	z = ops.index_update(z, ops.index[dim:], z[dim:]+h/2*f(z[:dim]) )
	#z[:dim] = z[:dim]+h*z[dim:]
	z = ops.index_update(z, ops.index[:dim], z[:dim]+h*z[dim:] )
	#z[dim:] = z[dim:]+h/2*f(z[:dim])
	z = ops.index_update(z, ops.index[dim:], z[dim:]+h/2*f(z[:dim]) )

	return z
	
#@partial(jit, static_argnums=(1,2,3))
def LP(z,n_h,h,forces):
## performes a Leapfrog step with step size h for force field f using n_h intermediate steps
# can compute multiple initial values simultanously, z[k]=list of k-component of all initial values

    
    h_gen =h/n_h
    
    #for j in range(0,n_h+1):
    #    z = Leapfrog(z.copy(),h_gen,forces)
    
    z = lax.fori_loop(0,n_h+1, lambda indx, z: Leapfrog(z,h_gen,forces),z)

    return z
