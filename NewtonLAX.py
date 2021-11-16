from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, jacfwd

def NewtonLAX(f,z,minstepsize=1e-10,maxiter=100):
    # implementation of Newton method working with JAX.lax for jitting    
    return MyNewtonLAX(f,z,jacfwd(f),minstepsize=minstepsize,maxiter=maxiter)



def MyNewtonLAX(f,z,fprime,minstepsize=1e-10,maxiter=100):
    # implementation of Newton method working with JAX.lax for jitting    
    
    dim = z.shape[0]
    
    dz = jnp.linalg.solve(fprime(z),f(z))
    
    izdz = jnp.hstack([0,z,dz])
    
    def body_fun (izdz):
        i  = izdz[0] 
        z  = izdz[1:dim+1]
        dz = izdz[dim+1:]
        
        z = z - dz
        fz = f(z)
        dz = jnp.linalg.solve(fprime(z),fz)
        
        i = i+1
        
        izdz = jnp.hstack([i,z,dz])
        
        return izdz
    
    def cond_fun(izdz):
        stepsize = jnp.max(jnp.abs(izdz[dim+1:]))
        return jnp.logical_and((stepsize > minstepsize) , (izdz[0] <= maxiter))
    
    
    izdz = lax.while_loop(cond_fun, body_fun, izdz)
    
    
    z  = izdz[1:dim+1] 
    
    
    
    return z
