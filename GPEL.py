from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax import ops, lax, jacfwd, jit
from functools import partial

import math
import itertools

class GPEL:

    def __init__(self,kernel,h,data_train):
        
        self.dim = int(data_train.shape[0]/3) # dimension of state space
        
        self.k = kernel
        self.h = h
        self.data_train = data_train
        self.Z = data_train[:2*self.dim,:].transpose() # trajectory observations with velocities
        
        self.dk = jacfwd(self.k)
        self.ddk = jacfwd(self.dk)
        
        
    def kZ(self,x):
            out = jnp.zeros(len(self.Z))
            #out = lax.fori_loop(0,len(self.Z), lambda j, out: ops.index_update(out,ops.index[j], self.k(x,self.Z[j]) ),out)
            out = lax.fori_loop(0,len(self.Z), lambda j, out: out.at[j].set(self.k(x,self.Z[j]) ),out)
            return out
    
    @partial(jit, static_argnums=(0,))
    def dkZ(self,x):
        return jacfwd(self.kZ)(x)
    
    @partial(jit, static_argnums=(0,))
    def ddkZ(self,x):
        return jacfwd(self.dkZ)(x)
        
    def d1kZ(self,x):
        return self.dkZ(x)[:,:self.dim]
    
    def d2kZ(self,x):
        return self.dkZ(x)[:,self.dim:]
    
    
    # for training data consistency equations
    @partial(jit, static_argnums=(0,))
    def el(self,a,y):
		
        z = a[:-self.dim]
		
        DK = self.dk(z,y)
        Hess = self.ddk(z,y)
		
        q = z[:self.dim]
        qdot = z[self.dim:]
        qdot2 = a[-self.dim:]
		
        return Hess[self.dim:,:self.dim] @ qdot + Hess[self.dim:,self.dim:] @ qdot2 - DK[:self.dim]
		
	
    def row(self,a):
        rw = jnp.zeros((len(self.Z),self.dim))
        #body_fun = lambda j,rw: ops.index_update(rw,ops.index[j],self.el(a,self.Z[j]))
        body_fun = lambda j,rw: rw.at[j].set(self.el(a,self.Z[j]))
        return lax.fori_loop(0,len(self.Z),body_fun,rw)
		
    def lhs_data_consistency(self):
        A = jnp.transpose(self.data_train)
        mt = jnp.zeros((len(A),len(self.Z),self.dim))
        #body_fun = lambda j,mt: ops.index_update(mt,ops.index[j],self.row(A[j]))
        body_fun = lambda j,mt: mt.at[j].set(self.row(A[j]))
        mt = lax.fori_loop(0,len(A),body_fun,mt)
		
        return jnp.reshape(mt,(self.dim*len(self.data_train[0,:]),len(self.Z)))

    
    
    def lhs_non_trivial(self):
    
    
        # non-triviality condition in training
        verts=jnp.array(list(itertools.product([0, 1], repeat=2*self.dim)),dtype=jnp.float64)
        normalise = 1./(math.factorial(self.dim) * 2**(2*self.dim)) * jnp.sum(jnp.sum(jnp.array([ self.d2kZ(v) for v in verts]),0),1)
        
        # normalise absolute value
        z0 = jnp.zeros(2*self.dim)
        normaliseTotal = self.kZ(z0)
        
        return jnp.vstack([normalise,normaliseTotal])

    
    def train(self,sympl_std_vol=1):
        
        print('compute data consistency equations and non-triviality conditions')
        
        lhs = jnp.vstack([self.lhs_data_consistency(),self.lhs_non_trivial()])
        rhs = jnp.zeros(lhs.shape[0])
        #rhs = ops.index_update(rhs,ops.index[-2], sympl_std_vol) # symplectic volume of unit simplex normalised to sympl_std_vol
        rhs = rhs.at[-2].set(sympl_std_vol) # symplectic volume of unit simplex normalised to sympl_std_vol
        
        # solve minimal norm / least square
        print('solve linear system dimensions: '+str(lhs.shape))
        kinvL,res,rank, _ =jnp.linalg.lstsq(lhs,rhs,rcond=None)
        
        self.kinvL = kinvL
        self.res = res
        self.rank = rank
        
        return kinvL,res,rank
        
    @partial(jit, static_argnums=(0,))
    def L(self,x):

        try:
            self.kinvL
        except AttributeError:
            print("GP has not been trained")
            
        return self.kZ(x) @ self.kinvL
    
    
    @partial(jit, static_argnums=(0,))
    def HamiltonianJet(self,x):
    
        # Hamiltonian in (q,qdot)
    
        L_insq = lambda qdot: self.L(jnp.hstack([x[:self.dim],qdot]))
        p = jacfwd(L_insq)(x[self.dim:])
    
        return jnp.dot(x[self.dim:],p) - self.L(x)
    
    
    

