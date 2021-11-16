from jax.config import config
config.update("jax_enable_x64", True)

from jax import ops,jit, jacfwd
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial

from NewtonLAX import NewtonLAX

class VarMPIntegrator:
    
    def __init__(self,h,L):
        self.h = h
        self.L = L
        
        
    def Lqdot(self,x):
        
        # dL/d(qdot)
        
        dim = int(len(x)/2)
        q = x[:dim]
        qdot = x[dim:]
        f = lambda qdot: self.L(jnp.block([q,qdot]))
        
        return jacfwd(f)(qdot)
    
    
    
        
    def mp(self,q0,q1):
        return jnp.block([1/2*(q0+q1),1/self.h*(q1-q0)])
    
    
    def Ldisc(self,q0,q1):
        return self.h*self.L(self.mp(q0,q1))        # discrete Lagrangian based on L
    
    def Ldisc_d1(self,q0,q1):
        f = lambda q0: self.Ldisc(q0,q1)
        return jacfwd(f)(q0)  # derivative w.r.t. first variable
    
    def Ldisc_d2(self,q0,q1):
        f = lambda q1: self.Ldisc(q0,q1)
        return jacfwd(f)(q1)  # derivative w.r.t. second variable
    
    # mixed second derivative
    def Ldisc_d1d2(self,q0,q1):
        f = lambda q1: self.Ldisc_d1(q0,q1)
        return jacfwd(f)(q1)
            
    # computation of momenta using discrete Lagrangian
    def p0_fun(self,q0,q1):
        return -self.Ldisc_d1(q0,q1)
    
    @partial(jit, static_argnums=(0,))
    def p1_fun(self,q0,q1):
        return self.Ldisc_d2(q0,q1)
    
    def DEL(self,q0,q1,q2):
        return self.Ldisc_d1(q1,q2) + self.Ldisc_d2(q0,q1)
    
    
    @partial(jit, static_argnums=(0,3,4))
    def q2_solve(self,q0,q1,minstepsize=1e-10,maxiter=100):
        # computation q2 using DEL

        #print("compiling q2_solve")

        p1 = self.p1_fun(q0,q1)
        obj = lambda q2: self.Ldisc_d1(q1,q2)+p1

        return NewtonLAX(obj,2*q1-q0,minstepsize,maxiter)
    
    def q1_solve(self,q0,p0):
        
        obj = lambda q1: self.p0_fun(q0,q1)-p0
        return NewtonLAX(obj,q0)
        
    @partial(jit, static_argnums=(0,))
    def qdot_solve(self,q,p,qdot_guess):
                
        obj = lambda qdot: self.Lqdot(jnp.block([q,qdot]))-p  
        return NewtonLAX(obj,qdot_guess)
    
    def flow(self,q0,p0):
        
        q1 = self.q1_solve(q0,p0)
        p1 = self.p1_fun(q0,q1)
        
        return q1,p1
    
    def move(self,x0):
        
        dim = int(len(x0)/2)
        
        p0 = self.Lqdot(x0)
        q1,p1 = self.flow(x0[:dim],p0)
        q1dot = self.qdot_solve(q1,p1,x0[dim:])
        
        return jnp.block([q1,q1dot])
    
    
    def trj_fill(self,trj,minstepsize=1e-10,maxiter=100):
        
        n = len(trj[0])
        
        for j in tqdm(range(n-2)):
            trj = ops.index_update(trj,ops.index[:,j+2],self.q2_solve(trj[:,j],trj[:,j+1],minstepsize=minstepsize,maxiter=maxiter))
            
        return trj
    
    def motion(self,q0,q0dot,n):
        
        dim = len(q0)
        p0 = self.Lqdot(jnp.block([q0,q0dot]))
        q1 = self.q1_solve(q0,p0)
        trj0 = jnp.hstack([jnp.array([q0]).transpose(),jnp.array([q1]).transpose(),jnp.zeros((dim,n-2))])
        
        return self.trj_fill(trj0)
    
    
    def conjugate_momenta(self,trj):
        
        n = len(trj[0])
        
        p0 = self.p0_fun(trj[:,0],trj[:,1])
        
        ps = jnp.zeros(trj.shape)
        ps = ops.index_update(ps,ops.index[:,0],p0)
        
        for j in tqdm(range(1,n)):
            ps = ops.index_update(ps,ops.index[:,j],self.p1_fun(trj[:,j-1],trj[:,j]))
            
        return ps
        
    def velocities(self,trj):
        
        n = len(trj[0])
        
        print("compute conjugate momenta")
        ps = self.conjugate_momenta(trj)
        
        print("compute velocities")
        
        v = jnp.zeros(ps.shape)
        
        # j= 0
        v_guess = (trj[:,1]-trj[:,0])/self.h
        v = ops.index_update(v,ops.index[:,0],self.qdot_solve(trj[:,0],ps[:,0],v_guess))
        
        # j = 1,...,n-2
        for j in tqdm(range(1,n-1)):
            
            v_guess = (trj[:,j+1]-trj[:,j-1])/(2*self.h)
            v = ops.index_update(v,ops.index[:,j],self.qdot_solve(trj[:,j],ps[:,j],v_guess))
            
        # j = n-1
        v_guess = (trj[:,n-1]-trj[:,n-2])/self.h
        v = ops.index_update(v,ops.index[:,n-1],self.qdot_solve(trj[:,n-1],ps[:,n-1],v_guess))
        
        return v,ps
    
    
    def Hamiltonian(self,trj,velocities,momenta):
        
        trj_jet=jnp.vstack([trj,velocities])
        LVals=jnp.array([self.L(x) for x in trj_jet.transpose() ])
        
        return jnp.sum(velocities*momenta,0) - LVals

        
    
