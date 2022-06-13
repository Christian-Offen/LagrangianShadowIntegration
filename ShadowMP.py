from jax import ops, lax, jacfwd, jacrev, jit, jvp, grad
import itertools
import math
from functools import partial
from tqdm import tqdm

from NewtonLAX import *

class GPMP:
    def __init__(self,kernel,h,DataPairs,DataTriples):
        self.k = kernel
        self.h = h
        self.DataPairs = DataPairs
        self.DataTriples = DataTriples
        self.dk = jacfwd(self.k)
        self.ddk = jacfwd(self.dk)
        self.dim = len(DataPairs[0]) # dimension of state space
        self.mp_d = jnp.block([[1/2*jnp.identity(self.dim), -1/self.h*jnp.identity(self.dim)],[1/2*jnp.identity(self.dim), 1/self.h*jnp.identity(self.dim)]]) # derivative of midpoint approximation 
        self.Z = jnp.hstack( [ 1/2*(self.DataPairs[:,:,0]+self.DataPairs[:,:,1]) ,  1/self.h*(self.DataPairs[:,:,1]-self.DataPairs[:,:,0]) ]) # points with maximal information
        
    # midpoint for midpoint discretisation
    def mp(self,q0,q1):
        return jnp.hstack([1/2*(q0+q1),1/self.h*(q1-q0)])
    
    
        
    # train GP     
    def train(self):
        
        print('start training')
        Z = self.Z # points with maximal information
                
        def kZ(x):
            out = jnp.zeros(len(Z))
            out = lax.fori_loop(0,len(Z), lambda j, out: out.at[j].set(self.k(x,Z[j]) ),out)
            return out

        dkZ = jacfwd(kZ)
        
        d2kZ = lambda x: dkZ(x)[:,self.dim:]
        
        def lhs0(qqq):
    
            m1 = self.mp(qqq[0],qqq[1])
            m2 = self.mp(qqq[1],qqq[2])

            dkZ2 = dkZ(m2)
            dkZ1 = dkZ(m1)

            return self.mp_d[:self.dim] @ dkZ2.transpose() + self.mp_d[self.dim:] @ dkZ1.transpose()
        
        # non-triviality condition
        print('calculate non-triviality condition')
        verts=jnp.array(list(itertools.product([0, 1], repeat=2*self.dim)),dtype=jnp.float64)
        normalise = 1./(math.factorial(self.dim) * 2**(2*self.dim)) * jnp.sum(jnp.sum(jnp.array([ d2kZ(v) for v in verts]),0),1)
        
        # normalise
        z0 = jnp.zeros(2*self.dim)
        normaliseTotal = kZ(z0)

        # data consistency equations
        print('calculate data consistency equations')
        lhsDEL = jnp.zeros((self.DataTriples.shape[0],self.dim,Z.shape[0]))
        bodyfun = lambda j, lhsDEL: lhsDEL.at[j].set(lhs0(self.DataTriples[j].transpose()))
        lhsDEL = lax.fori_loop(0,len(self.DataTriples), bodyfun,lhsDEL)
        lhsDEL=lhsDEL.reshape(self.DataTriples.shape[0]*self.dim,Z.shape[0])
        lhs = jnp.vstack([lhsDEL,normalise,normaliseTotal])
        
        
        rhs = jnp.zeros(lhs.shape[0])
        rhs = rhs.at[-2].set(1.) # symplectic volume of unit simplex normalised to 1
        
        # solve minimal norm / least square
        print('solve linear system dimensions: '+str(lhs.shape))
        kinvL,res,rank,singVals=jnp.linalg.lstsq(lhs,rhs,rcond=None)
        
        self.kinvL = kinvL
        self.res = res
        self.rank = rank
        
        return kinvL,res,rank

    
    # inverse modified Lagrangian and derivatives

    #Lmi = lambda y: np.array([k(y,qq) for qq in Z]) @ kinvL    
    # inverse modified Lagrangian
    def Lmi(self,y):

        kyqq = jnp.zeros(len(self.Z))
        bodyfun = lambda j,kyqq0: kyqq0.at[j].set(self.k(y,self.Z[j]))
        kyqq = lax.fori_loop(0,len(self.Z), bodyfun,kyqq)

        return kyqq @ self.kinvL

    def Lmi_d(self,z):
        return jacfwd(self.Lmi)(z) # gradient
    
    def Lmi_dd(self,z):
        return jacfwd(self.Lmi_d)(z) # Hessian
    
    def Lmi_ddd(self,z):
        return jacfwd(self.Lmi_dd)(z) # 3rd derivative
    
    def Lmi_dddd(self,z):
        return jacfwd(self.Lmi_ddd)(z) # 4rth derivative


    # discrete Lagrangian based on inverse modified Lagrangian and derivatives
    def Ldisc(self,q0,q1):
        return self.h*self.Lmi(self.mp(q0,q1))        # discrete Lagrangian based on inverse modified Lagrangian

    def Ldisc_d1(self,q0,q1):
        return self.mp_d[:self.dim] @ (self.h*self.Lmi_d(self.mp(q0,q1)))  # derivative w.r.t. first variable
    
    def Ldisc_d2(self,q0,q1):
        return self.mp_d[self.dim:] @ (self.h*self.Lmi_d(self.mp(q0,q1)))  # derivative w.r.t. second variable

    # mixed second derivative
    def Ldisc_d1d2(self,q0,q1):
        return self.mp_d[self.dim:] @ (self.h*self.Lmi_dd(self.mp(q0,q1))) @ self.mp_d[:self.dim]

    ## Hessian 
    def Ldisc_dd(self,q0,q1):
        return self.mp_d.transpose() @ (self.h*self.Lmi_dd(self.mp(q0,q1))) @ self.mp_d
    
    # computation of momenta using discrete Lagrangian
    @partial(jit, static_argnums=(0,))
    def p0_fun(self,q0,q1):
        print('compiling discrete conjugate momentum formula')
        return -self.Ldisc_d1(q0,q1)
    
    def p1_fun(self,q0,q1):
        return self.Ldisc_d2(q0,q1)
    
    def DEL(self,q0,q1,q2):
        return self.Ldisc_d1(q1,q2) + self.Ldisc_d2(q0,q1)
    
    
    @partial(jit, static_argnums=(0,))
    def q2_solve(self,q0,q1):
        # computation q2 using DEL

        print("compiling q2_solve")

        p1 = self.p1_fun(q0,q1)

        obj = lambda q2: self.Ldisc_d1(q1,q2)+p1
        obj_prime = jacfwd(obj)

        q2 = MyNewtonLAX(obj,q1,fprime=obj_prime)

        return q2
    
    # recovered Lagrangian from backward error analysis, two spacial dimensions
    @partial(jit, static_argnums=(0,))
    def Lrec2(self,y):

        print('JIT compiling BEA formula for Lagrangian L (or a derivative of it)')

        y1d = y[2]
        y2d = y[3]

        L_0 = self.Lmi(y)

        # compute jet
        gradL = self.Lmi_d(y)
        HessL = self.Lmi_dd(y)

        # assign variables compatible with formulas spit out by Mathematica
        L1000 = gradL[0]
        L0100 = gradL[1]
        L0010 = gradL[3]
        L0001 = gradL[4]

        L2000 = HessL[0,0]
        L0200 = HessL[1,1]
        L0020 = HessL[2,2]
        L0002 = HessL[3,3]

        L1100 = HessL[0,1]
        L1010 = HessL[0,2]
        L1001 = HessL[0,3]
        L0110 = HessL[1,2]
        L0101 = HessL[1,3]
        L0011 = HessL[2,3]


        # compute higher order correction terms
        L_1 = 0

        L_2 = -1/24*(2*L0011*(L0100 - L1001*y1d - L0101*y2d)*(-L1000 + L1010*y1d + L0110*y2d) + L0002*(-L1000 + L1010*y1d + L0110*y2d)**2 + L0020*(L0100**2 + L1001**2*y1d**2 - L0002*L2000*y1d**2 + 2*L0101*L1001*y1d*y2d - 2*L0002*L1100*y1d*y2d + L0101**2*y2d**2 - L0002*L0200*y2d**2 - 2*L0100*(L1001*y1d + L0101*y2d)) + L0011**2*(L2000*y1d**2 + y2d*(2*L1100*y1d + L0200*y2d)))/(L0011**2 - L0002*L0020)


        # compute recovered Lagrangian truncated to order 0,2

        Lrecov2 = L_0 + self.h**2*L_2



        return Lrecov2
    
    
    def Lrec2_d(self,z):
        return jacfwd(self.Lrec2)(z)
    
    def Lrec2_dd(self,z):
        return jacfwd(self.Lrec2_d)(z)
    
    
    def compute_motion(self,q0,q0dot,n):
        
        # compute q1 from q0,q0dot

        p0  = self.Lrec2_d(jnp.hstack([q0,q0dot]))[self.dim:]
        obj = lambda q1: self.p0_fun(q0,q1) - p0
        q1  = MyNewtonLAX(obj,q0,jacfwd(obj))
        
        
        # compute trajectory
        trj = jnp.zeros((self.dim,n+1))
        trj = trj.at[:,0].set(q0)
        trj = trj.at[:,1].set(q1)

        body_update = lambda j, trj: trj.at[:,j+2].set(self.q2_solve(trj[:,j],trj[:,j+1]) )
        trj = lax.fori_loop(0,len(trj[0])-2, body_update,trj)

        return trj
    
    def DELtrj(self,trj):
		# check discrete EL equation on trajectory
		# output: 1-norm of all errors

        err_q2_solver = jnp.zeros((self.dim,len(trj[0])-2))
		
        for j in tqdm(range(len(trj[0])-2)):
                err_q2_solver = err_q2_solver.at[:,j].set(self.DEL(trj[:,j],trj[:,j+1],trj[:,j+2]))

        return jnp.sum(jnp.abs(err_q2_solver)) # 1-norm of internal error (worst case: all errors sum up)

        
    # approximate recovery of qdot based on truncation of recovered L

    # truncation to second order
    @partial(jit, static_argnums=(0,))
    def qdotRecover2(self,q,p,qdotGuess):

        print("compiling BEA formula for velocity")

        obj = lambda qdot: self.Lrec2_d(jnp.hstack([q,qdot]))[self.dim:] - p
        obj_prime = lambda qdot: self.Lrec2_dd(jnp.block([q,qdot]))[self.dim:,self.dim:]

        return MyNewtonLAX(obj,qdotGuess,obj_prime)

    # computes derivatives to trajectory using the Lagrangian framework
    def motionqdot(self,trj):
        qdotRectrj = jnp.zeros((2,len(trj[0])-1))
        qdotRectrj = qdotRectrj.at[:,0].set(self.qdotRecover2(trj[:,0],self.p0_fun(trj[:,0],trj[:,1]),jnp.array([0.,0.])) )
        for j in tqdm(range(1,len(trj[0])-1)):    
            qdotRectrj = qdotRectrj.at[:,j].set(self.qdotRecover2(trj[:,j],self.p0_fun(trj[:,j],trj[:,j+1]),qdotRectrj[:,j-1]))
            
        qdotfinal = self.qdotRecover2(trj[:,-1],self.p1_fun(trj[:,-2],trj[:,-1]),qdotRectrj[:,-1])
        qdotRectrj=jnp.block([ qdotRectrj, jnp.array([qdotfinal]).transpose()  ])
        
        return qdotRectrj
    
    # compute Hamiltonian (energy) along a trajectory using 2nd order truncation of identified L
    def energy_motion(self,trj,trjqdot):

        qqdot_trj = jnp.vstack([trj,trjqdot])

        p_trj= [self.p0_fun(trj[:,j],trj[:,j+1]) for j in range(len(trj[0])-1)]
        p_trj.append(self.p1_fun(trj[:,-2],trj[:,-1]))
        p_trj = jnp.array(p_trj)
        L_trj=jnp.array([self.Lrec2(qqdot_trj[:,j]) for j in tqdm(range(0,len(trj[0])))])
        HRec2 = jnp.sum(trjqdot*p_trj.transpose(),axis=0) - L_trj.flatten()

        return HRec2
    
    # compute Hamiltonian to z using 2nd order truncation of identified L
    @partial(jit, static_argnums=(0,))
    def Hamiltonian(self,z):
    
        print('JIT compiling Hamiltonian function')
        p2 = lambda z: self.Lrec2_d(z)[self.dim:]
    
        return jnp.dot(z[self.dim:],p2(z)) - self.Lrec2(z)
    
    
    
    
