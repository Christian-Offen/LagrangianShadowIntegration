from jax.config import config
config.update("jax_enable_x64", True)

from jax import value_and_grad, hessian, jit
import jax.numpy as jnp
from functools import partial

def modL(L,h,y):

	# modified Lagrangian obtained by variational backward error analysis

	q  = y[0]
	qd = y[1]

	L_0 = L(y)

	# compute jet
	L_0, gradL = value_and_grad(L)(y)
	HessL = hessian(L)(y)


	# assign variables compatible with formulas spit out by Mathematica
	L10 = gradL[0]
	L01 = gradL[1]

	L20 = HessL[0,0]
	L11 = HessL[0,1]
	L02 = HessL[1,1]


	# compute higher order correction term
	L_2 = (-(L20*qd**2) + (L10 - L11*qd)**2/L02)/24.

	# modified Lagrangian truncated to second order
	L2 = L_0 + h**2*L_2

	return L2

@partial(jit, static_argnums=(0,))
def modH(L,h,y):
	
	# modified Hamiltonian
	
	Lqdot = lambda qdot: modL(L,h,jnp.array([y[0],qdot]))
	Lvalue, p = value_and_grad(Lqdot)(y[1])
	
	return y[1]*p-Lvalue
	
