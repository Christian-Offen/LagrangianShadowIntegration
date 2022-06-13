import jax.numpy as jnp
from jax import lax, random

from skopt.space import Space
from skopt.sampler import Halton

from LeapfrogJAX import LP


def GenerateTrainingData(spacedim,forces,n,n_l,h,n_h = 800, NoiseSigma=0):
    
    ## input
    # spacedim = list of tuples
    # forces = lambda function, vectorised
    # n = number of trajectories to be created
    # n_l = length of trajectories
    # h = snapshot time
    # n_h = number of intermediate steps in Leapfrog scheme between snapshots
    
    ## output
    # return array of size (len(spacedim),n,n_l+1) with trajectory data.
    # data[:,k,:] is the data of the kth trajectory
    
    space = Space(spacedim)
    dim = int(len(spacedim)/2) # dimension of state space
    h_gen = h/n_h

    # Compute flow map from Halton sequence to generate learnin data
    halton = Halton()
    start = halton.generate(space, n)
    start = jnp.array(start).transpose()

    data = jnp.zeros((len(spacedim),n,n_l+1))
    data = data.at[:,:,0].set(start)
    #data = ops.index_update(data,ops.index[:,:,0],start)    
    data = lax.fori_loop(0,n_l, lambda j,data: data.at[:,:,j+1].set(LP(data[:,:,j],n_h,h,forces)) , data)
    #data = lax.fori_loop(0,n_l, lambda j,data: ops.index_update(data,ops.index[:,:,j+1],LP(data[:,:,j],n_h,h,forces) ) , data)
    
     # add random noise
    noise = NoiseSigma*random.normal(random.PRNGKey(0),data.shape)
    data = data + noise    
    
    # organise data into consecutive tuples and triples
    DataPairs = [data[:dim,k,j:j+2] for k in range(n) for j in range(n_l)]
    DataPairs = jnp.array(DataPairs)

    DataTriples = [data[:dim,k,j:j+3] for k in range(n) for j in range(n_l-1)]
    DataTriples = jnp.array(DataTriples)

    return data,DataPairs,DataTriples

