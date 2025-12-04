import sys
import os
from typing import List

import numpy as np
import dolfin as dl
import pyvista as pv

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp

def rprint(comm, *args, **kwargs):
    """Print only on rank 0."""
    if comm.rank == 0:
        print(*args, **kwargs)


def samplePrior(prior):
    """Wrapper to sample from a :code:`hIPPYlib` prior.

    Args:
        :code:`prior`: :code:`hIPPYlib` prior object.

    Returns:
        :code:`dl.Vector`: noisy sample of prior.
    """
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue


class parameter2NoisyObservations:
    def __init__(self, pde:hp.PDEProblem, m:dl.Vector, noise_level:float=0.02, B:hp.PointwiseStateObservation=None):
        self.pde = pde
        self.m = m
        self.noise_level = noise_level
        self.B = B
        
        # set up vector for the data
        self.true_data = self.pde.generate_state()
        
        self.noise_std_dev = None
        
    def generateNoisyObservations(self):
        # generate state, solve the forward problem
        utrue = self.pde.generate_state()
        x = [utrue, self.m, None]
        self.pde.solveFwd(x[hp.STATE], x)
        
        # store the true data
        self.true_data.axpy(1., x[hp.STATE])
        
        # apply observation operator, determine noise
        if self.B is not None:
            self.noisy_data = dl.Vector(self.B.mpi_comm())  # shape vector to match B
            self.B.init_vector(self.noisy_data, 0)          # initialize vector
            self.noisy_data.axpy(1., self.B*x[hp.STATE])
        else:
            self.noisy_data = self.pde.generate_state()
            self.noisy_data.axpy(1., x[hp.STATE])
        
        MAX = self.noisy_data.norm("linf")
        self.noise_std_dev = self.noise_level * MAX
        
        # generate noise
        noise = dl.Vector(self.noisy_data)
        noise.zero()
        hp.parRandom.normal(self.noise_std_dev, noise)
        
        # add noise to measurements
        self.noisy_data.axpy(1., noise)


def add_noise_to_observations(true_data:dl.Vector, noise_level:float, B:hp.PointwiseStateObservation=None)->dl.Vector:
    """Add noise to a vector.

    Args:
        true_data (dl.Vector): True data.
        noise_level (float): Noise level.
        B (hp.PointwiseStateObservation, optional): Pointwise observation operator. Defaults to None.

    Returns:
        (dl.Vector): Noisy data.
        (float): Noise standard deviation.
    """
    
    # apply observation operator
    if B is not None:
        noisy_data = dl.Vector(B.mpi_comm())
        B.init_vector(noisy_data, 0)
        noisy_data.axpy(1., B*true_data)
    else:
        noisy_data = dl.Vector(true_data)
        noisy_data.axpy(1., true_data)
    
    # determine noise
    MAX = noisy_data.norm("linf")
    noise_std_dev = noise_level * MAX
    
    # generate noise
    noise = dl.Vector(noisy_data)
    noise.zero()
    hp.parRandom.normal(noise_std_dev, noise)
    
    # add noise to measurements
    noisy_data.axpy(1., noise)
    
    return noisy_data, noise_std_dev
