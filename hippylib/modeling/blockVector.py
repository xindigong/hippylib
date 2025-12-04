# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2025, The University of Texas at Austin 
# & University of California--Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import hippylib as hp

class BlockVector:
    """
    A class to store multiple vectors.
    """
    def __init__(self, data):
        self.data = data
    
    @property
    def nv(self):
        return len(self.data)
    
    @property
    def isHierarchical(self):
        if hasattr(self.data[0], 'nv'):
            return True
        else:
            return False
    
    @classmethod
    def fromOther(cls, other):
        data = []
        for d in other.data:
            data.append(d.copy())
            
        return cls(data)             
                    
    @classmethod
    def fromVector(cls, v, Nv):
        data = []
        for i in range(Nv):
            data.append(v.copy())
            
        return cls(data)
    
    @classmethod
    def fromFunctionSpace(cls, Vh, Nv):
        data = []
        for i in range(Nv):
            data.append(dl.Function(Vh).vector() )
            
        return cls(data)
            
    @staticmethod
    def fromFunctionSpaces(cls, Vhs):
        data = []
        for Vh in Vhs:
            data.append(dl.Function(Vh).vector() )
            
        return cls(data)
        
            
    def randn_perturb(self,std_dev):
        """
        Add a random perturbation :math:`\eta_i \sim \mathcal{N}(0, \mbox{std_dev}^2 I)`
        to each of the snapshots.
        """
        for d in self.data:
            if hasattr(d, 'randn_perturb'):
                d.randn_perturb(std_dev)
            else:
                hp.parRandom.normal_perturb(std_dev, d)

    
    def axpy(self, a, other):
        """
        Compute :code:`x = x + a*other` snapshot per snapshot.
        """
        assert self.nv == other.nv
        for i in range(self.nv):
            self.data[i].axpy(a,other.data[i])
        
    def zero(self):
        """
        Zero out each subvector.
        """
        for d in self.data:
            d.zero()
                        
    def __imul__(self, alpha):
        """
        Scale by scalar
        """
        for d in self.data:
            d *= alpha
        return self
    
    def copy(self):
        """
        Return a copy 
        """
        return BlockVector.fromOther(self)
    