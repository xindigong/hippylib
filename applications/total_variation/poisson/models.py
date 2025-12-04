import ufl
import dolfin as dl

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp

from utils import parameter2NoisyObservations

# boundaries for the unit square
def u0_boundary(x, on_boundary):
    return on_boundary

class PoissonVarf:
    def __init__(self, f):
        self.f = f

    def __call__(self, u, m, p):
        return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - self.f*p*ufl.dx        

class PoissonBox():
    def __init__(self, N):
        self.SEP = "\n"+"#"*80+"\n"  # for printing
        self.N = N
        
    def setupMesh(self):
        mesh = dl.UnitSquareMesh(self.N, self.N)
        self.mesh = mesh
        self.rank = dl.MPI.rank(self.mesh.mpi_comm())
        self.nproc = dl.MPI.size(self.mesh.mpi_comm())
        
    def setupFunctionSpaces(self):
        Vh2 = dl.FunctionSpace(self.mesh, 'Lagrange', 2)
        Vh1 = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        # Vh = [Vh2, Vh1, Vh2]
        Vh = [Vh1, Vh1, Vh1]
        self.Vh = Vh
        ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
        if self.rank == 0:
            print(self.SEP, "Set up the mesh and finite element spaces", self.SEP)
            print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")
    
    def setupPDE(self):
        # variational form
        f = dl.Constant(1.0)
        pde_varf = PoissonVarf(f)
        
        # boundary conditions
        zero = dl.Constant(0.0)
        bc = dl.DirichletBC(self.Vh[hp.STATE], zero, u0_boundary)
        bc0 = bc
        
        # pde problem
        pde = hp.PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)
        self.pde = pde
        
    def setupTrueParameter(self):
        mtrue_exp = dl.Expression('1.0 + 7.0*(x[0]<=0.8)*(x[0]>=0.2)*(x[1]<=0.8)*(x[1]>=0.2)', degree=1)
        mtrue = dl.interpolate(mtrue_exp, self.Vh[hp.PARAMETER])
        self.mtrue = mtrue
        
    def generateObservations(self, noise_level):
        self.p2o = parameter2NoisyObservations(self.pde, self.mtrue.vector(), noise_level)
        self.p2o.generateNoisyObservations()

    def setupMisfit(self):
        self.misfit = hp.ContinuousStateObservation(Vh=self.Vh[hp.STATE],
                                       dX=ufl.dx,
                                       data=self.p2o.noisy_data,
                                       noise_variance=self.p2o.noise_std_dev**2,
                                       bcs=self.pde.bc)


class MultiPoissonVarf:
    def __init__(self, f, idx):
        self.f = f
        self.idx = idx

    def __call__(self, u, m, p):
        # Note u, p are unique for specific problem, m is a vector with all parameters.
        current_m = ufl.split(m)[self.idx]  # index parameter for the current problem        
        return ufl.exp(current_m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - self.f*p*ufl.dx
    
class MultiPoissonBox():
    def __init__(self, npde, nh):
        self.SEP = "\n"+"#"*80+"\n"  # for printing
        self.npde = npde
        self.nh = nh
        
    def setupMesh(self):
        mesh = dl.UnitSquareMesh(self.nh, self.nh)
        self.mesh = mesh
        self.rank = dl.MPI.rank(self.mesh.mpi_comm())
        self.nproc = dl.MPI.size(self.mesh.mpi_comm())
        
    def setupFunctionSpaces(self):
        Vhu = dl.FunctionSpace(self.mesh, 'Lagrange', 1)  # for state / adjoint
        Vhm = dl.VectorFunctionSpace(self.mesh, 'Lagrange', 1, dim=self.npde)  # for all parameters
        self.Vhm0 = dl.FunctionSpace(self.mesh, 'Lagrange', 1)  # for a single parameter
        self.Vh = [Vhu, Vhm, Vhu]
        ndofs = [self.Vh[hp.STATE].dim(), self.Vh[hp.PARAMETER].dim(), self.Vh[hp.ADJOINT].dim()]
        if self.rank == 0:
            print(self.SEP, "Set up the mesh and finite element spaces", self.SEP)
            print(f"Number of dofs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")
        
        # set up a function assigner
        self.assigner = dl.FunctionAssigner(self.Vh[hp.PARAMETER], [self.Vhm0]*self.npde)
    
    def setupPDE(self):
        # variational form
        f = dl.Constant(1.0)
        
        # boundary conditions
        zero = dl.Constant(0.0)
        bc = dl.DirichletBC(self.Vh[hp.STATE], zero, u0_boundary)
        bc0 = bc
        
        # setup the pde problems
        pdes = []
        for i in range(self.npde):
            pde_varf = MultiPoissonVarf(f, i)
            pde = hp.PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)
            pdes.append(pde)
        
        self.pde = hp.MultiPDEProblem(pdes)
        
    def get_component(self, out:dl.Function, x:dl.Function, idx:float):
        """Get component of a vector and assign to a function space.

        Args:
            out (dl.Function): Function to assign to.
            x (dl.Function): Vector / Mixed Element Function to draw component from.
            idx (float): Index of component to assign.
            
        Returns:
            None (write to out)
        """
        fa = dl.FunctionAssigner(self.Vhm0, self.Vh[hp.PARAMETER].sub(idx))
        fa.assign(out, x)
        
    def split_component(self, x:dl.Function, idx:float):
        """Split a vector into a component.

        Args:
            x (dl.Function): Vector / Mixed Element Function to draw component from.
            idx (float): Index of component to grab.

        Returns:
            dl.Function: Function representing the component.
        """
        out = x.sub(idx, deepcopy=True)
        return out
    
class splitCircle(dl.UserExpression):
    """Expression implementing a circle with different values on the left and right sides.
    """
    def __init__(self, cx, cy, r, vl, vr, vo, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.cx = cx  # center x-coordinate
        self.cy = cy  # center y-coordinate
        self.vl = vl  # left value
        self.vr = vr  # right value
        self.vo = vo  # outside value
        
    def eval_cell(self, values, x, cell):        
        if (pow(x[0]-self.cx,2)+pow(x[1]-self.cy,2) < pow(self.r,2)):
            if (x[0] < self.cx):
                values[0] = self.vl
            else:
                values[0] = self.vr
        else:
            values[0] = self.vo
