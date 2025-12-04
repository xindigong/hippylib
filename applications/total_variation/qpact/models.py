import sys
import os

import ufl
import dolfin as dl

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp


def rprint(comm, *args, **kwargs):
    """Print only on rank 0."""
    if comm.rank == 0:
        print(*args, **kwargs)


class DiffusionApproximation:
    def __init__(self, D:dl.Constant, u0:dl.Constant):
        """Define the forward model for the diffusion approximation to radiative transfer equations.

        Args:
            D (dl.Constant): diffusion coefficient 1/mu_eff with mu_eff = sqrt(3 mu_a (mu_a + mu_ps) ), where mu_a is the unknown absorption coefficient, and mu_ps is the reduced scattering coefficient_description_
            u0 (dl.Constant): Incident fluence (Robin condition)
        """
        
        self.D = D
        self.u0 = u0
        
    def __call__(self, u:dl.Function, m:dl.Function, p:dl.Function) -> ufl.form.Form:
        return ufl.inner(self.D*ufl.grad(u), ufl.grad(p))*ufl.dx + \
               ufl.exp(m)*ufl.inner(u, p)*ufl.dx + \
               dl.Constant(.5)*ufl.inner(u - self.u0, p)*ufl.ds


class DiffusionApproximationMisfitForm:
    def __init__(self, d:dl.Function, sigma2:float):
        """Constructor for the PACT misfit form.

        Args:
            d (dl.Function): Data.
            sigma2 (float): Variance of the data.
            m0 (dl.Function): Parameter mean.
        """
        self.sigma2 = sigma2
        self.d = d
        
    def __call__(self, u, m):
        return (dl.Constant(0.5/self.sigma2))*ufl.inner(u*ufl.exp(m) - self.d, u*ufl.exp(m) - self.d)*ufl.dx

class PDEExperiment(object):
    """Base class for PDE experiments.
    """
    
    def generate_state(self):
        """ Return a vector in the shape of the state. """
        raise NotImplementedError("Child class should implement method generate_state")
    
    def setupMesh(self):
        """ Load / construct the mesh and set the self.mesh attribute. """
        raise NotImplementedError("Child class should implement method setupMesh")

    def setupFunctionSpaces(self):
        """ Set up the appropriate function spaces and set the self.Vh attribute. """
        raise NotImplementedError("Child class should implement method setupFunctionSpaces")

    def setupPDE(self):
        """ Instantiate the PDE problem and set the self.pde attribute. """
        raise NotImplementedError("Child class should implement method setupPDE")


class qPACT_DA(PDEExperiment):
    """qPACT problem with Diffusion Approximation.
    """
    def __init__(self, comm, mesh_fpath:str):
        self.SEP = "\n"+"#"*80+"\n"  # for printing
        self.comm = comm
        self.mesh_fpath = mesh_fpath
        
    def setupMesh(self):
        self.mesh = dl.Mesh(self.comm)
        with dl.XDMFFile(self.mesh_fpath) as fid:
            fid.read(self.mesh)
    
    def setupFunctionSpaces(self):
        Vhu = dl.FunctionSpace(self.mesh, 'Lagrange', 1)  # for state / adjoint
        Vhm = dl.FunctionSpace(self.mesh, 'Lagrange', 1)  # for absorbance parameters
        self.Vh = [Vhu, Vhm, Vhu]
        
        # report ndofs
        ndofs = [self.Vh[hp.STATE].dim(), self.Vh[hp.PARAMETER].dim(), self.Vh[hp.ADJOINT].dim()]
        if self.comm.rank == 0:
            print(self.SEP, "Set up the mesh and finite element spaces", self.SEP)
            print(f"Number of STATE dofs: {ndofs[0]}")
            print(f"Number of PARAMETER dofs: {ndofs[1]}")
    
    def setupPDE(self, u0:float, D:float):
        pde_handler = DiffusionApproximation(dl.Constant(D), dl.Constant(u0))
        self.pde = hp.PDEVariationalProblem(self.Vh, pde_handler, [], [],  is_fwd_linear=True)


class circularInclusion(dl.UserExpression):
    """Expression implementing a circular inclusion.
    """
    def __init__(self, cx, cy, r, vin, vo, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.cx = cx    # center x-coordinate
        self.cy = cy    # center y-coordinate
        self.vin = vin  # inside value
        self.vo = vo    # outside value
        
    def eval_cell(self, values, x, cell):        
        if (pow(x[0]-self.cx,2)+pow(x[1]-self.cy,2) < pow(self.r,2)):
                values[0] = self.vin
        else:
            values[0] = self.vo
