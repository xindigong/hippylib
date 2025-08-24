# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2022, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
# Copyright (c) 2023-2024, The University of Texas at Austin 
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
import ufl
import numpy as np

from ..algorithms.linSolvers import PETScKrylovSolver, PETScLUSolver
from ..utils.vector2function import vector2Function

class TVPrior:
    """
    This class implements the primal-dual formulation for the total variation prior.
    
    References:
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear primal-dual method for total variation-based image restoration." SIAM journal on scientific computing 20.6 (1999): 1964-1977.
    """
    
    # primal-dual implementation for (vector) total variation prior
    def __init__(self, Vhm:dl.FunctionSpace, Vhw:dl.FunctionSpace, Vhwnorm:dl.FunctionSpace, alpha:float, beta:float, peps:float=1e-3, rel_tol:float=1e-12, max_iter:int=100, solver_type="krylov", lu_method="default"):
        """Constructor for the TVPrior class.

        Args:
            Vhm (dl.FunctionSpace): Function space for the parameter.
            Vhw (dl.FunctionSpace): Function space for slack variable.
            Vhwnorm (dl.FunctionSpace): Function space for slack variable norm.
            alpha (array-like): Weights for each component of wVTV functional.
            beta (float): Smoothing parameter for the TV functional.
            peps (float, optional): Mass matrix preconditioner scaling. Defaults to 1e-3.
            rel_tol (float, optional): Relative tolerance for Krylov solver. Defaults to 1e-12.
            max_iter (int, optional): Maximum number of iterations for Krylov solver. Defaults to 100.
            solver_type (str, optional): type of solver to use for solving linear systems involving matrix. Options are "krylov" or "lu" (default is krylov).
            lu_method (str, optional): method to use for the LU solver (used when :code:`solver_type == "lu"`, default is "default").
        """
        
        self.alpha = dl.Constant(alpha)
        self.beta = dl.Constant(beta)
        
        self.Vhm = Vhm  # function space for the parameter
        self.Vhw = Vhw  # function space for the slack variable
        self.Vhwnorm = Vhwnorm  # function space for the norm of the slack variable

        # linearization point
        self.m_lin = None
        self.w_lin = None
        
        self.gauss_newton_approx = False  # by default don't use GN approximation to Hessian
        self.peps = peps  # mass matrix perturbation for preconditioner

        # assemble mass matrix for parameter, slack variable, norm of slack variable
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.solver_type = solver_type
        self.lu_method = lu_method
        self.m_trial, self.m_test, self.M, self.Msolver = self._setupM(self.Vhm)
        self.w_trial, self.w_test, self.Mw, self.Mwsolver = self._setupM(self.Vhw)
        self.wnorm_trial, self.wnorm_test, self.Mwnorm, self.Mwnormsolver = self._setupM(self.Vhwnorm)
        
        self.ncomp = self.Vhm.num_sub_spaces()        # number of components in the parameter
        self.ndim = self.Vhm.mesh().topology().dim()  # number of spatial dimensions

    def _setupM(self, Vh:dl.FunctionSpace):
        # helper function to set up mass matrix, solver
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)
        
        # assemble mass matrix from variational form
        varfM = dl.inner(trial, test)*dl.dx
        M = dl.assemble(varfM)
        
        # set up PETSc solver object to apply M^{-1}
        if self.solver_type == "krylov":
            Msolver = PETScKrylovSolver(Vh.mesh().mpi_comm(), "cg", "jacobi")
            Msolver.set_operator(M)
            Msolver.parameters["maximum_iterations"] = self.max_iter
            Msolver.parameters["relative_tolerance"] = self.rel_tol
            Msolver.parameters["error_on_nonconvergence"] = True
            Msolver.parameters["nonzero_initial_guess"] = False
        elif self.solver_type == "lu":
            Msolver = PETScLUSolver(Vh.mesh().mpi_comm(), method=self.lu_method)
            Msolver.set_operator(M)
            Msolver.parameters["symmetric"] = True
        else:
            raise ValueError(f"Unknown solver type {self.solver_type}")
        
        # return objects
        return trial, test, M, Msolver


    def _fTV(self, m:dl.Function)->dl.Function:
        """Computes the TV functional.

        Args:
            m (dl.Function): Function to compute the TV norm of.

        Returns:
            dl.Function: TV norm of m.
        """
        return dl.sqrt( dl.inner(dl.grad(m), dl.grad(m)) + self.beta)
    
    
    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)
    
    
    def setLinearizationPoint(self, m:dl.Vector, w:dl.Vector, gauss_newton_approx:bool):
        self.m_lin = vector2Function(m, self.Vhm)
        self.w_lin = vector2Function(w, self.Vhw)
        self.gauss_newton_approx = gauss_newton_approx
    
    
    def cost(self, m):
        # (smoothed) TV functional
        m = vector2Function(m, self.Vhm)
        return dl.assemble( self.alpha * self._fTV(m)*dl.dx )
    
    
    def grad(self, m, out):
        out.zero()
        m = vector2Function(m, self.Vhm)
        
        TVm = self._fTV(m)
        grad_tv = self.alpha * dl.Constant(1.)/TVm*dl.inner(dl.grad(m), dl.grad(self.m_test))*dl.dx
        
        # assemble the UFL form to a vector, add to out
        dl.assemble(grad_tv, tensor=out)
    
    
    def hess_action(self, m, w, m_dir):
        TVm = self._fTV(m)
        
        Acoeff = self.primal_hess_coeff(m, w, TVm)
        
        if self.ncomp == 0:
            return self.alpha * dl.inner(ufl.dot(Acoeff, dl.grad(m_dir)), dl.grad(self.m_test))*dl.dx
        else:
            # tensor contraction for vector case
            i,j,k,l = ufl.indices(4)
            return self.alpha * ufl.grad(m_dir)[i,j]*Acoeff[i,j,k,l]*ufl.grad(self.m_test)[k,l] * dl.dx
    
    
    
    def primal_hess_coeff(self, m, w, TVm):
        """Variational form of the primal Hessian, symmetrized (5.1) and (5.2) from [1]
        """
        
        if self.ncomp == 0:
            # only one parameter
            Acoeff = dl.Constant(1.)/TVm * ( dl.Identity(self.ndim) 
                                    - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                    - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        else:
            # shape should be (ncomp, ndim, ncomp, ndim) to be compatible with the gradient
            i, j, k, l = ufl.indices(4)
            eye_ncomp = ufl.Identity(self.ncomp)
            eye_ndim = ufl.Identity(self.ndim)
            eye = ufl.as_tensor(eye_ncomp[i, k] * eye_ndim[j, l], [i, j, k, l])
            
            Acoeff = dl.Constant(1.)/TVm * ( eye
                                    - dl.Constant(0.5)*dl.outer(w, dl.grad(m)/TVm)
                                    - dl.Constant(0.5)*dl.outer(dl.grad(m)/TVm, w) )
        
        return Acoeff
    
    
    def applyR(self, dm, out):
        out.zero()  # zero out the output
        
        m_dir = vector2Function(dm, self.Vhm)
        hessian_action_form = self.hess_action(self.m_lin, self.w_lin, m_dir)
        
        dl.assemble(hessian_action_form, tensor=out)
    
    
    def compute_w_hat(self, m, w, m_hat, w_hat):
        m = vector2Function(m, self.Vhm)
        m_hat = vector2Function(m_hat, self.Vhm)
        w = vector2Function(w, self.Vhw)
        
        TVm = self._fTV(m)
        
        Acoeff = self.primal_hess_coeff(m, w, TVm)
        
        # expression for incremental slack variable (3.6) from [1]
        if self.ncomp == 0:
            dw = Acoeff*dl.grad(m_hat) - w + dl.grad(m)/TVm
        else:
            # tensor contraction for vector case
            i,j,k,l = ufl.indices(4)
            dw = ufl.as_tensor(Acoeff[i,j,k,l] * dl.grad(m_hat)[k,l], [i,j]) - w + dl.grad(m)/TVm
        
        dw = dl.assemble( dl.inner(self.w_test, dw)*dl.dx )
        
        # project into appropriate space
        self.Mwsolver.solve(w_hat, dw)
    
    
    def wnorm(self, w):
        w = vector2Function(w, self.Vhw)
        
        # compute functional and assemble
        nw = dl.inner(w, w)
        nw = dl.assemble( dl.inner(self.wnorm_test, nw)*dl.dx )
        
        # project into appropriate space
        out = dl.Vector(self.Mwnorm.mpi_comm())
        self.Mwnorm.init_vector(out, 0)
        self.Mwnormsolver.solve(out, nw)
        
        return out
    
    
    def generate_slack(self):
        """ Return a vector in the shape of the slack variable. """
        return dl.Function(self.Vhw).vector()
    
    
    def compute_w(self, m:dl.Vector):
        m = vector2Function(m, self.Vhm)
        
        TVm = self._fTV(m)
        w = dl.grad(m)/TVm
        w = dl.assemble( dl.inner(self.w_test, w)*dl.dx )
        
        out = dl.Vector(self.Mw.mpi_comm())
        self.Mw.init_vector(out, 0)
        self.Mwsolver.solve(out, w)
        return out
    
    
    def Psolver(self):
        # set up the preconditioner for the Hessian
        varfHTV = self.hess_action(self.m_lin, self.w_lin, self.m_trial)
        varfM = dl.inner(self.m_trial, self.m_test)*dl.dx
        varfP = varfHTV + self.peps*varfM
        
        # assemble the preconditioner and set as operator for solver
        P = dl.assemble(varfP)
        
        if self.solver_type == "krylov":
            Psolver = PETScKrylovSolver(self.Vhm.mesh().mpi_comm(), "cg", "hypre_amg")
            Psolver.parameters["nonzero_initial_guess"] = False
            Psolver.set_operator(P)
        elif self.solver_type == "lu":
            Psolver = PETScLUSolver(self.Vhm.mesh().mpi_comm(), method="default")
            Psolver.set_operator(P)
        else:
            raise ValueError(f"Unknown solver type {self.solver_type}")
        
        return Psolver
    
    
    def mpi_comm(self):
        return self.Vhm.mesh().mpi_comm()


class weightedVTVPrior:
    """
    This class implements the primal-dual formulation for the weighted vector total variation prior.
    
    The functional is defined as:
    :math:`wVTV(m) := \\int_\\Omega \\sqrt( \\sum_{i=1}^n \\alpha_i | \\nabla m_i |^2 + \\beta ) dx`
    where \alpha_i are the weights for each component of the parameter vector m.
    
    References:
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear primal-dual method for total variation-based image restoration." SIAM journal on scientific computing 20.6 (1999): 1964-1977.
    """
    
    # primal-dual implementation for (vector) total variation prior
    def __init__(self, Vhm:dl.FunctionSpace, Vhw:dl.FunctionSpace, Vhwnorm:dl.FunctionSpace, alpha, beta:float, peps:float=1e-3, rel_tol:float=1e-12, max_iter:int=100, solver_type="krylov", lu_method="default"):
        """Constructor for the weightedTVPrior class.

        Args:
            Vhm (dl.FunctionSpace): Function space for the parameter.
            Vhw (dl.FunctionSpace): Function space for slack variable.
            Vhwnorm (dl.FunctionSpace): Function space for slack variable norm.
            alpha (array-like): Weights for each component of wVTV functional.
            beta (float): Smoothing parameter for the TV functional.
            peps (float, optional): Mass matrix preconditioner scaling. Defaults to 1e-3.
            rel_tol (float, optional): Relative tolerance for Krylov solver. Defaults to 1e-12.
            max_iter (int, optional): Maximum number of iterations for Krylov solver. Defaults to 100.
            solver_type (str, optional): type of solver to use for solving linear systems involving matrix. Options are "krylov" or "lu" (default is krylov).
            lu_method (str, optional): method to use for the LU solver (used when :code:`solver_type == "lu"`, default is "default").
        """
        
        # defensive checks to ensure the user is choosing the correct class.
        assert len(alpha) > 1, "alpha must be a vector of weights for each component of the parameter."
        assert len(alpha) == Vhm.num_sub_spaces(), "alpha must have the same length as the number of components in the parameter."
        
        self.alpha_vec = np.array(alpha)
        self.beta = dl.Constant(beta)
        
        self.Vhm = Vhm  # function space for the parameter
        self.Vhw = Vhw  # function space for the slack variable
        self.Vhwnorm = Vhwnorm  # function space for the norm of the slack variable

        # linearization point
        self.m_lin = None
        self.w_lin = None
        
        self.gauss_newton_approx = False  # by default don't use GN approximation to Hessian
        self.peps = peps  # mass matrix perturbation for preconditioner

        # assemble mass matrix for parameter, slack variable, norm of slack variable
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.solver_type = solver_type
        self.lu_method = lu_method
        self.m_trial, self.m_test, self.M, self.Msolver = self._setupM(self.Vhm)
        self.w_trial, self.w_test, self.Mw, self.Mwsolver = self._setupM(self.Vhw)
        self.wnorm_trial, self.wnorm_test, self.Mwnorm, self.Mwnormsolver = self._setupM(self.Vhwnorm)
        
        self.ncomp = self.Vhm.num_sub_spaces()        # number of components in the parameter
        self.ndim = self.Vhm.mesh().topology().dim()  # number of spatial dimensions
        
        # generate the tensor object for the weights (ndim, ndim)
        self.alpha_mat = np.eye(self.ncomp)
        np.fill_diagonal(self.alpha_mat, self.alpha_vec)
        self.alpha = dl.Constant(self.alpha_mat)

    def _setupM(self, Vh:dl.FunctionSpace):
        # helper function to set up mass matrix, solver
        trial = dl.TrialFunction(Vh)
        test = dl.TestFunction(Vh)
        
        # assemble mass matrix from variational form
        varfM = dl.inner(trial, test)*dl.dx
        M = dl.assemble(varfM)
        
        # set up PETSc solver object to apply M^{-1}
        if self.solver_type == "krylov":
            Msolver = PETScKrylovSolver(Vh.mesh().mpi_comm(), "cg", "jacobi")
            Msolver.set_operator(M)
            Msolver.parameters["maximum_iterations"] = self.max_iter
            Msolver.parameters["relative_tolerance"] = self.rel_tol
            Msolver.parameters["error_on_nonconvergence"] = True
            Msolver.parameters["nonzero_initial_guess"] = False
        elif self.solver_type == "lu":
            Msolver = PETScLUSolver(Vh.mesh().mpi_comm(), method=self.lu_method)
            Msolver.set_operator(M)
            Msolver.parameters["symmetric"] = True
        else:
            raise ValueError(f"Unknown solver type {self.solver_type}")
        
        # return objects
        return trial, test, M, Msolver


    def _fVTV(self, m:dl.Function)->dl.Function:
        """Computes the weighted VTV functional.
        """
        return dl.sqrt( dl.inner( self.alpha * dl.grad(m), dl.grad(m) ) + self.beta )
    
    
    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)
    
    
    def setLinearizationPoint(self, m:dl.Vector, w:dl.Vector, gauss_newton_approx:bool):
        self.m_lin = vector2Function(m, self.Vhm)
        self.w_lin = vector2Function(w, self.Vhw)
        self.gauss_newton_approx = gauss_newton_approx
    
    
    def cost(self, m):
        """Compute the weighted VTV cost functional.
        """
        m = vector2Function(m, self.Vhm)
        return dl.assemble( self._fVTV(m) * dl.dx )
    
    
    def grad(self, m, out):
        """Compute the gradient of the VTV functional.
        """
        out.zero()
        m = vector2Function(m, self.Vhm)
        
        wvtv_form = self._fVTV(m)
        grad_form = dl.inner( self.alpha * dl.grad(m), dl.grad(self.m_test) ) / wvtv_form
        
        # assemble the UFL form to a vector, add to out
        dl.assemble(grad_form*dl.dx, tensor=out)
    
    
    def hess_action(self, m, w, m_dir):
        """Compute the action of the Hessian in a direction (m_dir).
        """
        wvtv_form = self._fVTV(m)  # compute the weighted VTV functional
        Acoeff = self.primal_hess_coeff(m, w, wvtv_form)  # compute the diffusion tensor
        
        # tensor contraction to apply hessian action
        i,j,k,l = ufl.indices(4)
        return ufl.grad(m_dir)[i,j]*Acoeff[i,j,k,l]*ufl.grad(self.m_test)[k,l] * dl.dx
    
    
    def primal_hess_coeff(self, m, w, wvtv_form):
        """Variational form of the primal Hessian, symmetrized (5.1) and (5.2) from [1]
        """
        
        i, j, k, l = ufl.indices(4)
        eye_ndim = ufl.Identity(self.ndim)
        # Construct the weighted identity tensor for the Hessian:
        # weighted_eye[i, j, k, l] = alpha[i, k] * I{j,l}
        # - alpha[i, k] is the (i, k) entry of the weight matrix (typically diagonal, with per-component weights)
        # - eye_ndim[j, l] is the identity in the spatial dimension (I{j, l})
        # The resulting tensor has shape (ncomp, ndim, ncomp, ndim) and is used as the diagonal part of the weighted vectorial TV Hessian.
        weighted_eye = ufl.as_tensor( self.alpha[i,k] * eye_ndim[j,l], [i,j,k,l] )
        
        # Acoeff is the weighted vectorial TV Hessian tensor:
        #   A_{ijkl} = (1/|m|_ε) * [ (weighted_eye) - (α_i α_j) * (v ⊗ w) ]
        # where:
        #   - weighted_eye is the weighted identity tensor with indices (i,j,k,l)
        #   - α_i, α_j are the weights for components i and j
        #   - v = grad(m) / |m|_ε is the normalized gradient of the parameter
        #   - w is the slack variable (normalized by the weighted VTV norm)
        #   - The subtraction terms represent the coupling between components in the Hessian
        v = dl.grad(m) / wvtv_form
        Acoeff = (1.0 / wvtv_form) * ( weighted_eye
                                - dl.Constant(0.5) * dl.outer(self.alpha*w, self.alpha*v)
                                - dl.Constant(0.5) * dl.outer(self.alpha*v, self.alpha*w) )
        
        return Acoeff
    
    
    def applyR(self, dm, out):
        out.zero()  # zero out the output
        
        m_dir = vector2Function(dm, self.Vhm)
        hessian_action_form = self.hess_action(self.m_lin, self.w_lin, m_dir)
        
        dl.assemble(hessian_action_form, tensor=out)
    
    
    def compute_w_hat(self, m, w, m_hat, w_hat):
        """Compute the incremental slack variable w_hat (in-place) using the primal-dual formulation.
        """
        m = vector2Function(m, self.Vhm)
        m_hat = vector2Function(m_hat, self.Vhm)
        w = vector2Function(w, self.Vhw)
        
        wvtv_form = self._fVTV(m)
        
        Acoeff = self.primal_hess_coeff(m, w, wvtv_form)
        
        # expression for incremental slack variable (3.6) from [1]
        i,j,k,l = ufl.indices(4)
        dw = ufl.as_tensor(Acoeff[i,j,k,l] * dl.grad(m_hat)[k,l], [i,j]) - w + ( dl.grad(m) / wvtv_form )  # todo: double check
        
        dw = dl.assemble( dl.inner(self.w_test, dw)*dl.dx )
        
        # project into appropriate space
        self.Mwsolver.solve(w_hat, dw)
    
    
    def wnorm(self, w):
        """Compute the norm of the slack variable w.
        """
        w = vector2Function(w, self.Vhw)
        
        # compute functional and assemble
        nw = dl.inner(w, w)
        nw = dl.assemble( dl.inner(self.wnorm_test, nw)*dl.dx )
        
        # project into appropriate space
        out = dl.Vector(self.Mwnorm.mpi_comm())
        self.Mwnorm.init_vector(out, 0)
        self.Mwnormsolver.solve(out, nw)
        
        return out
    
    
    def generate_slack(self):
        """Return a vector in the shape of the slack variable.
        """
        return dl.Function(self.Vhw).vector()
    
    
    def compute_w(self, m:dl.Vector):
        """Compute the slack variable.
        """
        m = vector2Function(m, self.Vhm)
        
        wvtv_form = self._fVTV(m)
        w = dl.grad(m) / wvtv_form
        w = dl.assemble( dl.inner(self.w_test, w)*dl.dx )
        
        out = dl.Vector(self.Mw.mpi_comm())
        self.Mw.init_vector(out, 0)
        self.Mwsolver.solve(out, w)
        return out
    
    
    def Psolver(self):
        """Set up a Krylov solver for Hessian preconditioner.
        """
        # set up the preconditioner for the Hessian
        varfHTV = self.hess_action(self.m_lin, self.w_lin, self.m_trial)
        varfM = dl.inner(self.m_trial, self.m_test)*dl.dx
        varfP = varfHTV + self.peps*varfM
        
        # assemble the preconditioner and set as operator for solver
        P = dl.assemble(varfP)
        
        if self.solver_type == "krylov":
            Psolver = PETScKrylovSolver(self.Vhm.mesh().mpi_comm(), "cg", "hypre_amg")
            Psolver.parameters["nonzero_initial_guess"] = False
            Psolver.set_operator(P)
        elif self.solver_type == "lu":
            Psolver = PETScLUSolver(self.Vhm.mesh().mpi_comm(), method=self.lu_method)
            Psolver.set_operator(P)
        else:
            raise ValueError(f"Unknown solver type {self.solver_type}")
        
        return Psolver
    
    
    def mpi_comm(self):
        """Return the MPI communicator.
        """
        return self.Vhm.mesh().mpi_comm()

