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

from typing import List
import math
from .variables import STATE, PARAMETER, ADJOINT, SLACK
from ..utils.vector2function import vector2Function

class ModelNS:
    """
    This class contains the full description of the inverse problem. This class handles both smooth and non-smooth priors.
    As inputs it takes a :code:`PDEProblem` object, a :code:`SmoothPrior` object, a :code:`NonSmoothPrior`, and a :code:`Misfit` object.
    
    In the following we will denote with

        - :code:`u` the state variable
        - :code:`m` the (model) parameter variable
        - :code:`p` the adjoint variable
        - :code:`w` a slack variable
        
    """
    
    def __init__(self, problem, misfit, prior=None, nsprior=None, which:List[bool]=[True, True, True]):
        """
        Create a model given:

            - problem: the description of the forward/adjoint problem and all the sensitivities
            - misfit: the misfit (data-fidelity) componenent of the cost functional
            - prior: the (smooth or Gaussian) prior component of the cost functional
            - nsprior: the non-smooth prior component of the cost functional
            - which: list that determines which parts of loss functional to use [misfit, smooth, nonsmooth]
        """
        self.problem = problem
        self.misfit = misfit
        self.prior = prior
        self.nsprior = nsprior
        self.which = which
        self.gauss_newton_approx = False
        
        self.n_fwd_solve = 0
        self.n_adj_solve = 0
        self.n_inc_solve = 0

                
    def generate_vector(self, component = "ALL"):
        """
        By default, return the list :code:`[u,m,p,w]` where:
        
            - :code:`u` is any object that describes the state variable
            - :code:`m` is a :code:`dolfin.Vector` object that describes the parameter variable. \
            (Needs to support linear algebra operations)
            - :code:`p` is any object that describes the adjoint variable
            - :code:`w` is any object that describes the slack variable
        
        If :code:`component = STATE` return only :code:`u`
            
        If :code:`component = PARAMETER` return only :code:`m`
            
        If :code:`component = ADJOINT` return only :code:`p`
        
        If :code:`component = SLACK` return only :code:`w`
        """ 
        if component == "ALL":
            x = [self.problem.generate_state(),
                 self.problem.generate_parameter(),
                 self.problem.generate_state(),
                 self.nsprior.generate_slack()]
        elif component == STATE:
            x = self.problem.generate_state()
        elif component == PARAMETER:
            x = self.problem.generate_parameter()
        elif component == ADJOINT:
            x = self.problem.generate_state()
        elif component == SLACK:
            x = self.nsprior.generate_slack()
            
        return x

    
    def init_parameter(self, m):
        """
        Reshape :code:`m` so that it is compatible with the parameter variable
        """
        if self.prior is not None:
            self.prior.init_vector(m,0)
        elif self.nsprior is not None:
            self.nsprior.init_vector(m,0)
        else:
            raise ValueError("No prior defined.")

            
    def cost(self, x):
        """
        Given the list :code:`x = [u,m,p,w]` which describes the state, parameter,
        adjoint, and slack variables, compute the cost functional as the sum of 
        the misfit functional and the regularization functional(s).
        
        Return the list [cost functional, regularization functional, misfit functional]
        
        .. note:: :code:`p` is not needed to compute the cost functional
        """
        misfit_cost = self.misfit.cost(x)
        
        # if there is a smooth portion, compute it
        if self.prior is not None and self.which[1]:
            smooth_reg_cost = self.prior.cost(x[PARAMETER])
        else:
            smooth_reg_cost = 0.
        
        # if there is a nonsmooth portion, compute it
        if self.nsprior is not None and self.which[2]:
            nonsmooth_reg_cost = self.nsprior.cost(x[PARAMETER])
        else:
            nonsmooth_reg_cost = 0.
            
        # sum both regularization terms
        reg_cost = smooth_reg_cost + nonsmooth_reg_cost
        
        return [misfit_cost+reg_cost, smooth_reg_cost, nonsmooth_reg_cost, misfit_cost]
    
    
    def solveFwd(self, out, x):
        """
        Solve the (possibly non-linear) forward problem.
        
        Parameters:

            - :code:`out`: is the solution of the forward problem (i.e. the state) (Output parameters)
            - :code:`x = [u, m, p, w]` provides

                1) the parameter variable :code:`m` for the solution of the forward problem
                2) the initial guess :code:`u` if the forward problem is non-linear
        
                .. note:: :code:`p` is not accessed.
                .. note:: :code:`w` is not accessed
        """
        self.n_fwd_solve = self.n_fwd_solve + 1
        self.problem.solveFwd(out, x)

    
    def solveAdj(self, out, x):
        """
        Solve the linear adjoint problem.

        Parameters:

            - :code:`out`: is the solution of the adjoint problem (i.e. the adjoint :code:`p`) (Output parameter)
            - :code:`x = [u, m, p, w]` provides

                1) the parameter variable :code:`m` for assembling the adjoint operator
                2) the state variable :code:`u` for assembling the adjoint right hand side

                .. note:: :code:`p` is not accessed
                .. note:: :code:`w` is not accessed
        """
        self.n_adj_solve = self.n_adj_solve + 1
        rhs = self.problem.generate_state()
        self.misfit.grad(STATE, x, rhs)
        rhs *= -1.
        self.problem.solveAdj(out, x, rhs)
    
    
    def evalGradientParameter(self, x, mg, misfit_only=False):
        """
        Evaluate the gradient for the variational parameter equation at the point :code:`x=[u,m,p,w]`.

        Parameters:

            - :code:`x = [u, m, p, w]` the point at which to evaluate the gradient.
            - :code:`mg` the variational gradient :math:`(g, mtest)`, mtest being a test function in the parameter space \
            (Output parameter)
            
            .. note:: :code:`w` is not accessed
        
        Returns the norm of the gradient in the correct inner product :math:`g_norm = sqrt(g,g)`
        """ 
        tmp = self.generate_vector(PARAMETER)
        
        if self.which[0]:
            self.problem.evalGradientParameter(x, mg)
            self.misfit.grad(PARAMETER,x,tmp)
            mg.axpy(1., tmp)
        
        if not misfit_only:
            if self.which[1]:
                self.prior.grad(x[PARAMETER], tmp)
                mg.axpy(1., tmp)
                
            if self.which[2]:
                self.nsprior.grad(x[PARAMETER], tmp)
                mg.axpy(1., tmp)
            
        if self.prior is not None:
            self.prior.Msolver.solve(tmp, mg)
        elif self.nsprior is not None:
            self.nsprior.Msolver.solve(tmp, mg)
        else:
            raise ValueError("No prior with mass matrix solver defined.")
            
        #self.prior.Rsolver.solve(tmp, mg)
        return math.sqrt(mg.inner(tmp))
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point :code:`x = [u,m,p,w]` at which the Hessian operator (or the Gauss-Newton approximation)
        needs to be evaluated.

        Parameters:

            - :code:`x = [u, m, p, w]`: the point at which the Hessian or its Gauss-Newton approximation needs to be evaluated.
            - :code:`gauss_newton_approx (bool)`: whether to use Gauss-Newton approximation (default: use Newton) 
            
        .. note:: This routine should either:

            - simply store a copy of x and evaluate action of blocks of the Hessian on the fly
            - or partially precompute the block of the hessian (if feasible)
        """
        self.gauss_newton_approx = gauss_newton_approx
        
        self.problem.setLinearizationPoint(x, self.gauss_newton_approx)
        
        self.misfit.setLinearizationPoint(x, self.gauss_newton_approx)
        
        if hasattr(self.prior, "setLinearizationPoint"):
            self.prior.setLinearizationPoint(x[PARAMETER], self.gauss_newton_approx)
            
        self.nsprior.setLinearizationPoint(x[PARAMETER], x[SLACK], self.gauss_newton_approx)

        
    def solveFwdIncremental(self, sol, rhs):
        """
        Solve the linearized (incremental) forward problem for a given right-hand side

        Parameters:

            - :code:`sol` the solution of the linearized forward problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, False)
        
        
    def solveAdjIncremental(self, sol, rhs):
        """
        Solve the incremental adjoint problem for a given right-hand side

        Parameters:

            - :code:`sol` the solution of the incremental adjoint problem (Output)
            - :code:`rhs` the right hand side of the linear system
        """
        self.n_inc_solve = self.n_inc_solve + 1
        self.problem.solveIncremental(sol,rhs, True)
    
    
    def applyC(self, dm, out):
        """
        Apply the :math:`C` block of the Hessian to a (incremental) parameter variable, i.e.
        :code:`out` = :math:`C dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`C` block on :code:`dm`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(ADJOINT,PARAMETER, dm, out)
    
    
    def applyCt(self, dp, out):
        """
        Apply the transpose of the :math:`C` block of the Hessian to a (incremental) adjoint variable.
        :code:`out` = :math:`C^t dp`

        Parameters:

            - :code:`dp` the (incremental) adjoint variable
            - :code:`out` the action of the :math:`C^T` block on :code:`dp`
            
        ..note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.problem.apply_ij(PARAMETER,ADJOINT, dp, out)

    
    def applyWuu(self, du, out):
        """
        Apply the :math:`W_{uu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{uu} du`
        
        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{uu}` block on :code:`du`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.misfit.apply_ij(STATE,STATE, du, out)
        if not self.gauss_newton_approx:
            tmp = self.generate_vector(STATE)
            self.problem.apply_ij(STATE,STATE, du, tmp)
            out.axpy(1., tmp)
    
    
    def applyWum(self, dm, out):
        """
        Apply the :math:`W_{um}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{um} dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{um}` block on :code:`du`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(STATE,PARAMETER, dm, out)
            tmp = self.generate_vector(STATE)
            self.misfit.apply_ij(STATE,PARAMETER, dm, tmp)
            out.axpy(1., tmp)

    
    def applyWmu(self, du, out):
        """
        Apply the :math:`W_{mu}` block of the Hessian to a (incremental) state variable.
        :code:`out` = :math:`W_{mu} du`
        
        Parameters:

            - :code:`du` the (incremental) state variable
            - :code:`out` the action of the :math:`W_{mu}` block on :code:`du`
        
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(PARAMETER, STATE, du, out)
            tmp = self.generate_vector(PARAMETER)
            self.misfit.apply_ij(PARAMETER, STATE, du, tmp)
            out.axpy(1., tmp)
    
    
    def applyR(self, dm, out):
        """
        Apply the regularization :math:`R` to a (incremental) parameter variable.
        :code:`out` = :math:`R dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of :math:`R` on :code:`dm`
        
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """       
        self.prior.R.mult(dm, out)
    
    
    def Psolver(self):
        """
        Return an object :code:`PSolver` that is a suitable preconditioner for the Hessian operator.
        
        The solver object should implment the method :code:`PSolver.solve(z,r)` such that
        :math:`Pz \approx r`.
        """
        if self.prior is not None and self.which[1]:
            # return the precision matrix if using a smooth prior
            return self.prior.Rsolver
        elif self.nsprior is not None and self.which[2]:
            # return solver object for the non-smooth portion + bit of mass matrix
            return self.nsprior.Psolver()
        else:
            raise ValueError("No preconditioner defined.")
    
    
    def Rsolver(self):
        """
        Return an object :code:`Rsolver` that is a suitable solver for the regularization
        operator :math:`R`.
        
        The solver object should implement the method :code:`Rsolver.solve(z,r)` such that
        :math:`Rz \approx r`.
        """
        if self.prior is not None and self.which[1]:
            return self.prior.Rsolver
        else:
            return None


    def applyRNS(self, dm, out):
        """
        Apply the non-smooth regularization :math:`R` to a (incremental) parameter variable.
        :code:`out` = :math:`R dm`
        
        Parameters:

            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of :math:`R` on :code:`dm`
        
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        self.nsprior.applyR(dm, out)

    
    def applyWmm(self, dm, out):
        """
        Apply the :math:`W_{mm}` block of the Hessian to a (incremental) parameter variable.
        :code:`out` = :math:`W_{mm} dm`
        
        Parameters:
        
            - :code:`dm` the (incremental) parameter variable
            - :code:`out` the action of the :math:`W_{mm}` on block :code:`dm`
            
        .. note:: This routine assumes that :code:`out` has the correct shape.
        """
        if self.gauss_newton_approx:
            out.zero()
        else:
            self.problem.apply_ij(PARAMETER,PARAMETER, dm, out)
            tmp = self.generate_vector(PARAMETER)
            self.misfit.apply_ij(PARAMETER,PARAMETER, dm, tmp)
            out.axpy(1., tmp)
       
            
    def apply_ij(self, i, j, d, out):
        if i == STATE and j == STATE:
            self.applyWuu(d,out)
        elif i == STATE and j == PARAMETER:
            self.applyWum(d,out)
        elif i == PARAMETER and j == STATE:
            self.applyWmu(d, out)
        elif i == PARAMETER and j == PARAMETER:
            self.applyWmm(d,out)
        elif i == PARAMETER and j == ADJOINT:
            self.applyCt(d,out)
        elif i == ADJOINT and j == PARAMETER:
            self.applyC(d,out)
        else:
            raise IndexError("apply_ij not allowed for i = {0}, j = {1}".format(i,j))
