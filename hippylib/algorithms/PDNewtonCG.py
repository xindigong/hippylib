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

import math
from ..utils.parameterList import ParameterList
from ..modeling.reducedHessian import NSReducedHessian
from ..modeling.variables import STATE, PARAMETER, ADJOINT, SLACK
from .cgsolverSteihaug import CGSolverSteihaug


def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]
    
    return ParameterList(parameters)


def ReducedSpacePDNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["cg_max_iter"]           = [100, "Maximum CG iterations"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    
    return ParameterList(parameters)
    

class ReducedSpacePDNewtonCG:
    
    """
    Primal-Dual implementation of Inexact Newton-CG method to solve constrained optimization problems in the reduced parameter space.
    The Newton system is solved inexactly by early termination of CG iterations via Eisenstat-Walker
    (to prevent oversolving) and Steihaug (to avoid negative curvature) criteria.
    The primal-dual Netwon algorithm with double line search is from [1]
    
    [1] Chan, Tony F., Gene H. Golub, and Pep Mulet. "A nonlinear primal-dual method for total variation-based image restoration." SIAM journal on scientific computing 20.6 (1999): 1964-1977.
    
    Globalization is performed using:

    - line search (LS) based on the armijo sufficient reduction condition

    The stopping criterion is based on a control on the norm of the gradient and a control of the
    inner product between the gradient and the Newton direction.
       
    The user must provide a model that describes the forward problem, cost functionals, and all the
    derivatives for the gradient and the Hessian.
    
    More specifically the model object should implement following methods:
    
       - :code:`generate_vector()` -> generate the object containing state, parameter, adjoint
       - :code:`cost(x)` -> evaluate the cost functional, report regularization part and misfit separately
       - :code:`solveFwd(out, x)` -> solve the possibly non linear forward problem
       - :code:`solveAdj(out, x)` -> solve the linear adjoint problem
       - :code:`evalGradientParameter(x, out)` -> evaluate the gradient of the parameter and compute its norm
       - :code:`setPointForHessianEvaluations(x)` -> set the state to perform hessian evaluations
       - :code:`solveFwdIncremental(out, rhs)` -> solve the linearized forward problem for a given :code:`rhs`
       - :code:`solveAdjIncremental(out, rhs)` -> solve the linear adjoint problem for a given :code:`rhs`
       - :code:`applyC(dm, out)`    --> Compute out :math:`= C_x dm`
       - :code:`applyCt(dp, out)`   --> Compute out = :math:`C_x  dp`
       - :code:`applyWuu(du,out)`   --> Compute out = :math:`(W_{uu})_x  du`
       - :code:`applyWmu(dm, out)`  --> Compute out = :math:`(W_{um})_x  dm`
       - :code:`applyWmu(du, out)`  --> Compute out = :math:`W_{mu}  du`
       - :code:`applyR(dm, out)`    --> Compute out = :math:`R  dm`
       - :code:`applyRNS(dm, out)`  --> Compute out = :math:`R_{NS} dm`
       - :code:`applyWmm(dm,out)`   --> Compute out = :math:`W_{mm} dm`
       - :code:`Psolver()`          --> A solver for the Hessian preconditioner
       
    Type :code:`help(Model)` for additional information
    """
    termination_reasons = [
                           "Maximum number of Iterations reached",                 #0
                           "Norm of the gradient less than tolerance",             #1
                           "Maximum number of (parameter) backtracking reached",   #2
                           "Maximum number of (slack) backtracking reached",       #3
                           "Norm of (g, dm) less than tolerance"                   #4
                           ]
    
    def __init__(self, model, parameters=ReducedSpacePDNewtonCG_ParameterList(), callback = None):
        """
        Initialize the ReducedSpaceNewtonCG.
        Type :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for list of default parameters
        and their descriptions.
        
        Parameters:
        :code:`model` The model object that describes the inverse problem
        :code:`parameters`: (type :code:`ParameterList`, optional) set parameters for primal-dual inexact Newton CG
        :code:`callback`: (type function handler with signature :code:`callback(it: int, x: list of dl.Vector): --> None`
               optional callback function to be called at the end of each iteration. Takes as input the iteration number, and
               the list of vectors for the state, parameter, adjoint.
        """
        self.model = model
        self.parameters = parameters
        
        self.it = 0
        self.converged = False
        self.total_cg_iter = 0
        self.ncalls = 0
        self.reason = 0
        self.final_grad_norm = 0
        
        self.callback = callback
        
    def solve(self, x):
        """

        Input: 
            :code:`x = [u, m, p, w]` represents the initial guess (u, p, and w may be None). 
            :code:`x` will be overwritten on return.
        """
        if self.model is None:
            raise TypeError("model can not be of type None.")
        
        if x[STATE] is None:
            x[STATE] = self.model.generate_vector(STATE)
            
        if x[ADJOINT] is None:
            x[ADJOINT] = self.model.generate_vector(ADJOINT)
            
        if x[SLACK] is None:
            # alternatively, one could think about computing the slack variable from nonzero initial parameter
            x[SLACK] = self.model.generate_vector(SLACK)
            
        if self.parameters["globalization"] == "LS":
            return self._solve_dls(x)
        else:
            raise ValueError(self.parameters["globalization"])
        
    def _solve_dls(self,x):
        """
        Solve the constrained optimization problem with initial guess :code:`x`.
        """
        rel_tol = self.parameters["rel_tolerance"]
        abs_tol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        print_level = self.parameters["print_level"]
        GN_iter = self.parameters["GN_iter"]
        cg_coarse_tolerance = self.parameters["cg_coarse_tolerance"]
        cg_max_iter         = self.parameters["cg_max_iter"]
        
        c_armijo = self.parameters["LS"]["c_armijo"]
        max_backtracking_iter = self.parameters["LS"]["max_backtracking_iter"]
        
        self.model.solveFwd(x[STATE], x)
        
        self.it = 0
        self.converged = False
        self.ncalls += 1
        
        mhat = self.model.generate_vector(PARAMETER)
        what = self.model.generate_vector(SLACK)
        mg = self.model.generate_vector(PARAMETER)
        
        x_star = [None, None, None, None]
        x_star[STATE]     = self.model.generate_vector(STATE)
        x_star[PARAMETER] = self.model.generate_vector(PARAMETER)
        x_star[SLACK]     = self.model.generate_vector(SLACK)
        
        cost_old, _, _, _ = self.model.cost(x)
        
        while (self.it < max_iter) and (self.converged == False):
            self.model.solveAdj(x[ADJOINT], x)
            
            self.model.setPointForHessianEvaluations(x, gauss_newton_approx=(self.it < GN_iter) )
            gradnorm = self.model.evalGradientParameter(x, mg)
            
            if self.it == 0:
                gradnorm_ini = gradnorm
                tol = max(abs_tol, gradnorm_ini*rel_tol)
                
            # check if solution is reached
            if (gradnorm < tol) and (self.it > 0):
                self.converged = True
                self.reason = 1
                break
            
            self.it += 1
            
            tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
            
            # compute m_hat using (3.5) from [1]
            HessApply = NSReducedHessian(self.model)
            solver = CGSolverSteihaug(comm = self.model.nsprior.mpi_comm())
            solver.set_operator(HessApply)
            solver.set_preconditioner(self.model.Psolver())
            solver.parameters["rel_tolerance"] = tolcg
            solver.parameters["max_iter"] = cg_max_iter
            solver.parameters["zero_initial_guess"] = True
            solver.parameters["print_level"] = print_level-1
            
            solver.solve(mhat, -mg)
            self.total_cg_iter += HessApply.ncalls
            
            # compute w_hat using (3.6) from [1]
            # In the first iteration, the slack variable is left as zero and not computed from the parameter
            # this ensures stability if the initial parameter is constant (zero gradient) and has similar convergence
            self.model.nsprior.compute_w_hat(x[PARAMETER], x[SLACK], mhat, what)
            
            ### line search for m
            alpha_m = 1.0
            descent_m = 0
            n_backtrack_m = 0
            
            mg_mhat = mg.inner(mhat)  # inner product in (5.8) from [1]
            
            while descent_m == 0 and n_backtrack_m < max_backtracking_iter:
                # update m and u
                x_star[PARAMETER].zero()
                x_star[PARAMETER].axpy(1., x[PARAMETER])
                x_star[PARAMETER].axpy(alpha_m, mhat)
                x_star[STATE].zero()
                x_star[STATE].axpy(1., x[STATE])
                self.model.solveFwd(x_star[STATE], x_star)
                
                cost_new, smooth_reg_new, nonsmooth_reg_new, misfit_new = self.model.cost(x_star)
                  
                # Check if armijo conditions are satisfied
                if (cost_new < cost_old + alpha_m * c_armijo * mg_mhat) or (-mg_mhat <= self.parameters["gdm_tolerance"]):
                    cost_old = cost_new
                    descent_m = 1
                    x[PARAMETER].zero()
                    x[PARAMETER].axpy(1., x_star[PARAMETER])
                    x[STATE].zero()
                    x[STATE].axpy(1., x_star[STATE])
                else:
                    n_backtrack_m += 1
                    alpha_m *= 0.5
            
            # if backtracking for m failed, exit
            if n_backtrack_m == max_backtracking_iter:
                self.converged = False
                self.reason = 2
                break
            
            ### line search for w
            alpha_w = 1.0
            descent_w = 0
            n_backtrack_w = 0
            
            while descent_w == 0 and n_backtrack_w < max_backtracking_iter:
                # update w and u
                x_star[SLACK].zero()
                x_star[SLACK].axpy(1., x[SLACK])
                x_star[SLACK].axpy(alpha_w, what)
                
                x_star[STATE].zero()
                x_star[STATE].axpy(1., x[STATE])
                
                norm_w = self.model.nsprior.wnorm(x_star[SLACK])
                if norm_w.norm("linf") <= 1:
                    # descent direction found, update the slack variable
                    descent_w = 1
                    x[SLACK].zero()
                    x[SLACK].axpy(1., x_star[SLACK])
                else:
                    n_backtrack_w += 1
                    alpha_w *= 0.5
            
            # if backtracking for w failed, exit
            if n_backtrack_m == max_backtracking_iter:
                self.converged = False
                self.reason = 3
                break
            
            if(print_level >= 0) and (self.it == 1):
                print( "\n{0:3} {1:3} {2:15} {3:15} {4:15} {5:15} {6:15} {7:14} {8:14} {9:14}".format(
                      "It", "cg_it", "cost", "misfit", "s reg", "ns reg", "(g,dm)", "||g||L2", "alpha", "tolcg") )
                
            if print_level >= 0:
                print( "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e} {5:15e} {6:14e} {7:14e} {8:14e} {9:14e}".format(
                        self.it, HessApply.ncalls, cost_new, misfit_new, smooth_reg_new, nonsmooth_reg_new, mg_mhat, gradnorm, alpha_m, tolcg) )
                
            if self.callback:
                self.callback(self.it, x)
            
            if -mg_mhat <= self.parameters["gdm_tolerance"]:
                self.converged = True
                self.reason = 4
                break
                            
        self.final_grad_norm = gradnorm
        self.final_cost      = cost_new
        return x
