import sys
import os
import time

import ufl
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

from models import qPACT_DA, DiffusionApproximationMisfitForm, circularInclusion, rprint

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp

COMM = dl.MPI.comm_world
dl.parameters['form_compiler']['quadrature_degree'] = 4  # set quadrature degree
FIG_DIR = "figs/qpact_diffusion_approx"
os.makedirs(FIG_DIR, exist_ok=True)  # ensure figure directory exists

SEP = "\n"+"#"*80+"\n"  # for printing
NOISE_VARIANCE = 1e-6  # variance of the noise
GAMMA = 0.05  # BiLaplacian prior parameter
DELTA = 1.    # BiLaplacian prior parameter

# for the PD TV solver
DO_LCURVE = False
VERBOSE = True
TVONLY = [True, False, True]
ALPHA = 1e0  # regularization parameter, picked from L-curve
BETA = 1e-3
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 1000
CG_MAX_ITER = 75
MAX_BACKTRACK = 25

# qPACT parameters
MESH_FPATH = "mesh/circle.xdmf"
C = [2., 2.]            # center of the inclusion
R = 1.0                 # radius of the inclusion
mu_a_background = 0.01  # background absorption coefficient
mu_a_inclusion = 0.2    # inclusion absorption coefficient
D0 = 1. / 24.           # diffusion coefficient
u0 = 1.0                # incident fluence

##################################################
# set up the problem
##################################################
rprint(COMM, SEP)
rprint(COMM,"Set up the qPACT problem with the diffusion approximation.")
rprint(COMM, f"Results will be stored at: {FIG_DIR}")
rprint(COMM, SEP)

qpact = qPACT_DA(COMM, MESH_FPATH)
qpact.setupMesh()
qpact.setupFunctionSpaces()
qpact.setupPDE(u0, D0)

##################################################
# setup the true parameter, generate noisy observations, setup the misfit
##################################################
rprint(COMM, SEP)
rprint(COMM, "Set up the true parameter, generate noisy observations.")
rprint(COMM, SEP)

m_true_expr = circularInclusion(cx=C[0], cy=C[1], r=R, vin=np.log(mu_a_inclusion), vo=np.log(mu_a_background))
m_fun_true = dl.interpolate(m_true_expr, qpact.Vh[hp.PARAMETER])

# generate true data
u_true = qpact.pde.generate_state()
x_true = [u_true, m_fun_true.vector(), None]
qpact.pde.solveFwd(u_true, x_true)
u_fun_true = hp.vector2Function(u_true, qpact.Vh[hp.STATE])

# add noise to the observations
noisy_data = dl.project(u_fun_true*ufl.exp(m_fun_true), qpact.Vh[hp.STATE])
hp.parRandom.normal_perturb(np.sqrt(NOISE_VARIANCE), noisy_data.vector())
noisy_data.rename("data", "data")

# visualization
hp.nb.plot(m_fun_true)
plt.savefig(os.path.join(FIG_DIR, "true_param.png"))
plt.close()

hp.nb.plot(u_fun_true)
plt.savefig(os.path.join(FIG_DIR, "true_state.png"))
plt.close()

hp.nb.plot(noisy_data)
plt.savefig(os.path.join(FIG_DIR, "noisy_data.png"))
plt.close()

# set up the misfit
misfit_form = DiffusionApproximationMisfitForm(noisy_data, dl.Constant(NOISE_VARIANCE))
misfit = hp.NonGaussianContinuousMisfit(qpact.Vh, misfit_form)

##################################################
# setup the Gaussian prior, solve for the MAP point
##################################################
rprint(COMM, SEP)
rprint(COMM, "Inverting for the MAP point with a Gaussian prior.")
rprint(COMM, SEP)

m0 = dl.interpolate(dl.Constant(np.log(mu_a_background)), qpact.Vh[hp.PARAMETER])  # remember, the physical parameter is exp(m)
gaussian_prior = hp.BiLaplacianPrior(qpact.Vh[hp.PARAMETER], GAMMA, DELTA)

model = hp.Model(qpact.pde, gaussian_prior, misfit)

xg = [model.generate_vector(hp.STATE), m0.vector(), model.generate_vector(hp.ADJOINT)]

# instantiate the solver and solve
parameters = hp.ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1e-6
parameters["abs_tolerance"] = 1e-9
parameters["max_iter"]      = 500
parameters["cg_coarse_tolerance"] = 5e-1
parameters["globalization"] = "LS"
parameters["GN_iter"] = 20
if COMM.rank != 0:
    parameters["print_level"] = -1
    
solver = hp.ReducedSpaceNewtonCG(model, parameters)
xg = solver.solve(xg)

mg_fun = hp.vector2Function(xg[hp.PARAMETER], qpact.Vh[hp.PARAMETER], name = "m_map")
ug_fun = hp.vector2Function(xg[hp.STATE], qpact.Vh[hp.STATE], name = "u_map")

obs_fun = dl.project(ug_fun*mg_fun, qpact.Vh[hp.STATE])
obs_fun.rename("obs", "obs")

# visualization
with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "param_reconstruction_gaussian.xdmf")) as fid:
    fid.write(mg_fun)

with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "state_reconstruction_gaussian.xdmf")) as fid:
    fid.write(ug_fun)

with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "obs_reconstruction_gaussian.xdmf")) as fid:
    fid.write(obs_fun)

##################################################
# setup the TV prior and primal-dual solver
##################################################
Vhw = dl.VectorFunctionSpace(qpact.mesh, "DG", 1)
Vhwnorm = dl.FunctionSpace(qpact.mesh, "DG", 0)
tvprior = hp.TVPrior(qpact.Vh[hp.PARAMETER], Vhw, Vhwnorm, alpha=ALPHA, beta=BETA, peps=PEPS*ALPHA)

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK
if COMM.rank != 0:
    solver_params["print_level"] = -1

model = hp.ModelNS(qpact.pde, misfit, None, tvprior, which=TVONLY)
solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

##################################################
# perform an L-Curve analysis for the TV regularization parameter (if requested)
##################################################
if DO_LCURVE:
    rprint(COMM, SEP)
    rprint(COMM, "Running L-Curve analysis to determine TV regularization coefficient.")
    rprint(COMM, SEP)
    
    ALPHAS = np.logspace(4, -2, num=16, base=10)
    misfits = np.zeros_like(ALPHAS)
    regs = np.zeros_like(ALPHAS)
    
    for i, alpha in enumerate(ALPHAS):
        print(f"\nRunning with alpha:\t {alpha:.2e}")
        nsprior = hp.TVPrior(qpact.Vh[hp.PARAMETER], Vhw, Vhwnorm, alpha, BETA, peps=PEPS*alpha)
        
        # set up the model describing the inverse problem
        model = hp.ModelNS(qpact.pde, misfit, None, nsprior, which=TVONLY)
        solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)
        
        # solve the system
        start = time.perf_counter()
        x = solver.solve([None, m0.vector(), None, None])
        if VERBOSE:
            print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
            print(f"Solver convergence criterion:\t{solver.termination_reasons[solver.reason]}")
            print(f"Number of Newton iterations:\t{solver.it}")
            print(f"Total number of CG iterations:\t{solver.total_cg_iter}")
        
        _, _, regs[i], misfits[i] = model.cost(x)
        
    fig, ax = plt.subplots()
    plt.loglog(misfits, regs / ALPHAS, 'x') #todo, might need to fix this
    plt.xlabel("Data Fidelity")
    plt.ylabel("TV Regularization")
    plt.title("L-Curve for qPACT Diffusion Approximation TV Denoising")
    [ax.annotate(fr"$\alpha$={ALPHAS[i]:.2e}", (misfits[i], regs[i]/ALPHAS[i])) for i in range(len(ALPHAS))]
    plt.savefig(os.path.join(FIG_DIR, "tv_poisson_lcurve.png"))

##################################################
# solve the inverse problem with total variation regularization
##################################################
rprint(COMM, SEP)
rprint(COMM, "Inverting for the parameter with TV regularization.")
rprint(COMM, SEP)

xtv = solver.solve([None, m0.vector(), None, None])
mtv_fun = hp.vector2Function(xtv[hp.PARAMETER], qpact.Vh[hp.PARAMETER], name = "m_map")
utv_fun = hp.vector2Function(xtv[hp.STATE], qpact.Vh[hp.STATE], name = "u_map")
obstv_fun = dl.project(utv_fun*mtv_fun, qpact.Vh[hp.STATE])
obstv_fun.rename("obs", "obs")

# visualization
with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "param_reconstruction_tv.xdmf")) as fid:
    fid.write(mtv_fun)

with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "state_reconstruction_tv.xdmf")) as fid:
    fid.write(utv_fun)

with dl.XDMFFile(COMM, os.path.join(FIG_DIR, "obs_reconstruction_tv.xdmf")) as fid:
    fid.write(obstv_fun)
