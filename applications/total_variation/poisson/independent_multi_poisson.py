import os
import sys
import time

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp

from models import PoissonBox, splitCircle
from utils import parameter2NoisyObservations

## constants, initializations
VERBOSE = True
TVONLY = [True, False, True]
NOISE_LEVEL = 0.02
ALPHA = 1e2  # regularization parameter, picked from L-curve
BETA = 1e-3
PEPS = 0.5  # mass matrix scaling in preconditioner
MAX_ITER = 1000
CG_MAX_ITER = 75
MAX_BACKTRACK = 25
DO_LCURVE = False
os.makedirs("figs/imp", exist_ok=True)  # ensure figure directory exists
os.makedirs("mesh", exist_ok=True)  # ensure mesh directory exists
NDIM = 64

COMM = dl.MPI.comm_world

C = [0.5, 0.5]
R = 0.4
VL = np.log(6.)
VR = np.log(2.)
VO = np.log(10.)
CLIM = [0.5, 3]  # limits for plotting

##################################################
# Set up the first poisson problem
##################################################
p1 = PoissonBox(NDIM)
p1.setupMesh()
p1.setupFunctionSpaces()
p1.setupPDE()

# set up the true parameter
expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p1.mtrue = dl.interpolate(expr, p1.Vh[hp.PARAMETER])

# set up observation operator for the top right corner
xx = np.linspace(0.5, 1.0, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B1 = hp.assemblePointwiseObservation(p1.Vh[hp.STATE], targets)

# generate noisy observations
p1.p2o = parameter2NoisyObservations(p1.pde, p1.mtrue.vector(), NOISE_LEVEL, B1)
p1.p2o.generateNoisyObservations()

# set up the misfit observation operator
p1.misfit = hp.DiscreteStateObservation(B=B1, data=p1.p2o.noisy_data, noise_variance=p1.p2o.noise_std_dev**2)

##################################################
# Perform L-Curve analysis for first problem
##################################################
# set up the function spaces for the TV prior
Vhm = p1.Vh[hp.PARAMETER]
Vhw = dl.VectorFunctionSpace(p1.mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(p1.mesh, 'DG', 0)

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK

# initial guess (zero)
m0 = p1.pde.generate_parameter()
m0.zero()

# run the l-curve analysis
if DO_LCURVE:
    ALPHAS = np.logspace(4, -2, num=16, base=10)
    misfits = np.zeros_like(ALPHAS)
    regs = np.zeros_like(ALPHAS)
    
    for i, alpha in enumerate(ALPHAS):
        print(f"\nRunning with alpha:\t {alpha:.2e}")
        nsprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, alpha, BETA, peps=PEPS*alpha)
        
        # set up the model describing the inverse problem
        model = hp.ModelNS(p1.pde, p1.misfit, None, nsprior, which=TVONLY)
        solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)
        
        # solve the system
        start = time.perf_counter()
        x = solver.solve([None, m0, None, None])
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
    plt.title("L-Curve for Poisson TV Denoising")
    [ax.annotate(fr"$\alpha$={ALPHAS[i]:.2e}", (misfits[i], regs[i]/ALPHAS[i])) for i in range(len(ALPHAS))]
    plt.savefig("figs/imp/tv_poisson_lcurve.png")

##################################################
# Set up the second poisson problem
##################################################
p2 = PoissonBox(NDIM)
p2.setupMesh()
p2.setupFunctionSpaces()
p2.setupPDE()

# set up the true parameter
expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VR, vr=VL, vo=VO)
p2.mtrue = dl.interpolate(expr, p2.Vh[hp.PARAMETER])

# set up observation operator for the top right corner
xx = np.linspace(0.02, 1.0, 50, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B2 = hp.assemblePointwiseObservation(p2.Vh[hp.STATE], targets)

# generate noisy observations
p2.p2o = parameter2NoisyObservations(p2.pde, p2.mtrue.vector(), NOISE_LEVEL, B2)
p2.p2o.generateNoisyObservations()

# set up the misfit observation operator
p2.misfit = hp.DiscreteStateObservation(B=B2, data=p2.p2o.noisy_data, noise_variance=p2.p2o.noise_std_dev**2)

##################################################
# Set up the third poisson problem
##################################################
p3 = PoissonBox(NDIM)
p3.setupMesh()
p3.setupFunctionSpaces()
p3.setupPDE()

# set up the true parameter
expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p3.mtrue = dl.interpolate(expr, p3.Vh[hp.PARAMETER])

# set up observation operator for the top right corner
xx = np.linspace(0.02, 0.5, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B3 = hp.assemblePointwiseObservation(p3.Vh[hp.STATE], targets)

# generate noisy observations
p3.p2o = parameter2NoisyObservations(p3.pde, p3.mtrue.vector(), NOISE_LEVEL, B3)
p3.p2o.generateNoisyObservations()

# set up the misfit observation operator
p3.misfit = hp.DiscreteStateObservation(B=B3, data=p3.p2o.noisy_data, noise_variance=p3.p2o.noise_std_dev**2)

##################################################
# Visualization
##################################################

# Write out the mesh
MESHFPATH = os.path.join("mesh", "unitsquare.xdmf")
with dl.XDMFFile(MESHFPATH) as fid:
    fid.write(p1.Vh[hp.STATE].mesh())

with dl.XDMFFile(COMM, "figs/imp/p1_mtrue.xdmf") as fid:
    fid.write(p1.mtrue)

with dl.XDMFFile(COMM, "figs/imp/p2_mtrue.xdmf") as fid:
    fid.write(p2.mtrue)

with dl.XDMFFile(COMM, "figs/imp/p3_mtrue.xdmf") as fid:
    fid.write(p3.mtrue)
    
# solve for the true states and plot
xtmp = [p1.pde.generate_state(), p1.mtrue.vector(), None]
p1.pde.solveFwd(xtmp[hp.STATE], xtmp)

uf = dl.Function(p1.Vh[hp.STATE])
uf.vector().zero()
uf.vector().axpy(1., xtmp[hp.STATE])

with dl.XDMFFile(COMM, "figs/imp/p1_utrue.xdmf") as fid:
    fid.write(uf)

xtmp = [p2.pde.generate_state(), p2.mtrue.vector(), None]
p2.pde.solveFwd(xtmp[hp.STATE], xtmp)

uf.vector().zero()
uf.vector().axpy(1., xtmp[hp.STATE])

with dl.XDMFFile(COMM, "figs/imp/p2_utrue.xdmf") as fid:
    fid.write(uf)

xtmp = [p3.pde.generate_state(), p3.mtrue.vector(), None]
p3.pde.solveFwd(xtmp[hp.STATE], xtmp)

uf.vector().zero()
uf.vector().axpy(1., xtmp[hp.STATE])

with dl.XDMFFile(COMM, "figs/imp/p3_utrue.xdmf") as fid:
    fid.write(uf)

##################################################
# Solve the problems individually
##################################################
# set up the function spaces for the TV prior
Vhm = p1.Vh[hp.PARAMETER]
Vhw = dl.VectorFunctionSpace(p1.mesh, 'DG', 0)
Vhwnorm = dl.FunctionSpace(p1.mesh, 'DG', 0)

# set up the solver parameters
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK
if COMM.rank != 0:
    solver_params["print_level"] = -1

# set up the model describing the inverse problem
tvprior1 = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
model1 = hp.ModelNS(p1.pde, p1.misfit, None, tvprior1, which=TVONLY)
solver1 = hp.ReducedSpacePDNewtonCG(model1, parameters=solver_params)

m0 = p1.pde.generate_parameter()
m0.zero()
start = time.perf_counter()
x1 = solver1.solve([None, m0, None, None])
if VERBOSE:
    print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
    print(f"Solver convergence criterion:\t{solver1.termination_reasons[solver1.reason]}")
    print(f"Number of Newton iterations:\t{solver1.it}")
    print(f"Total number of CG iterations:\t{solver1.total_cg_iter}")
_, _, reg1, misfit1 = model1.cost(x1)

# set up and solve the second problem
tvprior2 = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
model2 = hp.ModelNS(p2.pde, p2.misfit, None, tvprior2, which=TVONLY)
solver2 = hp.ReducedSpacePDNewtonCG(model2, parameters=solver_params)

m0 = p2.pde.generate_parameter()
m0.zero()
start = time.perf_counter()
x2 = solver2.solve([None, m0, None, None])
if VERBOSE:
    print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
    print(f"Solver convergence criterion:\t{solver2.termination_reasons[solver2.reason]}")
    print(f"Number of Newton iterations:\t{solver2.it}")
    print(f"Total number of CG iterations:\t{solver2.total_cg_iter}")
_, _, reg2, misfit2 = model2.cost(x2)

# set up and solve the third problem
tvprior3 = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
model3 = hp.ModelNS(p3.pde, p3.misfit, None, tvprior3, which=TVONLY)
solver3 = hp.ReducedSpacePDNewtonCG(model3, parameters=solver_params)

m0 = p3.pde.generate_parameter()
m0.zero()
start = time.perf_counter()
x3 = solver3.solve([None, m0, None, None])
if VERBOSE:
    print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
    print(f"Solver convergence criterion:\t{solver3.termination_reasons[solver3.reason]}")
    print(f"Number of Newton iterations:\t{solver3.it}")
    print(f"Total number of CG iterations:\t{solver3.total_cg_iter}")
_, _, reg3, misfit3 = model3.cost(x3)

##################################################
# Visualization
##################################################
mf = dl.Function(p1.Vh[hp.PARAMETER])
mf.vector().zero()
mf.vector().axpy(1., x1[hp.PARAMETER])
with dl.XDMFFile(COMM, "figs/imp/p1_mreconstruct.xdmf") as fid:
    fid.write(mf)

uf.vector().zero()
uf.vector().axpy(1., x1[hp.STATE])
with dl.XDMFFile(COMM, "figs/imp/p1_infer_state.xdmf") as fid:
    fid.write(uf)

mf.vector().zero()
mf.vector().axpy(1., x2[hp.PARAMETER])
with dl.XDMFFile(COMM, "figs/imp/p2_mreconstruct.xdmf") as fid:
    fid.write(mf)

uf.vector().zero()
uf.vector().axpy(1., x2[hp.STATE])
with dl.XDMFFile(COMM, "figs/imp/p2_infer_state.xdmf") as fid:
    fid.write(uf)

mf.vector().zero()
mf.vector().axpy(1., x3[hp.PARAMETER])
with dl.XDMFFile(COMM, "figs/imp/p3_mreconstruct.xdmf") as fid:
    fid.write(mf)
    
uf.vector().zero()
uf.vector().axpy(1., x3[hp.STATE])
with dl.XDMFFile(COMM, "figs/imp/p3_infer_state.xdmf") as fid:
    fid.write(uf)
