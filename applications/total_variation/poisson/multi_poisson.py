import os
import sys
import time

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../../") )
import hippylib as hp

from models import MultiPoissonBox, splitCircle
from utils import add_noise_to_observations

# constants, initializations
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
os.makedirs("figs/multi_poisson", exist_ok=True)  # ensure figure directory exists
os.makedirs("mesh", exist_ok=True)  # ensure mesh directory exists
NDIM = 64
NPOISSON = 3

COMM = dl.MPI.comm_world

# parameters for the split circle
C = [0.5, 0.5]
R = 0.4
VL = np.log(6.)
VR = np.log(2.)
VO = np.log(10.)
CLIM = [0.5, 3]  # limits for plotting

# setup the multiple poisson problem
mp = MultiPoissonBox(NPOISSON, NDIM)
mp.setupMesh()
mp.setupFunctionSpaces()
mp.setupPDE()

phys_dim = mp.pde.Vh[hp.STATE].mesh().topology().dim()  # the physical dimension of the mesh

# assign the true parameter(s)
expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p1mtrue = dl.interpolate(expr, mp.Vhm0)

expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VR, vr=VL, vo=VO)
p2mtrue = dl.interpolate(expr, mp.Vhm0)

expr = splitCircle(cx=C[0], cy=C[1], r=R, vl=VL, vr=VR, vo=VO)
p3mtrue = dl.interpolate(expr, mp.Vhm0)

mtrue = dl.Function(mp.Vh[hp.PARAMETER])
mp.assigner.assign(mtrue, [p1mtrue, p2mtrue, p3mtrue])

# solve the model forward
utrue = mp.pde.generate_state()
mp.pde.solveFwd(utrue, [utrue, mtrue.vector(), None])

##################################################
# Setup observation operators
##################################################

# set up observation operator for the top right corner
xx = np.linspace(0.5, 1.0, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B1 = hp.assemblePointwiseObservation(mp.pde.Vh[hp.STATE], targets)

# set up observation operator for the full domain
xx = np.linspace(0.02, 1.0, 50, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B2 = hp.assemblePointwiseObservation(mp.Vh[hp.STATE], targets)

# set up observation operator for the bottom left corner
xx = np.linspace(0.02, 0.5, 25, endpoint=False)
xv, yv = np.meshgrid(xx, xx)
targets = np.vstack([xv.ravel(), yv.ravel()]).T
print(f"Number of observation points: {targets.shape[0]}")
B3 = hp.assemblePointwiseObservation(mp.Vh[hp.STATE], targets)

# write out the mesh and true parameters
MESHFPATH = os.path.join("mesh", "unitsquare.xdmf")
with dl.XDMFFile(MESHFPATH) as fid:
    fid.write(mp.Vh[hp.STATE].mesh())

m1, m2, m3 = mtrue.split()

with dl.XDMFFile(COMM, "figs/multi_poisson/p1_mtrue.xdmf") as fid:
    fid.write(m1)

with dl.XDMFFile(COMM, "figs/multi_poisson/p2_mtrue.xdmf") as fid:
    fid.write(m2)

with dl.XDMFFile(COMM, "figs/multi_poisson/p3_mtrue.xdmf") as fid:
    fid.write(m3)

##################################################
# generate noisy observations, set up misfits
##################################################
obsops = [B1, B2, B3]
misfits = []
for idx, BB in enumerate(obsops):
    noisy_data, noise_std_dev = add_noise_to_observations(utrue.data[idx], NOISE_LEVEL, BB)
    misfits.append(hp.DiscreteStateObservation(B=BB, data=noisy_data, noise_variance=noise_std_dev**2))

misfit = hp.MultiStateMisfit(misfits)

##################################################
# setup V-TV prior and model
##################################################
Vhm = mp.Vh[hp.PARAMETER]
Vhw = dl.TensorFunctionSpace(mp.mesh, "DG", 0, shape=(NPOISSON, phys_dim))
Vhwnorm = dl.FunctionSpace(mp.mesh, "DG", 0)

tvprior = hp.TVPrior(Vhm, Vhw, Vhwnorm, ALPHA, BETA, peps=PEPS*ALPHA)
model = hp.ModelNS(mp.pde, misfit, None, tvprior, which=TVONLY)

##################################################
# setup the solver
##################################################
solver_params = hp.ReducedSpacePDNewtonCG_ParameterList()
solver_params["max_iter"] = MAX_ITER
solver_params["cg_max_iter"] = CG_MAX_ITER
solver_params["LS"]["max_backtracking_iter"] = MAX_BACKTRACK
if COMM.rank != 0:
    solver_params["print_level"] = -1

solver = hp.ReducedSpacePDNewtonCG(model, parameters=solver_params)

# solve the system
m0 = mp.pde.generate_parameter()
m0.zero()
start = time.perf_counter()

xsol = solver.solve([None, m0, None, None])

print(f"Time to solve: {(time.perf_counter()-start)/60:.2f} minutes")
print(f"Solver convergence criterion:\t{solver.termination_reasons[solver.reason]}")
print(f"Number of Newton iterations:\t{solver.it}")
print(f"Total number of CG iterations:\t{solver.total_cg_iter}")

# plot the reconstructed parameters
msol = hp.vector2Function(xsol[hp.PARAMETER], mp.Vh[hp.PARAMETER], name="m_reconstruct")

with dl.XDMFFile(COMM, "figs/multi_poisson/p1_mreconstruct.xdmf") as fid:
    fid.write(mp.split_component(msol, 0))

with dl.XDMFFile(COMM, "figs/multi_poisson/p2_mreconstruct.xdmf") as fid:
    fid.write(mp.split_component(msol, 1))

with dl.XDMFFile(COMM, "figs/multi_poisson/p3_mreconstruct.xdmf") as fid:
    fid.write(mp.split_component(msol, 2))

# plot the reconstructed states
uf = dl.Function(mp.Vh[hp.STATE])
uf.vector().zero()
uf.vector().axpy(1., xsol[hp.STATE].data[0])
with dl.XDMFFile(COMM, "figs/multi_poisson/p1_infer_state.xdmf") as fid:
    fid.write(uf)

uf.vector().zero()
uf.vector().axpy(1., xsol[hp.STATE].data[1])
with dl.XDMFFile(COMM, "figs/multi_poisson/p2_infer_state.xdmf") as fid:
    fid.write(uf)

uf.vector().zero()
uf.vector().axpy(1., xsol[hp.STATE].data[2])
with dl.XDMFFile(COMM, "figs/multi_poisson/p3_infer_state.xdmf") as fid:
    fid.write(uf)

# plot the true states
uf.vector().zero()
uf.vector().axpy(1., utrue.data[0])
with dl.XDMFFile(COMM, "figs/multi_poisson/p1_true_state.xdmf") as fid:
    fid.write(uf)

uf.vector().zero()
uf.vector().axpy(1., utrue.data[1])
with dl.XDMFFile(COMM, "figs/multi_poisson/p2_true_state.xdmf") as fid:
    fid.write(uf)

uf.vector().zero()
uf.vector().axpy(1., utrue.data[2])
with dl.XDMFFile(COMM, "figs/multi_poisson/p3_true_state.xdmf") as fid:
    fid.write(uf)
