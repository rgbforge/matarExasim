""" Plotting results... """
import numpy as np
import os, sympy
from numpy import *
from run.pdeparams import pdeparams
from astropy.constants import G, k_B, h, M_earth, R_earth
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# Add Exasim to python search path
cdir = os.getcwd()
ii = cdir.find("Exasim")
exec(open(cdir[0:(ii + 6)] + "/Installation/setpath.py").read())

# import internal modules
import Preprocessing, Postprocessing, Gencode, Mesh

# Create pde object and mesh object
pde, mesh = Preprocessing.initializeexasim()

# Define a PDE model: governing equations and boundary conditions
pde['model'] = "ModelD"  # ModelC, ModelD, ModelW
pde['modelfile'] = "pdemodel"  # name of a file defining the PDE model

# Choose computing platform and set number of processors
pde['platform'] = "cpu"
pde['mpiprocs'] = 1  # number of MPI processors
pde, mesh = pdeparams(pde, mesh)

EUV_vec = np.linspace(0.3, 1.5, 5)

sol1 = np.load("solutions/test_EUV/sol0.npy")

dgnodes = Preprocessing.createdgnodes(mesh["p"],
                                      mesh["t"],
                                      mesh["f"],
                                      mesh["curvedboundary"],
                                      mesh["curvedboundaryexpr"],
                                      pde["porder"])
rho0 = pde["physicsparam"][6]
H0 = pde["physicsparam"][11]

fig, ax = plt.subplots(figsize=(4, 3))
rho = np.ndarray.flatten(np.exp(sol1[:, 0, :, -1]).T)
vr = np.ndarray.flatten(sol1[:, 1, :, -1].T)/np.sqrt(rho)
T = np.ndarray.flatten(sol1[:, 2, :, -1].T)/np.sqrt(rho)
grid = np.ndarray.flatten(dgnodes[:, 0, :].T)
phys_grid = ((grid*H0 - R_earth.value)*u.m).to(u.km)

ax.plot(phys_grid.T, rho.T, label=r"$\rho$")
ax.plot(phys_grid.T, vr.T, label=r"$v_{r}$")
ax.plot(phys_grid.T, T.T, label=r"$T$")
ax.set_xlabel("Altitude")
ax.legend()
ax.set_xticks([100, 200, 300, 400, 500])
plt.tight_layout()
plt.savefig("sensitivity/figs/euv_sol1.png")
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
line_style_list = ["--", "-.", ">-", ":", "-"]
for ii in range(5):
    sol = np.load("sensitivity/solutions/test_EUV/sol" + str(ii) + ".npy")
    rho_phys = np.ndarray.flatten((np.exp(sol[:, 0, :, -1]) * rho0).T)
    ax.plot(rho_phys.T, phys_grid.T, line_style_list[ii],
            label=r"$\epsilon = $" + str(round(EUV_vec[ii], 3)), linewidth=3)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel(r"Neutral Density [kg/m$^3$]")
ax.legend()
ax.set_yticks([100, 200, 300, 400, 500])
ax.set_ylim(100, 500)
ax.set_xscale("log")
# Hide the right and top spines
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks([1e-9,  1e-12, 1e-15])
plt.tight_layout()
plt.savefig("sensitivity/figs/ensemble.png")
plt.show()