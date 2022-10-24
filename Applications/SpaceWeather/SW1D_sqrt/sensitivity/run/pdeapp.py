"""Module to run the 1D sqrt formulation of GITM (1D in altitude)
Latest update: Oct 13th, 2022. [OI]
"""
# import external modules
import os, sympy
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_earth
import astropy.units as u
from pdeparams import pdeparams

# Add Exasim package to python search path
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

# specify model input parameters for summer solstice.
parameters = {
    "planet": "Earth",  # Planet
    "species": "O",  # Set species (or "air" for mixture) # todo: (oxygen?)
    "coord": "2",  # (0:Cartesian, 1:cylindrical, 2:spherical)
    "t_step": 5 * u.s,  # time step (seconds)
    "t_simulation": 2 * u.d,  # length of simulation (days)
    "frequency_save": 30 * u.min,  # frequency of data (minutes)
    "t_restart": 0*u.s,  # restart at given time step (second)
    "longitude": 0*u.deg,  # longitude coordinates #todo: try San Diego coords (long=32.7, lat=360-117.16)
    "latitude": 0*u.deg,  # latitude coordinates
    "euv_efficiency": 1.2,  # EUV efficiency # todo: what are the units?
    "altitude_lower": (100*u.km).to(u.m),  # computational domain altitude lower bound (meters)
    "altitude_upper": (750*u.km).to(u.m),  # computational domain altitude upper bound (meters)
    "lambda0": 1e-9 * u.m,  # reference euv wavelength (meter)
    "temp_lower": 200 * u.K,  # temperature at the lower bound (kelvin)
    "temp_upper": 1000 * u.K,  # temperature at the upper bound (kelvin)
    "EUV_input_file_directory": "../inputs/euv.csv",  # EUV input file location
    "orbits_input_file_directory": "../inputs/orbits.csv",  # orbits input file location
    "neutrals_input_file_directory": "../inputs/neutrals.csv",  # neutrals input file location
    "gamma": 5/3,  # ratio of specific heats
    "reference_temp_lower": 1,  # reference value for temperature at the lower boundary
    "exp_mu": 0.5,  # exponential of reference mu
    "tau_a": 10,  # parameter relating to solver. #todo: define this better.
    "ref_mu_scale": 1,  # multiply the reference value of the dynamic viscosity by this value
    "ref_kappa_scale": 1,  # multiply the reference value of the thermal conductivity by this value
    "ref_rho_scale": 1,  # multiply the reference value of the density by this value
    "p_order": 2,  # order of polynomial in solver
    "t_order": 2,  # grid parameter in solver # todo: understand this better.
    "n_stage": 2,  # grid parameter in solver # todo: understand this better.
    "ext_stab": 1,  # solver parameter # todo: understand this better.
    "tau": 0.0,  # discontinuous galerkin stabilization parameter
    "GMRES_restart": 29,  # number of GMRES (linear solver) restarts
    "linear_solver_tol": 1e-16,  # GMRES (linear solver) solver tolerance
    "linear_solver_iter": 30,  # GMRES (linear solver) solver iterations
    "pre_cond_matrix_type": 2,  # preconditioning type
    "newton_tol": 1e-10,  # newton iterations
    "mat_vec_tol": 1e-7,  # todo: define
    "rb_dim": 8,  # todo: define
    "resolution": 16,  # set mesh resolution
    "boundary_epsilon": 1e-3  # boundary epsilon for mesh
}

# run executable file to compute solution and store it in dataout folder
pde, mesh = pdeparams(pde=pde, mesh=mesh, parameters=parameters)

# search compilers and set options
pde = Gencode.setcompilers(pde)

# generate input files and store them in datain folder
pde, mesh, master, dmd = Preprocessing.preprocessing(pde, mesh)

# generate source codes and store them in app folder
Gencode.gencode(pde)

# compile source codes to build an executable file and store it in app folder
compilerstr = Gencode.compilecode(pde)

# todo: what is 1 here?
runstr = Gencode.runcode(pde, 1)

# get solution from output files in dataout folder
sol = Postprocessing.fetchsolution(pde, master, dmd, "dataout")

# generate mesh nodes
dgnodes = Preprocessing.createdgnodes(mesh["p"],
                                      mesh["t"],
                                      mesh["f"],
                                      mesh["curvedboundary"],
                                      mesh["curvedboundaryexpr"],
                                      pde["porder"])
# get parameters
rho0 = pde["physicsparam"][6]
H0 = pde["physicsparam"][11]

fig, ax = plt.subplots(figsize=(4, 3))
rho = np.ndarray.flatten(np.exp(sol[:, 0, :, -1]).T)
vr = np.ndarray.flatten(sol[:, 1, :, -1].T) / np.sqrt(rho)
T = np.ndarray.flatten(sol[:, 2, :, -1].T) / np.sqrt(rho)
grid = np.ndarray.flatten(dgnodes[:, 0, :].T)
phys_grid = ((grid * H0 - R_earth.value) * u.m).to(u.km)

ax.plot(phys_grid.T, rho.T, label=r"$\rho$")
ax.plot(phys_grid.T, vr.T, label=r"$v_{r}$")
ax.plot(phys_grid.T, T.T, label=r"$T$")

ax.plot(phys_grid.T, np.ndarray.flatten(np.exp(sol[:, 0, :, -1]).T), label=r"$u_{1}$")
ax.plot(phys_grid.T, np.ndarray.flatten(sol[:, 1, :, -1].T), label=r"$u_{2}$")
ax.plot(phys_grid.T, np.ndarray.flatten(sol[:, 2, :, -1].T), label=r"$u_{3}$")

ax.set_xlabel("Altitude")
ax.legend()
ax.set_xticks([100, 200, 300, 400, 500])
plt.tight_layout()
plt.savefig("figs/sol_example.png")
plt.show()
