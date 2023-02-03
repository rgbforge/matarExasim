"""Module to run the 1D sqrt formulation of GITM (1D in altitude)
Latest update: Jan 17th, 2022. [OI]
"""
# import external modules
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_earth
import astropy.units as u
from Applications.SpaceWeather.SW1D_sqrt.python.pdeparamsMSIS import pdeparams
import time

# start timer
start_time = time.time()

# Add Exasim package to python search path
cdir = os.getcwd()
ii = cdir.find("Exasim")
exec(open(cdir[0:(ii + 6)] + "/Installation/setpath.py").read())

# import internal modules
import Preprocessing, Postprocessing, Gencode

# Create pde object and mesh object
pde, mesh = Preprocessing.initializeexasim()

# Define a PDE model: governing equations and boundary conditions
pde['model'] = "ModelD"  # ModelC, ModelD, ModelW
pde['modelfile'] = "pdemodelMSIS"  # name of a file defining the PDE model

# Choose computing platform and set number of processors
pde['platform'] = "cpu"
pde['mpiprocs'] = 1  # number of MPI processors

# specify model input parameters for summer solstice.
parameters = {
    "planet": "Earth",  # Planet
    "coord": "2",  # (0:Cartesian, 1:cylindrical, 2:spherical)
    "date": "2013-01-01 00:00:00",  # read in data for this day, i.e. F10.7 measurements. year-month-day hr:min:sec
    "t_step": 5 * u.s,  # time step (seconds)
    "t_simulation": 0.002 * u.d,  # length of simulation (days)
    "frequency_save": 30 * u.min,  # frequency of data (minutes)
    "t_restart": 0*u.s,  # restart at given time step (second)
    "longitude": -117.1611*u.deg,  # longitude coordinates # todo: try San Diego coords (lat=32.7157, lon=-117.1611)
    "latitude": 32.7157*u.deg,  # latitude coordinates
    "euv_efficiency": 0.21,  # EUV efficiency # todo: what are the units?
    "altitude_lower": (100*u.km).to(u.m),  # computational domain altitude lower bound (meters)
    "altitude_upper": (600*u.km).to(u.m),  # computational domain altitude upper bound (meters)
    "lambda0": 1e-9 * u.m,  # reference euv wavelength (meter)
    "EUV_input_file_directory": "inputs/euv.csv",  # EUV input file location
    "orbits_input_file_directory": "inputs/orbits.csv",  # orbits input file location
    "neutrals_input_file_directory": "inputs/neutrals.csv",  # neutrals input file location
    "gamma": 5/3,  # ratio of specific heats
    "exp_mu": 0.5,  # exponential of reference mu
    "exp_kappa": 0.75,  # exponential of reference kappa
    "tau_a": 5,  # parameter relating to solver. # todo: define this better.
    "ref_mu_scale": 2,  # multiply the reference value of the dynamic viscosity by this value
    "ref_kappa_scale": 0.4,  # multiply the reference value of the thermal conductivity by this value
    "ref_rho_scale": 1,  # multiply the reference value of the density by this value
    "p_order": 2,  # order of polynomial in solver
    "t_order": 2,  # grid parameter in solver # todo: understand this better.
    "n_stage": 2,  # grid parameter in solver # todo: understand this better.
    "ext_stab": 1,  # solver parameter # todo: understand this better.
    "tau": 0.0,  # discontinuous galerkin stabilization parameter # todo: Jordi, what is tau_a vs tau?
    "GMRES_restart": 29,  # number of GMRES (linear solver) restarts
    "linear_solver_tol": 1e-16,  # GMRES (linear solver) solver tolerance
    "linear_solver_iter": 30,  # GMRES (linear solver) solver iterations
    "pre_cond_matrix_type": 2,  # preconditioning type
    "newton_tol": 1e-10,  # newton iterations
    "mat_vec_tol": 1e-7,  # todo: define
    "rb_dim": 8,  # todo: define
    "resolution": 16,  # set one-dimensional mesh resolution
    "boundary_epsilon": 1e-3,  # boundary epsilon for mesh
    "F10p7_uncertainty": 10 * (1E-22 * u.W*u.Hz/(u.m**2)),  # added factor F10.7 cm radio emissions
    # measured in solar flux units uncertainty
    "F10p7-81_uncertainty": 1 * (1E-22 * u.W*u.Hz/(u.m**2)),  # F10.7 of the last
    # 81-days measured in solar flux units uncertainty
    "chemical_species": ["O", "N2", "O2", "He"],  # chemical species we are solving for
    "nu_eddy": 100,  # eddy viscosity
    "alpha_eddy": 35,  # eddy conductivity
    "n_radial_MSIS": 101,   # number of mesh points in the radial direction for MSIS simulation
    "n_longitude_MSIS": 72,  # number of mesh points in the longitude direction for MSIS simulation
    "n_latitude_MSIS": 35,  # number of mesh points in the longitude direction for MSIS simulation
    "initial_dr": 0.5 * u.km  # mesh offset used to measure derivatives in initial condition 1D pressure.
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

# run source code and save solution in dataout folder.
runstr = Gencode.runcode(pde, 1)

# save time it took to run in sec.
np.savetxt("time.txt", float(time.time() - start_time))

# get solution from output files in dataout folder
sol = Postprocessing.fetchsolution(pde, master, dmd, cdir + "/dataout")


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
phys_grid = ((grid * float(H0) - R_earth.value) * u.m).to(u.km)

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
#plt.savefig("figs/sol_example.png")
plt.show()
