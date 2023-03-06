"""Module to run the 1D sqrt formulation of GITM (1D in altitude)

Latest update: March 2nd, 2022. [OI]
"""
# import external modules
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_earth
import astropy.units as u
from pdeparamsMSIS import pdeparams
import time
import copy
import sys

# Add Exasim package to python search path & others
cdir = os.getcwd()
ii = cdir.find("Exasim")
sys.path.append(cdir[:(ii + 6)] + "/src/Python/Preprocessing/")
from initializeexasim import initializeexasim
from Preprocessing import preprocessing
sys.path.append(cdir[:(ii + 6)] + "/src/Python/Gencode/")
from compilecode import compilecode
from Gencode import gencode
from setcompilers import setcompilers
from runcode import runcode
sys.path.append(cdir[:(ii + 6)] + "/src/Python/Postprocessing/")
from fetchsolution import fetchsolution

# save everything outside of dir.
os.chdir('../')

# start timer
start_time = time.time()

# Create pde object and mesh object
pde, mesh = initializeexasim()

# fidelity
fidelity = "f3"

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
    # date formatting: year-month-day hr-min-sec
    "date": "2002-03-18 00:00:00",  # read in data for this day, i.e. F10.7 measurements. year-month-day hr:min:sec
    "t_step": 20 * u.s,  # time step (seconds)
    "t_simulation": 3 * u.d,  # length of simulation (days)
    "frequency_save": 30 * u.min,  # frequency of data (minutes)
    "t_restart": 0,  # restart at given time step (discrete value)
    "longitude": 254.75*u.deg,  # longitude coordinates todo: changed to Boulder from SD.
    "latitude": 40.02*u.deg,  # latitude coordinates todo: changed to Boulder from SD.
    "euv_efficiency": 0.25,  # EUV efficiency
    "altitude_lower": (100*u.km).to(u.m),  # computational domain altitude lower bound (meters)
    "altitude_upper": (600*u.km).to(u.m),  # computational domain altitude upper bound (meters)
    "lambda0": 1e-9 * u.m,  # reference euv wavelength (meter)
    "EUV_input_file_directory": str(os.getcwd()) + "/inputs/euv.csv",  # EUV input file location
    "orbits_input_file_directory": str(os.getcwd()) + "/inputs/orbits.csv",  # orbits input file location
    "neutrals_input_file_directory": str(os.getcwd()) + "/inputs/neutrals.csv",  # neutrals input file location
    "gamma": 5/3,  # ratio of specific heats
    "exp_mu": 0.5,  # exponential of reference mu
    "exp_kappa": 0.69,  # exponential of reference kappa
    "tau_a": 5,  # parameter relating to solver. todo: define this better.
    "ref_mu_scale": 5,  # multiply the reference value of the dynamic viscosity by this value
    "ref_kappa_scale": 1,  # multiply the reference value of the thermal conductivity by this value
    "p_order": 2,  # order of polynomial in solver
    "t_order": 2,  # Runge-Kutta integrator order.
    "n_stage": 2,  # Runge-Kutta number of stages order.
    "resolution": 30,  # set one-dimensional mesh resolution
    "ext_stab": 1,  # solver parameter todo: understand this better.
    "tau": 0.0,  # discontinuous galerkin stabilization parameter todo: Jordi, what is tau_a vs tau?
    "GMRES_restart": 29,  # number of GMRES (linear solver) restarts
    "linear_solver_tol": 1e-16,  # GMRES (linear solver) solver tolerance
    "linear_solver_iter": 40,  # GMRES (linear solver) solver iterations
    "pre_cond_matrix_type": 2,  # preconditioning type
    "newton_tol": 1e-16,  # newton tolerance
    "newton_iter": 2,  # newton iterations
    "mat_vec_tol": 1e-8,  # todo: define
    "rb_dim": 8,  # todo: define
    "boundary_epsilon": 1e-3,  # boundary epsilon for mesh
    "F10p7_uncertainty": 0,  # added factor F10.7 cm radio emissions
    # measured in solar flux units uncertainty
    "F10p7-81_uncertainty": 0,  # F10.7 of the last
    # 81-days measured in solar flux units uncertainty
    "chemical_species": ["O", "N2", "O2", "He"],  # chemical species we are solving for
    "nu_eddy": 100,  # eddy viscosity
    "alpha_eddy": 10,  # eddy conductivity
    "n_radial_MSIS": 101,   # number of mesh points in the radial direction for MSIS simulation
    "n_longitude_MSIS": 72,  # number of mesh points in the longitude direction for MSIS simulation
    "n_latitude_MSIS": 35,  # number of mesh points in the longitude direction for MSIS simulation
    "initial_dr": 0.5 * u.km  # mesh offset used to measure derivatives in initial condition 1D pressure.
}

# run executable file to compute solution and store it in dataout folder
pde, mesh = pdeparams(pde=pde, mesh=mesh, parameters=parameters)

# search compilers and set options
pde = setcompilers(pde)

# generate input files and store them in datain folder
pde, mesh, master, dmd = preprocessing(pde, mesh)

# save model setup.
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/pde", pde)
mesh_copy = copy.deepcopy(mesh)
mesh_copy["boundaryexpr"] = None
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/mesh", mesh_copy)
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/master", master)
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/dmd", dmd)
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/parameters", parameters)

# generate source codes and store them in app folder
# this is only when you change the pdemodel file. so we do not need this anymore.
gencode(pde)

# compile source codes to build an executable file and store it in app folder
compilerstr = compilecode(pde)

# run source code and save solution in dataout folder.
runstr = runcode(pde, 1)

# save time it took to run in sec.
np.savetxt(os.getcwd() + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/time.txt", np.array([time.time() - start_time]))

# get solution from output files in dataout folder
# solution dimensions: (s1) x (s2) x (s3) x  (s4)
# s1 => number of discretization points per element
# s2 => number of components of your solution :
# log(rho), sqrt(rho)*v, sqrt(rho)*T, and their corresponding derivatives (d·/dx)
# s3 => number of elements of the grid (resolution)
# s4 => number of saved time steps.
sol = fetchsolution(pde, master, dmd, os.getcwd() + "/dataout")
np.save(os.path.dirname(cdir) + "/solutions_CHAMP_2002/Boulder/" + str(fidelity) + "/sol.npy", sol)

# get parameters
rho0 = pde["physicsparam"][19]
H0 = pde["physicsparam"][17]
T0 = pde["physicsparam"][18]

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(7, 7))
index = -1
# convert results to density, radial velocity, and temperature.
rho = np.exp(sol[:, 0, :, index]).flatten("F")
vr = sol[:, 1, :, index].flatten("F") / np.sqrt(rho)
T = (sol[:, 2, :, index].flatten("F") / np.sqrt(rho))
# computational grid.
computational_grid = mesh["dgnodes"].flatten("F")
# computational grid in km.
phys_grid = ((computational_grid * float(H0) - R_earth.value) * u.m).to(u.km)

ax[0].plot(phys_grid, rho*float(rho0))
ax[1].plot(phys_grid, vr)
ax[2].plot(phys_grid, T*float(T0))

ax[0].set_ylabel(r"$\rho$ [kg/$m^3$]")
ax[1].set_ylabel(r"$v_{r}$ [m/s]")
ax[2].set_ylabel(r"T [K]")

ax[2].set_xlabel("Altitude [km]")
ax[2].set_xticks([100, 200, 300, 400, 500, 600])
ax[2].set_xlim(100, 600)
ax[0].set_yscale("log")
plt.tight_layout()
plt.savefig(os.getcwd() + "/figs/GITM_1D_results_" + str(fidelity) + ".png", dpi=600)


fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
n_time_step = np.shape(sol)[-1]
rho_time_dependent = np.zeros((len(rho), n_time_step))
v_time_dependent = np.zeros((len(rho), n_time_step))
T_time_dependent = np.zeros((len(rho), n_time_step))
for ii in range(n_time_step):
    rho_time_dependent[:, ii] = np.exp(sol[:, 0, :, ii]).flatten("F")
    v_time_dependent[:, ii] = sol[:, 1, :, ii].flatten("F") / np.sqrt(rho_time_dependent[:, ii])
    T_time_dependent[:, ii] = sol[:, 2, :, ii].flatten("F") / np.sqrt(rho_time_dependent[:, ii])

pos = ax[0].imshow(X=rho_time_dependent*float(rho0), aspect="auto")
cbar = fig.colorbar(pos, ax=ax[0])
pos = ax[1].imshow(X=v_time_dependent, aspect="auto")
cbar = fig.colorbar(pos, ax=ax[1])
pos = ax[2].imshow(X=T_time_dependent*float(T0), aspect="auto")
cbar = fig.colorbar(pos, ax=ax[2])

ax[0].set_title(r"$\rho$ [kg/$m^3$]")
ax[1].set_title(r"$v_{r}$ [m/s]")
ax[2].set_title(r"T [K]")

ax[2].set_xlabel("time-interval")
ax[0].set_ylabel("Altitude")
ax[1].set_ylabel("Altitude")
ax[2].set_ylabel("Altitude")
plt.tight_layout()
plt.savefig(os.getcwd() + "/figs/GITM_1D_results_time_dependent_" + str(fidelity) + ".png", dpi=600)
plt.show()