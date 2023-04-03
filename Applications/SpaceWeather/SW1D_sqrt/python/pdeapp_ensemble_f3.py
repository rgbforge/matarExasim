"""Module to run the 1D sqrt formulation of GITM (1D in altitude)

Latest update: April 3rd, 2022. [OI]
"""
# import external modules
import os
import numpy as np
import matplotlib.pyplot as plt
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

samples = np.load(os.path.dirname(cdir) + "/sensitivity/samples/test_mfmc_50.npy")

for ii in range(np.shape(samples)[0]):
    # Define a PDE model: governing equations and boundary conditions
    pde['model'] = "ModelD"  # ModelC, ModelD, ModelW
    pde['modelfile'] = "pdemodelMSIS"  # name of a file defining the PDE model

    # Choose computing platform and set number of processors
    pde['platform'] = "cpu"
    pde['mpiprocs'] = 1  # number of MPI processors

    # specify model input parameters for summer solstice.
    parameters = {
        "planet": "Earth",  # Planet
        "coord": "2",  # (0 => cartesian, 1 => cylindrical, 2 => spherical)
        # date formatting: year-month-day hr-min-sec
        "date": "2002-03-18 00:00:00",  # read in data for this day, i.e. F10.7 measurements. year-month-day hr:min:sec
        "t_step": 20 * u.s,  # time step (seconds)
        "t_simulation": 3 * u.d,  # length of simulation (days)
        "frequency_save": 30 * u.min,  # frequency of data (minutes)
        "t_restart": 0,  # restart at given time step (discrete value)
        "longitude": -117.1611 * u.deg,  # longitude coordinates (These are San Diego coordinates!)
        "latitude": 32.7157 * u.deg,  # latitude coordinates (These are San Diego coordinates!)
        "euv_efficiency": samples[ii, 2],  # EUV efficiency
        "altitude_lower": (100 * u.km).to(u.m),  # computational domain altitude lower bound (meters)
        "altitude_upper": (600 * u.km).to(u.m),  # computational domain altitude upper bound (meters)
        "lambda0": 1e-9 * u.m,  # reference euv wavelength (meter)
        "EUV_input_file_directory": str(os.getcwd()) + "/inputs/euv.csv",  # EUV input file location
        "orbits_input_file_directory": str(os.getcwd()) + "/inputs/orbits.csv",  # orbits input file location
        "neutrals_input_file_directory": str(os.getcwd()) + "/inputs/neutrals.csv",  # neutrals input file location
        "gamma": 5 / 3,  # ratio of specific heats
        "exp_mu": samples[ii, 4],  # exponential of reference mu
        "exp_kappa": samples[ii, 3],  # exponential of reference kappa
        "tau_a": 5,  # parameter relating to solver. # todo: define this better.
        "ref_mu_scale": samples[ii, 0],  # multiply the reference value of the dynamic viscosity by this value
        "ref_kappa_scale": samples[ii, 1],  # multiply the reference value of the thermal conductivity by this value
        "p_order": 2,  # order of polynomial in solver
        "t_order": 2,  # Runge-Kutta integrator order.
        "n_stage": 2,  # Runge-Kutta number of stages order.
        "resolution": 30,  # set one-dimensional mesh resolution
        "ext_stab": 1,  # solver parameter # todo: understand this better.
        "tau": 0.0,  # discontinuous galerkin stabilization parameter # todo: Jordi, what is tau_a vs tau?
        "GMRES_restart": 29,  # number of GMRES (linear solver) restarts
        "linear_solver_tol": 1e-16,  # GMRES (linear solver) solver tolerance
        "linear_solver_iter": 40,  # GMRES (linear solver) solver iterations
        "pre_cond_matrix_type": 2,  # preconditioning type
        "newton_tol": 1e-16,  # newton tolerance
        "newton_iter": 2,  # newton iterations
        "mat_vec_tol": 1e-6,  # todo: define
        "rb_dim": 8,  # todo: define
        "boundary_epsilon": 1e-3,  # boundary epsilon for mesh
        "F10p7_uncertainty": samples[ii, 5],  # added factor F10.7 cm radio emissions
        # measured in solar flux units uncertainty
        "F10p7-81_uncertainty": samples[ii, 6],  # F10.7 of the last
        # 81-days measured in solar flux units uncertainty
        "chemical_species": ["O", "N2", "O2", "He"],  # chemical species we are solving for
        "nu_eddy": samples[ii, 7],  # eddy viscosity
        "alpha_eddy": samples[ii, 8],  # eddy conductivity
        "n_radial_MSIS": 101,  # number of mesh points in the radial direction for MSIS simulation
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
    dir_name = os.path.dirname(cdir) + "/sensitivity/ensemble/" + str(fidelity) + "/sample_" + str(ii) + "/"
    np.save(dir_name + "pde.npy", pde)
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy["boundaryexpr"] = None
    np.save(dir_name + "mesh.npy", mesh_copy)
    np.save(dir_name + "master.npy", master)
    np.save(dir_name + "dmd.npy", dmd)
    np.save(dir_name + "parameters.npy", parameters)

    # generate source codes and store them in app folder
    # this is only when you change the pdemodel file. so we do not need this anymore.
    gencode(pde)

    # compile source codes to build an executable file and store it in app folder
    compilerstr = compilecode(pde)

    # run source code and save solution in dataout folder.
    runstr = runcode(pde, 1)

    # save time it took to run in sec.
    np.savetxt(dir_name + "time.txt", np.array([time.time() - start_time]))

    # get solution from output files in dataout folder
    # solution dimensions: (s1) x (s2) x (s3) x  (s4)
    # s1 => number of discretization points per element
    # s2 => number of components of your solution :
    # log(rho), sqrt(rho)*v, sqrt(rho)*T, and their corresponding derivatives (d·/dx)
    # s3 => number of elements of the grid (resolution)
    # s4 => number of saved time steps.
    sol = fetchsolution(pde, master, dmd, os.getcwd() + "/dataout")
    np.save(dir_name + "sol.npy", sol)
