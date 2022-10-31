""" Module to define the model input parameters driven by solar conditions.
Latest update: Oct 13th, 2022. [OI]
"""
import numpy as np
from pandas import read_csv
from math import ceil
from Applications.SpaceWeather.SW1D_sqrt.python.mesh1D_adapted import mesh1D_adapted
from astropy.constants import G, k_B, h, M_earth, R_earth, c
import astropy.units as u
import spaceweather as sw


def pdeparams(pde, mesh, parameters):
    """ Set the model input parameters.

    :param pde: contains pde parameters (dict).
    :param mesh: contains mesh grid parameters (dict).
    :param parameters: contains specified input parameters (dict).

    :return: updated pde (dict) , updated mesh (dict).
    """
    # read input csv files
    euv = read_csv(parameters["EUV_input_file_directory"], header=None)
    orbits = read_csv(parameters["orbits_input_file_directory"])
    neutrals = read_csv(parameters["neutrals_input_file_directory"], delimiter=";")

    # set planet information
    period_day = (float(orbits.values[orbits.values[:, 0] == parameters["planet"], 13]) * u.h).to(u.s)
    radius_in = R_earth + parameters["altitude_lower"]
    radius_out = R_earth + parameters["altitude_upper"]
    declination_sun0 = float(orbits.values[orbits.values[:, 0] == parameters["planet"], 19])
    # add the declaration of the sun
    declination_sun = np.arcsin(-np.sin(declination_sun0) * np.cos(
        2 * np.pi * (parameters["day_of_year"] + 9) / 365.24 + np.pi * 0.0167 * 2 * np.pi * (
                    parameters["day_of_year"] - 3) / 365.24))

    # set species information
    species_euv = euv.values[4:, 1]
    i_species = np.where(neutrals.values[:, 0] == parameters["species"])[0]
    i_species_euv = np.where(species_euv == parameters["species"])[0]
    neutrals = neutrals.values[:, 1:]
    gam = parameters["gamma"]
    amu = 1.66e-27 * u.kg  # atomic mass unit
    if parameters["species"] == 'air':
        m = 1
        rho0 = 2
        kappa0 = 1  # todo: how can you define this for air?
        crossSections_d = 1  # todo: how can you define this for air?
    else:
        # mass of neutrals (kg)
        m = (neutrals[i_species, 0][0] * amu)
        # reference density (kg/m^3)
        rho0 = (neutrals[i_species, -1][0] * m) * (1 / u.m ** 3)
        # kappa exponential using empirical models
        expKappa = neutrals[i_species, 3][0]
        # reference thermal conductivity (J/m*K)
        kappa0 = (neutrals[i_species, 2][0] * (parameters["temp_lower"].value ** expKappa)) * u.J / (u.m * u.K * u.s)
        # photo absortion cross section (m^2) # todo: verify with jordi.
        crossSections_d = euv.values[i_species_euv + 4, 5:42] * float(euv.values[i_species_euv + 4, 3]) * u.m ** 2

    lambda_d = 0.5 * (euv.values[0, 5:42] + euv.values[1, 5:42]) * 1e-10
    AFAC = euv.values[3, 5:42]
    F74113_d = euv.values[2, 5:42] * float(euv.values[2, 3]) * 1e4

    # define physical quantities
    # universal gas constant (J/K*kg)
    R = k_B / m
    # gravitational acceleration (m/s**2)
    g = G * M_earth / radius_in ** 2
    # rotation speed of Earth (1/s)
    omega = 2 * np.pi / period_day
    # specific heat capacity (J/K*kg)
    cp = gam * R / (gam - 1)
    # reference scale height (m)
    H0 = (R * parameters["temp_lower"] / g).decompose()

    # define reference quantities
    # reference velocity (m/s)
    v0 = np.sqrt(gam * R * parameters["temp_lower"]).decompose()
    # reference time scale (s)
    t0 = (H0 / v0)
    # reference length scale ratio (dimensionless): mesh lower and upper bounds.
    R0 = (radius_in / H0).decompose()
    R1 = (radius_out / H0).decompose()
    # reference dynamic viscosity (kg /m*s)
    mu0 = (1.3e-4 * (u.kg / (u.K * u.s ** 2)) * (parameters["temp_lower"] / R) ** parameters["exp_mu"]).decompose()

    # rescale mu0, kappa0, rho0
    mu0 = parameters["ref_mu_scale"] * mu0
    kappa0 = parameters["ref_kappa_scale"] * kappa0
    rho0 = parameters["ref_rho_scale"] * rho0

    # define euv heating parameters
    # flux photos corresponding wavelength
    # todo: add units
    lambda_EUV = lambda_d / parameters["lambda0"]
    # photo absorption cross section
    # todo: add units
    crossSections = crossSections_d / H0 ** 2
    # the normalized EUV flux spectrum
    # todo: add units
    F74113 = F74113_d * (H0 ** 2 * t0)


    # read in F10.7 data
    data = sw.sw_daily()
    F10p7 = data.f107_adj[parameters["date"]]
    F10p7_81 = data.f107_81lst_adj[parameters["date"]]

    # dimensionless numbers
    # Grasshoff dimensionless number
    Gr = (g * H0 ** 3 / (mu0 / rho0) ** 2).decompose()
    # Prandtl dimensionless number
    Pr = (mu0 * cp / kappa0).decompose()
    # Froude dimensionless number
    Fr = omega * np.sqrt((H0 / g))
    #  ratio of kinetic to photoionization energy
    Keuv = (gam * k_B * parameters["temp_lower"]) / ((h * c) / parameters["lambda0"])
    #  the amount of particles in a given volume
    M = rho0 * (H0 ** 3) / m

    # set time parameters
    t_step_star = parameters["t_step"] / t0
    n_time_steps = np.ceil(parameters["t_simulation"] * period_day / parameters["t_step"])
    freq_time_steps = np.ceil(parameters["frequency_save"].to(u.s) / parameters["t_step"])

    # set discretization parameters, physical parameters, and solver parameters
    pde['porder'] = parameters["p_order"]  # polynomial degree
    pde['torder'] = parameters["t_order"]  # time-stepping order of accuracy
    pde['nstage'] = parameters["p_order"]  # time-stepping number of stages
    pde['dt'] = t_step_star.value * np.ones([int(n_time_steps.value), 1])  # time step sizes
    pde['visdt'] = pde['dt'][0]  # visualization timestep size
    pde['saveSolFreq'] = freq_time_steps  # solution is saved every 100 time steps
    # steps at which solution are collected
    pde['soltime'] = np.arange(freq_time_steps.value, pde['dt'].shape[0], freq_time_steps)
    pde['timestepOffset'] = parameters["t_restart"].value  # restart parameter

    # store physical parameters
    pde['physicsparam'] = np.array([parameters["gamma"],  # 0
                                    Gr.value,  # 1
                                    Pr.value,  # 2
                                    Fr.value,  # 3
                                    Keuv.value,  # 4
                                    M.value,  # 5
                                    rho0.value,  # 6
                                    parameters["reference_temp_lower"],  # 7
                                    (parameters["temp_upper"] / parameters["temp_lower"]).value,  # 8
                                    R0.value,  # 9
                                    R1.value,  # 10
                                    H0.value,  # 11
                                    parameters["euv_efficiency"],  # 12
                                    parameters["coord"],  # 13
                                    parameters["longitude"].value,  # 14
                                    parameters["latitude"].value,  # 15
                                    declination_sun,  # 16
                                    parameters["tau_a"],  # 17
                                    t0.value,  # 18
                                    F10p7 + parameters["F10p7_uncertainty"],  # 19
                                    F10p7_81 + parameters["F10p7-81_uncertainty"]  # 20
                                    ])

    # store external parameters
    pde['externalparam'] = np.hstack([lambda_EUV.value, crossSections[0, :], AFAC, F74113.value])

    # set solver parameters
    pde['extStab'] = parameters["ext_stab"]
    pde['tau'] = parameters["tau"]  # DG stabilization parameter
    pde['GMRESrestart'] = parameters["GMRES_restart"]  # number of GMRES restarts
    pde['linearsolvertol'] = parameters["linear_solver_tol"]  # GMRES tolerance
    pde['linearsolveriter'] = parameters["linear_solver_iter"]  # number of GMRES iterations
    pde['precMatrixType'] = parameters["pre_cond_matrix_type"]  # preconditioning type
    pde['NLtol'] = 1e-10  # Newton toleranccd dataoue
    pde['NLiter'] = 2  # Newton iterations
    pde['matvectol'] = 1e-7  # todo: define
    pde['RBdim'] = 8  # todo: define

    # set computational mesh
    mesh['p'], mesh['t'] = mesh1D_adapted(r1=R0, r2=R1, nx=parameters["resolution"])
    # expressions for domain boundaries
    mesh['boundaryexpr'] = [lambda p: (p[0, :] < R0 + parameters["boundary_epsilon"]),
                            lambda p: (p[0, :] > R1 - parameters["boundary_epsilon"])]
    mesh['boundarycondition'] = np.array([1, 2])  # Set boundary condition for each boundary
    return pde, mesh
