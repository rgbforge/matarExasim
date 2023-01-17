""" Module to define the model input parameters driven by solar conditions.
Latest update: Jan 11, 2023 [OI]
"""
import numpy as np
from pandas import read_csv
from math import ceil
from Applications.SpaceWeather.SW1D_sqrt.python.mesh1D_adapted import mesh1D_adapted
from Applications.SpaceWeather.SW1D_sqrt.python.MSIS_reference_values import MSIS_reference_values
from astropy.constants import G, k_B, h, M_earth, R_earth, c
import astropy.units as u
import spaceweather as sw
from datetime import date


def pdeparams(pde, mesh, parameters):
    """ Set the model input parameters.

    :param pde: contains pde parameters (dict).
    :param mesh: contains mesh grid parameters (dict).
    :param parameters: contains specified input parameters (dict).

    :return: updated pde (dict) , updated mesh (dict).
    """
    # read in F10.7 data
    data = sw.sw_daily()
    F10p7 = data.f107_adj[parameters["date"]] * (1E-22 * u.W * u.Hz / (u.m ** 2))
    F10p7_81 = data.f107_81lst_adj[parameters["date"]] * (1E-22 * u.W * u.Hz / (u.m ** 2))

    # read input csv files
    euv = read_csv(parameters["EUV_input_file_directory"], header=None)
    orbits = read_csv(parameters["orbits_input_file_directory"])
    neutrals = read_csv(parameters["neutrals_input_file_directory"], delimiter=";")

    # set planet information
    period_day = (float(orbits.values[orbits.values[:, 0] == parameters["planet"], 13]) * u.h).to(u.s)
    radius_in = R_earth + parameters["altitude_lower"]
    radius_out = R_earth + parameters["altitude_upper"]
    declination_sun0 = float(orbits.values[orbits.values[:, 0] == parameters["planet"], 19])
    # get the day of the year
    day_of_year = date(int(parameters["date"][:4]),
                       int(parameters["date"][5:7]),
                       int(parameters["date"][8:10])).timetuple().tm_yday
    # add the declaration of the sub, Jordi, why do you not have this line implemented in the MATLAB version?
    # also, why do you repeat the calculation in pdemodelMSIS?
    # declination_sun = np.arcsin(-np.sin(declination_sun0) * np.cos(
    #     2 * np.pi * (day_of_year + 9) / 365.24 + np.pi * 0.0167 * 2 * np.pi * (day_of_year - 3) / 365.24))
    # Answer: the computation is performed in pdemodel because the declination angle changes with time
    # For this reason, the value that we give as parameter is this declination0 (maximum declination, the one at solstice)
    # then the current declination is computed at every time-step. Otherwise, it would be constant.

    # set species information
    i_species = np.zeros(len(parameters["chemical_species"]))
    i_species_euv = np.zeros(len(parameters["chemical_species"]))
    for ii in range(len(parameters["chemical_species"])):
        i_species[ii] = np.where(neutrals.values[:, 0] == parameters["chemical_species"][ii])[0]
        #  todo: Jordi, can you check this line? should it be 2 or 4?
        #  todo: the line in MATLAB:
        #  todo: iSpeciesEUV(isp) = find(strcmp(table2array(EUV(:, 2)), species(isp)));
        i_species_euv[ii] = np.where(euv.values[4:, 1] == parameters["chemical_species"][ii])[0]

    amu = 1.66e-27 * u.kg  # atomic mass unit
    # mass of neutrals (kg)
    mass = (neutrals[i_species, 1][0] * amu)
    # reference thermal conductivity (J/m*K)
    ckappa0 = neutrals[i_species, 3][0]
    #  todo: in MATLAB version:
    #  todo: expKappa = table2array(neutrals(iSpecies,5));
    #  todo: expKappa = 0.75;
    #  todo: Jordi, what should I use here?
    # initially in Armstrongs
    lambda_d = 0.5 * (euv.values[0, 5:42] + euv.values[1, 5:42]) * 1e-10
    AFAC = euv.values[3, 5:42]
    # todo: EUV.values starts at 4 above, should we do the same here?
    F74113_d = euv.values[2, 5:42] * float(euv.values[2, 3]) * 1e4
    # photo absortion cross section (m^2) # todo: verify with jordi.
    crossSections_d = euv.values[i_species_euv + 4, 5:42] * float(euv.values[i_species_euv + 4, 3]) * u.m ** 2

    # MSIS reference values
    # todo return: chi are the mass fractions (rho_i/rho) over altitude
    # todo return: cchi are the coefficients ai of the fit: chi ~ a1*exp(a2*(h-H0)) + a3*exp(a4*(h-H0))
    # for each of the species except one (atomic O) which is computed as 1-sum{chi}
    rho0, T0, chi, cchi = MSIS_reference_values(parameters=parameters, mass=mass)

    # define physical quantities
    m = mass[0]
    # universal gas constant (J/K*kg)
    R = k_B / m
    # gravitational acceleration (m/s**2)
    g = G * M_earth / radius_in ** 2
    # rotation speed of Earth (1/s)
    omega = 2 * np.pi / period_day
    # specific heat capacity (J/K*kg)
    cp = parameters["gamma"] * R / (parameters["gamma"] - 1)
    # reference scale height (m)
    H0 = (R * T0 / g).decompose()

    # define reference quantities
    # reference velocity (m/s)
    v0 = np.sqrt(parameters["gamma"] * R * T0).decompose()
    # reference time scale (s)
    t0 = (H0 / v0)
    # reference length scale ratio (dimensionless): mesh lower and upper bounds.
    R0 = (radius_in / H0).decompose()
    R1 = (radius_out / H0).decompose()

    # reference viscosity and conductivity
    # reference dynamic viscosity (kg /m*s)
    # todo: matlab code:
    #  cmu0 = 2 * 1.3e-4;
    #  expMu = 0.5;
    #  kappa0 = chi(1,:)*ckappa0 * T0 ^ expKappa;
    #  % kappa0 = ckappa0(2) * T0 ^ expKappa;      # what is this?
    #  ckappai = ckappa0 / (chi(1,:) * ckappa0);  # do we need this?
    #  kappa0 = 0.4 * kappa0;
    #  alpha0 = kappa0 / (rho0 * cp);
    #  mu0 = cmu0 * (T0 / R) ^ expMu;
    #  nu0 = mu0 / rho0;
    # todo: Jordi, should I multiply the line below by 2?
    cmu0 = 1.3e-4 * (u.kg / (u.K * u.s ** 2))
    mu0 = (cmu0 * (T0 / R) ** parameters["exp_mu"]).decompose()
    # todo: Jordi, what is the size of kappa0? should I scale it with 0.4
    kappa0 = chi[0, :] * ckappa0 * (T0 ** parameters["exp_kappa"])

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
    # todo add from MATLAB: mass = mass/m; Jordi, do we need this?

    # dimensionless numbers
    # Grasshoff dimensionless number
    Gr = (g * H0 ** 3 / (mu0 / rho0) ** 2).decompose()
    # Prandtl dimensionless number
    Pr = (mu0 * cp / kappa0).decompose()
    # Froude dimensionless number
    Fr = omega * np.sqrt((H0 / g))
    #  ratio of kinetic to photoionization energy
    Keuv = (parameters["gamma"] * k_B * T0) / ((h * c) / parameters["lambda0"])
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
                                    parameters["euv_efficiency"],  # 6
                                    declination_sun0,  # 7
                                    F10p7.value + parameters["F10p7_uncertainty"].value,  # 8
                                    F10p7_81.value + parameters["F10p7-81_uncertainty"].value,  # 9
                                    day_of_year,  # 10
                                    parameters["exp_mu"],  # 11
                                    parameters["exp_kappa"],  # 12
                                    parameters["nu_eddy"],  # 13
                                    parameters["alpha_eddy"],  # 14
                                    R0.value,  # 15
                                    R1.value,  # 16
                                    H0.value,  # 17
                                    T0.value,  # 18
                                    rho0.value,  # 19
                                    t0.value,  # 20
                                    parameters["tau_a"],  # 21
                                    parameters["latitude"].value,  # 22
                                    parameters["longitude"].value,  # 23
                                    parameters["coord"],  # 24
                                    parameters["date"][:4]  # 25 # todo why do we need this?
                                    ])

    # store external parameters
    # todo: Jordi, I am not sure how to convert this from your matlab code. ordering is different.
    # todo MATLAB: pde.externalparam = [lambda,AFAC,F74113,reshape(crossSections',[37*nspecies,1])',
    #  reshape(cchi',[4*(nspecies-1),1])',mass',ckappai'];
    pde['externalparam'] = np.hstack([lambda_EUV.value, AFAC, F74113.value, crossSections[0, :], ])

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

    # todo: initial condition (Jordi, we should discuss how to implement this).
    #  [s1, s2, s3] = size(mesh.dgnodes);
    #  ndg = s1 * s3;
    #  nc = 6;
    #  xdg = reshape(mesh.dgnodes, [s2, ndg])';
    #  paramsMSIS = [R0, latitude, longitude, year, doy, sec, F10p7, F10p7a, hbot, H, T0, rho0, Fr, m];
    #  u0 = MSIS_initialCondition1D_pressure(xdg, paramsMSIS, indicesMSIS, mass);
    #  mesh.udg = pagetranspose(reshape(u0',[nc,s1,s3]));
    return pde, mesh
