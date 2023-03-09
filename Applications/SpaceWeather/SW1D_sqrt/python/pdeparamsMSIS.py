""" Module to define the model input parameters driven by solar conditions.

Latest update: March 7th, 2023 [OI]
"""
import numpy as np
from pandas import read_csv
from mesh_1D_adapted import mesh1D_adapted
from MSIS_reference_values import MSIS_reference_values
from initial_condition_1D_pressure import MSIS_initial_condition_1D_pressure
from astropy.constants import G, k_B, h, M_earth, R_earth, c
import astropy.units as u
import spaceweather as sw
from pdemodelMSIS import EUVsource1D
import matplotlib.pyplot as plt
import os
import sys
from datetime import date

# import internal modules
# Add Exasim package to python search path
cdir = os.getcwd()
ii = cdir.find("Exasim")
sys.path.append(cdir[:(ii + 6)] + "/src/Python/Preprocessing/")
from createdgnodes import createdgnodes


def pdeparams(pde, mesh, parameters):
    """ A function to set the model input parameters, mesh and initial conditions from MSIS.

    :param pde: contains pde parameters (dict).
    :param mesh: contains mesh grid parameters (dict).
    :param parameters: contains specified input parameters (dict) defined in pdeapp.py.

    :return: updated pde (dict) & updated mesh (dict).
    """
    # read in F10.7 data
    data = sw.sw_daily(update=True)
    F10p7 = data.f107_adj[parameters["date"]]
    F10p7_81 = data.f107_81lst_adj[parameters["date"]]

    # read input csv files
    euv = read_csv(parameters["EUV_input_file_directory"], header=None)
    orbits = read_csv(parameters["orbits_input_file_directory"])
    neutrals = read_csv(parameters["neutrals_input_file_directory"], delimiter=";")

    # set planet information
    period_day = (float(orbits.values[orbits.values[:, 0] == parameters["planet"], 13]) * u.h).to(u.s)
    radius_in = R_earth + parameters["altitude_lower"]
    radius_out = R_earth + parameters["altitude_upper"]
    # maximum declination at summer/winter solstice.
    declination_sun0 = float(orbits.values[orbits.values[:, 0] == parameters["planet"], 19])
    # get the day of the year
    day_of_year = date(int(parameters["date"][:4]),
                       int(parameters["date"][5:7]),
                       int(parameters["date"][8:10])).timetuple().tm_yday

    # set species information
    i_species = np.zeros(len(parameters["chemical_species"]), dtype=int)
    i_species_euv = np.zeros(len(parameters["chemical_species"]), dtype=int)
    for ii in range(len(parameters["chemical_species"])):
        i_species[ii] = np.where(neutrals.values[:, 0] == parameters["chemical_species"][ii])[0]
        i_species_euv[ii] = np.where(euv.values[4:, 1] == parameters["chemical_species"][ii])[0]

    # atomic mass unit
    amu = 1.66e-27 * u.kg
    # mass of neutrals (kg)
    mass = np.array(neutrals.values[i_species, 1], dtype=float) * amu
    # reference thermal conductivity (J/m*K)
    ckappa0 = neutrals.values[i_species, 3]
    # initially in Armstrongs
    lambda_d = 0.5 * (euv.values[0, 5:42] + euv.values[1, 5:42]) * 1e-10
    AFAC = euv.values[3, 5:42]
    F74113_d = euv.values[2, 5:42] * float(euv.values[2, 3]) * 1e4
    # photo absortion cross section (m^2)
    crossSections_d = (euv.values[i_species_euv + 4, 5:42].T * euv.values[i_species_euv + 4, 3] * u.m ** 2).T

    # get MSIS reference values
    # chi are the mass fractions (rho_{species}/rho) over MSIS uniform altitude mesh
    # c_chi are the coefficients ai of the fit: chi ~ a1*exp(a2*(h-H0)) + a3*exp(a4*(h-H0))
    # for each of the species except one (atomic O) which is computed as 1-sum{chi}
    rho0, T0, chi, c_chi = MSIS_reference_values(parameters=parameters, mass=mass)

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
    c_mu0 = 1.3e-4 * (u.kg / (u.K * u.s ** 2))
    mu0 = (c_mu0 * ((T0 / R) ** parameters["exp_mu"])).decompose()
    kappa0 = np.dot(chi[0, :], ckappa0) * (T0.value ** parameters["exp_kappa"]) * (u.joule / (u.K * u.m * u.s))
    c_kappa_i = ckappa0 / np.dot(chi[0, :], ckappa0)

    # rescale mu0, kappa0, rho0
    mu0 = parameters["ref_mu_scale"] * mu0
    kappa0 = parameters["ref_kappa_scale"] * kappa0

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
    # non-dimensional mass (scaled with mass of oxygen)
    mass = mass / m

    # dimensionless numbers
    # Grasshoff dimensionless number
    Gr = (g * H0 ** 3 / (mu0 / rho0) ** 2).decompose()
    # Prandtl dimensionless number
    Pr = (mu0 * cp / kappa0).decompose()
    # Froude dimensionless number
    Fr = omega * np.sqrt((H0 / g))
    # ratio of kinetic to photoionization energy
    Keuv = (parameters["gamma"] * k_B * T0) / ((h * c) / parameters["lambda0"])
    # the amount of particles in a given volume
    M = rho0 * (H0 ** 3) / m

    # set time parameters
    t_step_star = parameters["t_step"] / t0
    n_time_steps = np.ceil(parameters["t_simulation"] * period_day / parameters["t_step"])
    freq_time_steps = np.ceil(parameters["frequency_save"].to(u.s) / parameters["t_step"])

    # set discretization parameters, physical parameters, and solver parameters
    pde['porder'] = parameters["p_order"]  # polynomial degree
    pde['torder'] = parameters["t_order"]  # time-stepping order of accuracy
    pde['nstage'] = parameters["n_stage"]  # time-stepping number of stages
    pde['dt'] = t_step_star.value * np.ones([int(n_time_steps.value), 1])  # time step sizes
    pde['visdt'] = pde['dt'][0]  # visualization timestep size
    pde['saveSolFreq'] = freq_time_steps  # solution is saved every parameters["frequency_save"]
    # steps at which solution are collected
    pde['soltime'] = np.arange(freq_time_steps.value, pde['dt'].shape[0], freq_time_steps)
    pde['timestepOffset'] = parameters["t_restart"]  # restart parameter

    # store physical parameters
    pde['physicsparam'] = np.array([parameters["gamma"],  # 0
                                    Gr.value,  # 1
                                    Pr.value,  # 2
                                    Fr.value,  # 3
                                    Keuv.value,  # 4
                                    M.value,  # 5
                                    parameters["euv_efficiency"],  # 6
                                    declination_sun0,  # 7
                                    F10p7 + parameters["F10p7_uncertainty"],  # 8
                                    F10p7_81 + parameters["F10p7-81_uncertainty"],  # 9
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
                                    ])

    # store external parameters
    pde['externalparam'] = np.hstack([lambda_EUV.value,  # 0
                                      AFAC,  # 1
                                      F74113.value,  # 2
                                      crossSections.flatten(order="C"),  # 3
                                      c_chi.flatten(order="C"),  # 4
                                      mass,  # 5
                                      c_kappa_i  # 6
                                      ])

    # set solver parameters
    pde['extStab'] = parameters["ext_stab"]
    pde['tau'] = parameters["tau"]  # DG stabilization parameter
    pde['GMRESrestart'] = parameters["GMRES_restart"]  # number of GMRES restarts
    pde['linearsolvertol'] = parameters["linear_solver_tol"]  # GMRES tolerance
    pde['linearsolveriter'] = parameters["linear_solver_iter"]  # number of GMRES iterations
    pde['precMatrixType'] = parameters["pre_cond_matrix_type"]  # preconditioning type
    pde['NLtol'] = parameters["newton_tol"]  # Newton tolerance
    pde['NLiter'] = parameters["newton_iter"]  # Newton iterations
    pde['matvectol'] = parameters["mat_vec_tol"]  # finite difference approach for Jacobian approximation
    pde['RBdim'] = parameters["rb_dim"]  # number of dimensions of reduced basis used to compute the conditioner and
    # initialization used for each time step.

    # set computational mesh
    mesh['p'], mesh['t'] = mesh1D_adapted(r1=R0, r2=R1, nx=parameters["resolution"])
    # expressions for domain boundaries
    mesh['boundaryexpr'] = [lambda p: (p[0, :] < R0 + parameters["boundary_epsilon"]),
                            lambda p: (p[0, :] > R1 - parameters["boundary_epsilon"])]
    mesh['boundarycondition'] = np.array([1, 2])  # Set boundary condition for each boundary
    # get the mesh nodes.
    # mesh dgnodes has dimensions:
    # [n_points_per_element x n_dimensions x n_elements]
    # (1) n_points_per_element: the number of nodal points (in 1D: p+1),
    # (2) n_dimensions: 1 because we are in 1D
    # (3) n_elements is the ones that you have set.
    mesh["dgnodes"] = createdgnodes(mesh["p"],
                                    mesh["t"],
                                    mesh["f"],
                                    mesh["curvedboundary"],
                                    mesh["curvedboundaryexpr"],
                                    pde["porder"])

    # todo: initial condition (Jordi, we should discuss how to implement this).
    #  [s1, s2, s3] = size(mesh.dgnodes);
    #  ndg = s1 * s3;
    #  nc = 6;
    #  xdg = reshape(mesh.dgnodes, [s2, ndg])';
    #  paramsMSIS = [R0, latitude, longitude, year, doy, sec, F10p7, F10p7a, hbot, H, T0, rho0, Fr, m];
    #  u0 = MSIS_initialCondition1D_pressure(xdg, paramsMSIS, indicesMSIS, mass);
    n_points_per_element, n_dimensions, n_elements = np.shape(mesh["dgnodes"])
    # get grid points in km
    altitude_mesh_grid = ((mesh["dgnodes"].flatten("F") - R0) * H0 + parameters["altitude_lower"]).to(u.km)
    u0 = MSIS_initial_condition_1D_pressure(x_dg=mesh["dgnodes"].flatten("F"),
                                            altitude_mesh_grid=altitude_mesh_grid,
                                            parameters=parameters,
                                            mass=mass,
                                            T0=T0,
                                            m=m,
                                            rho0=rho0,
                                            H0=H0,
                                            Fr=Fr,
                                            R0=R0)
    # todo:
    #  mesh.udg = pagetranspose(reshape(u0',[nc,s1,s3]));
    #   Jordi, can you verify this is the correct operation?
    #   can we avoid the transposing and directly provide the right ordering from u0?
    #   [48 x 6] ---> transpose ---> [6 x 3 x 16] -----> transpose -----> [3 x 6 x 16] result.
    #   need to test this.
    udg = np.zeros((n_points_per_element, 6, n_elements))
    for elem in range(n_elements):
        for node in range(n_points_per_element):
            udg[node, :, elem] = u0[n_points_per_element * elem + node, :]

    mesh['udg'] = udg

    # # todo: testing EUV heating!!! please delete me after debugging :)
    # t = 0
    # # sol = np.zeros((n_points_per_element, n_elements, len(pde["soltime"])))
    # # # loop over all timesteps.
    # # for tt in range(len(pde["soltime"])):
    # #     index_t = pde["soltime"][tt]
    # #     t += pde["saveSolFreq"].value*pde["dt"][int(index_t)][0]
    # #     for i_elem in range(n_elements):
    # #         for i_point in range(n_points_per_element):
    # #             euv_output = EUVsource1D(u=mesh["udg"][i_point, :, i_elem], x=mesh["dgnodes"][i_point, :, i_elem],
    # #                                      t=t, mu=pde["physicsparam"], eta=pde['externalparam'].value)
    # #
    # #             sol[i_point, i_elem, tt] = euv_output
    # #
    # #
    # #
    # # np.save("sol.npy", sol)
    # sol = np.zeros((n_points_per_element, n_elements))
    # t = 60*60*12/t0.value
    # for i_elem in range(n_elements):
    #     if i_elem == 10:
    #         print("stop here ")
    #     for i_point in range(n_points_per_element):
    #         euv_output = EUVsource1D(u=mesh["udg"][i_point, :, i_elem], x=mesh["dgnodes"][i_point, :, i_elem],
    #                                  t=t, mu=pde["physicsparam"], eta=pde['externalparam'].value)
    #
    #         sol[i_point, i_elem] = euv_output
    #
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # number_of_plots = 1
    # colormap = plt.cm.nipy_spectral
    # colors = colormap(np.linspace(0, 1, number_of_plots))
    # ax.set_prop_cycle('color', colors)
    # t = 0
    # skip = 0
    # ax.plot(altitude_mesh_grid, sol[:, :].flatten("F"), label="t = " + str(round(t, 2)))
    # ax.legend()
    # ax.set_xlabel("Altitude [km]")
    # ax.set_ylabel("source term output")
    # plt.show()

    return pde, mesh
