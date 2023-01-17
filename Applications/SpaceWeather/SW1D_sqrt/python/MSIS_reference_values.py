""" Module to call MSIS from Python.
Latest update: Jan 17th, 2023 [OI]
"""
from pymsis import msis
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize as opt


def get_MSIS_species(MSIS_output, parameters):
    """A function to get the requested MSIS species results.

    :param MSIS_output: MSIS output for the requested species.
    :param parameters: dictionary defined in pdeapp.py with all the model parameters (solver and physical parameters).
    :return: MSIS tensor results dimensions: (nspecies, n_longitude, n_latitude, n_altitude)
    """
    # initialize the dataset we need.
    MSIS = np.zeros((len(parameters["chemical_species"]),
                     parameters["n_longitude_MSIS"],
                     parameters["n_latitude_MSIS"],
                     parameters["n_radial_MSIS"])) * 1 / u.m ** 3

    # loop over chemical species
    for ii in range(len(parameters["chemical_species"])):
        if parameters["chemical_species"][ii] == "N2":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 1] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "O2":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 2] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "O":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 3] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "He":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 4] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "H":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 5] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "Ar":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 6] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "N":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 7] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "Anomalous oxygen":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 8] * 1 / u.m ** 3
        elif parameters["chemical_species"][ii] == "NO":
            MSIS[ii, :, :, :] = MSIS_output[0, :, :, :, 9] * 1 / u.m ** 3
    return MSIS


def data_model_error(coefficients, altitude_low_boundary, altitude_mesh, data):
    """ A function to evaluate the model and data misfit. We later minimize this function to find the optimal
    coefficients.

    :param coefficients: [a1, a2, a3, a4]
    :param altitude_low_boundary:
    :param altitude_mesh: mesh in altitude.
    :param data: mass fraction of a certain specie
    :return:
    """
    # # the parameters are stored as a vector of values, so unpack the vector
    a1, a2, a3, a4 = coefficients
    # model evaluation using the input coefficients.
    model_eval = a1 * np.exp(a2 * (altitude_mesh - altitude_low_boundary)) + \
                 a3 * np.exp(a4 * (altitude_mesh - altitude_low_boundary))
    # return the model misfit in the 2-norm
    return np.linalg.norm(model_eval - data)


def coefficient_fit(parameters, altitude_mesh, data):
    """ A function to fit data ~ a1 * exp(a2 * (h - H0)) + a3 * exp(a4 * (h - H0))

        :param altitude_mesh: MSIS results altitude mesh (in km).
        :param parameters: dictionary defined in pdeapp.py with all the model parameters.
        :param data: mass fraction of a particular species.
        :return: [a1, a2, a3, a4] **optimal in L2 sense**
    """
    # minimize the loss function using a conjugate gradient optimizer.
    minimization_results = opt.minimize(fun=lambda *args: data_model_error(*args),
                                        x0=[0, 0, 0, 0],
                                        method="CG",
                                        args=(parameters["altitude_lower"].to(u.m).value,
                                              altitude_mesh.to(u.m).value,
                                              data))
    return minimization_results["x"]


def MSIS_reference_values(parameters, mass):
    """A function to get MSIS reference values,
    i.e. the reference density (rho0), temperature (T0), mass fraction (chi), and coefficients of the fit (cchi).

    :param parameters: dictionary defined in pdeapp.py with all the model parameters (solver and physical parameters).
    :param mass: molecular mass of each species (we need it to compute mass density from species number densities).

    :return: (1) rho0 : type: float, units: [kg/m^3]
                        reference density at the lower altitude boundary (100km) todo: verify with Jordi.
             (2) T0 : type: float, units: [K]
                        reference temperature at the lower altitude boundary (100km).
             (3) mass_fraction_species : type: [n_altitude_MSIS, n_species], units: [dimensionless]
                        mass fractions (rho_i/rho) over altitude
             (4) optimal_coefficient : [n_species -1, 4] (skip oxygen here), units: [dimensionless]
                        coefficients of the fit: mass fraction ~ a1*exp(a2*(h-H0)) + a3*exp(a4*(h-H0))
                        returned [a1, a2, a3, a4]

    """
    # we do not need the mesh here, we only need the position of the lower and upper boundary
    # we want to obtain certain quantities at the lower boundary + a distribution in space of the partial densities
    # to do that we can use any radial distribution of points we want, not necessarily linked to the mesh
    # Actually we want more points, so don't use resolution, but another parameter. I set it to have 101 points.
    altitude_mesh = np.linspace(parameters["altitude_lower"].to(u.km).value,
                                parameters["altitude_upper"].to(u.km).value,
                                parameters["n_radial_MSIS"])
    longitude_mesh = np.linspace(-180, 175, parameters["n_longitude_MSIS"])
    latitude_mesh = np.linspace(-85, 85, parameters["n_latitude_MSIS"])

    # get data (F10.7, F10.7_81, Ap) needed to run MSIS.
    f10p7_msis, f10p7a_msis, ap_msis = msis.get_f107_ap(dates=parameters["date"])

    # run MSIS, output is of dimensions: (ndates, nlons, nlats, nalts, 11)
    # 11 stands for each species in the following order:
    # [Total mass density (kg / m3),  N2 density (m-3), O2 density (m-3), O density (m-3),
    #  He density (m-3), H density (m-3), Ar density (m-3), N density (m-3),
    #  Anomalous oxygen density (m-3), NO density (m-3), Temperature(K)]
    MSIS_output = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                           lons=longitude_mesh,  # (list of floats) – Longitudes of interest
                           lats=latitude_mesh,  # (list of floats) – Latitudes of interest
                           alts=altitude_mesh,  # (list of floats) – Altitudes of interest
                           # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                           f107s=f10p7_msis + (parameters["F10p7_uncertainty"] * 1E22).value,
                           # F10.7 running 81-day average centered on the given date(s)
                           f107as=f10p7a_msis + (parameters["F10p7-81_uncertainty"] * 1E22).value,
                           # Daily Ap
                           aps=[ap_msis])

    # temperate at the bottom.
    T0 = np.mean(MSIS_output[0, :, :, 0, -1]) * u.K

    # initialize the dataset we need. dimensions: (nspecies, nlons, nlats, nalts)
    MSIS = get_MSIS_species(MSIS_output=MSIS_output, parameters=parameters)

    # loop over all altitudes and compute the mean density
    number_density_mean_species = np.zeros((len(altitude_mesh), len(parameters["chemical_species"]))) * (1 / u.m**3)
    for ii in range(len(altitude_mesh)):
        for jj in range(len(parameters["chemical_species"])):
            number_density_mean_species[ii, jj] = np.mean(MSIS[jj, :, :, ii].value) * (1 / u.m**3)

    # density mean total, size [n_altitude]
    density_mean_total = np.dot(number_density_mean_species, mass)
    # mass fraction of the species, size [n_altitude, n_species], in matlab this is called "chi"
    mass_fraction_species = np.zeros((parameters["n_radial_MSIS"], len(parameters["chemical_species"])))
    species_mass = number_density_mean_species * mass
    for ii in range(len(parameters["chemical_species"])):
        mass_fraction_species[:, ii] = (species_mass[:, ii] / density_mean_total).value

    # optimal coefficients to fit to exponential function, in MATLAB this is called "cchi".
    optimal_coefficient = np.zeros((len(parameters["chemical_species"]) - 1, 4))
    # skip oxygen since it does not fit well to the model.
    for ii in range(len(parameters["chemical_species"]) - 1):
        optimal_coefficient[ii, :] = coefficient_fit(parameters=parameters,
                                                     data=mass_fraction_species[:, ii + 1],  # skip oxygen
                                                     altitude_mesh=altitude_mesh*u.km)
    return density_mean_total[0], T0, mass_fraction_species, optimal_coefficient
