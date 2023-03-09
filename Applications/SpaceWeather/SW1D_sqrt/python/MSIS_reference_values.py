""" Module to call MSIS from Python.
Latest update: March 2nd, 2023 [OI]
"""
from pymsis import msis
import numpy as np
import astropy.units as u
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def get_MSIS_species(MSIS_output, parameters):
    """A function to get the requested MSIS species results.

    :param MSIS_output: MSIS output for the requested species.
    :param parameters: dictionary defined in pdeapp.py with all the model parameters (solver and physical parameters).
    :return: MSIS tensor results dimensions: (n_species, n_longitude, n_latitude, n_altitude)
    """
    # initialize the dataset we need.
    n_date, n_longitude, n_latitude, n_altitude, n_species = np.shape(MSIS_output)

    # initialize the returned tensor.
    MSIS = np.zeros((len(parameters["chemical_species"]),
                     n_longitude,
                     n_latitude,
                     n_altitude)) * 1 / u.m ** 3

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


def exp_model(altitude_mesh, a1, a2, a3, a4, altitude_low_boundary):
    """ A function to evaluate the model (exponential form).
    We later minimize this function to find the optimal coefficients.

    :param a1, a2, a3, a4 coefficients
    :param altitude_low_boundary: altitude lower boundary (in km)
    :param altitude_mesh: MSIS results altitude mesh (in km).
    :return: error in L2-norm.
    """
    # model evaluation using the input coefficients.
    return a1 * np.exp(a2 * (altitude_mesh - altitude_low_boundary)) + \
           a3 * np.exp(a4 * (altitude_mesh - altitude_low_boundary))


def minimize_func(theta, altitude_low_boundary, altitude_mesh, data, weights, flag):
    """ loss function for nonlinear least squares minimization.

    :param theta: list of parameters [a1, a2, a3, a4]
    :param altitude_low_boundary: altitude lower boundary (in km)
    :param altitude_mesh: MSIS results altitude mesh (in km).
    :param data: partial density for a particular chemical specie.
    :param weights: weighted least squares (especially for N2 which has a more challenging shape to fit to).
    :return: loss (wighted square error).
    """
    a1, a2, a3, a4 = theta
    model_eval = exp_model(altitude_mesh=altitude_mesh, a1=a1, a2=a2, a3=a3, a4=a4,
                           altitude_low_boundary=altitude_low_boundary)
    if flag:
        if np.min(model_eval) < 0:
            return 1E3 * (model_eval - data) * weights
        else:
            return (model_eval - data) * weights
    else:
        return (model_eval - data) * weights


def coefficient_fit(altitude_lower, altitude_mesh, data, species):
    """ A function to fit:

            data ~ a1 * exp(a2 * (h - H0)) + a3 * exp(a4 * (h - H0))

            h - altitude mesh (in m)
            H0 - lower altitude boundary (in m)
            data - mass fraction of a particular species (dimensionless quantity)

    Use non-linear least squares to fit a model to data.

    :param altitude_mesh: MSIS results altitude mesh (later converted to meters).
    :param altitude_lower: altitude lower boundary (later converted to meters).
    :param data: mass fraction of a particular species (dimensionless quantity).
    :param species: str. type of species will change the initialization.
    :return: optimal coefficients [a1, a2, a3, a4] that minimize the sum of squares of the error between the
             nonlinear model and the MSIS data.
    """
    if species == "N2":
        weights = np.arange(len(altitude_mesh))[::-1]
        # weights = np.ones(len(altitude_mesh))
        p0 = np.array([355, -1.33e-5, -355, -1.33e-5])
        flag = False

    elif species == "O2":
        weights = np.ones(len(altitude_mesh))
        p0 = np.array([9.6e-2, -1e-5, 0.1, -5e-5])
        flag = False

    elif species == "He":
        weights = np.ones(len(altitude_mesh))
        p0 = np.array([5e-5, 1e-5, 3e-4, 1e-5])
        flag = True

    else:
        weights = np.ones(len(altitude_mesh))
        p0 = np.array([0, 0, 0, 0])

    minimization_results = leastsq(func=minimize_func,
                                   x0=p0,
                                   full_output=True,
                                   args=(altitude_lower.to(u.m).value, altitude_mesh.to(u.m).value, data, weights, flag),
                                   ftol=1e-11,
                                   maxfev=int(1e9))
    return minimization_results[0]


def MSIS_reference_values(parameters, mass):
    """A function to get MSIS reference values,
    i.e. the reference density (rho0), temperature (T0), mass fraction (chi), and coefficients of the fit (c_chi).

    :param parameters: dictionary defined in pdeapp.py with all the model parameters (solver and physical parameters).
    :param mass: molecular mass of each species (we need it to compute mass density from species number densities).

    :return: (1) rho0 : type: float, units: [kg/m^3]
                        reference density at the lower altitude boundary (e.g. @ 100km)
             (2) T0 : type: float, units: [K]
                        reference temperature at the lower altitude boundary (e.g. @ 100km).
             (3) chi : type: array size [n_altitude_MSIS, n_species], units: [dimensionless]
                        mass fractions (rho_i/rho) over altitude
             (4) c_chi : type: array size [n_species -1, 4] (skip oxygen here), units: [dimensionless]
                        coefficients of the fit: mass fraction ~ a1*exp(a2*(h-H0)) + a3*exp(a4*(h-H0))
                        returned [a1, a2, a3, a4]

    """
    # define altitude uniform mesh.
    altitude_mesh = np.linspace(parameters["altitude_lower"].to(u.km).value,
                                parameters["altitude_upper"].to(u.km).value,
                                parameters["n_radial_MSIS"])
    # define longitude uniform mesh.
    longitude_mesh = np.linspace(-180, 175, parameters["n_longitude_MSIS"])
    # define latitude uniform mesh.
    latitude_mesh = np.linspace(-85, 85, parameters["n_latitude_MSIS"])

    # get data (F10.7, F10.7_81, Ap) needed to run MSIS.
    f10p7_msis, f10p7a_msis, ap_msis = msis.get_f107_ap(dates=parameters["date"])
    # check the read indices are accurate.
    if not np.isfinite(f10p7_msis):
        raise ValueError("F10.7 for this time-period is Nan. ")
    if not np.isfinite(f10p7a_msis):
        raise ValueError("81-day average F10.7 for this time-period is Nan. ")
    if not np.isfinite(ap_msis).any():
        raise ValueError("Ap index for this time-period is Nan. ")

    # run MSIS, output is a tensor of dimensions: (n_dates, n_longitude, n_latitude, n_altitude, 11)
    # 11 stands for each species in the following order:
    #  (1) Total mass density (kg / m3),  (2) N2 density (m-3), (3) O2 density (m-3), (4) O density (m-3),
    #  (5) He density (m-3), (6) H density (m-3), (7) Ar density (m-3), (8) N density (m-3),
    #  (9) Anomalous oxygen density (m-3), (10) NO density (m-3), (11) Temperature(K)
    MSIS_output = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                           lons=longitude_mesh,  # (list of floats) – Longitudes of interest
                           lats=latitude_mesh,  # (list of floats) – Latitudes of interest
                           alts=altitude_mesh,  # (list of floats) – Altitudes of interest
                           # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                           f107s=f10p7_msis,
                           # F10.7 running 81-day average centered on the given date(s)
                           f107as=f10p7a_msis,
                           # Daily Ap
                           aps=[ap_msis])

    # temperate at the bottom.
    T0 = np.mean(MSIS_output[0, :, :, 0, -1]) * u.K

    # initialize the dataset we need. dimensions: (nspecies, nlons, nlats, nalts)
    MSIS = get_MSIS_species(MSIS_output=MSIS_output, parameters=parameters)

    # loop over all altitudes and compute the mean density
    number_density_mean_species = np.zeros((len(altitude_mesh), len(parameters["chemical_species"]))) * (1 / u.m ** 3)
    for ii in range(len(altitude_mesh)):
        for jj in range(len(parameters["chemical_species"])):
            number_density_mean_species[ii, jj] = np.mean(MSIS[jj, :, :, ii].value) * (1 / u.m ** 3)

    # density mean total, size [n_altitude]
    density_mean_total = np.dot(number_density_mean_species, mass)
    # mass fraction of the species, size [n_altitude, n_species], in matlab this variable is called "chi".
    chi = np.zeros((parameters["n_radial_MSIS"], len(parameters["chemical_species"])))
    species_mass = number_density_mean_species * mass
    for ii in range(len(parameters["chemical_species"])):
        chi[:, ii] = (species_mass[:, ii] / density_mean_total).value  # dimensionless

    # optimal coefficients to fit to exponential function, in MATLAB this variable is called "cchi".
    c_chi = np.zeros((len(parameters["chemical_species"]) - 1, 4))

    # skip oxygen since it does not fit well to the exponential sum model (measured in meters).
    # also double check that the atomic oxygen partial density is non-negative!!
    atomic_oxygen_model = np.ones(len(altitude_mesh))
    atomic_oxygen_data = np.ones(len(altitude_mesh))
    for ii in range(len(parameters["chemical_species"]) - 1):
        c_chi[ii, :] = coefficient_fit(altitude_lower=parameters["altitude_lower"].to(u.km),  # in km
                                       data=chi[:, ii + 1],  # skip oxygen
                                       altitude_mesh=altitude_mesh * u.km,
                                       species=parameters["chemical_species"][ii + 1])  # in km

        model_results = exp_model(altitude_mesh=(altitude_mesh * u.km).to(u.m).value,
                                  a1=c_chi[ii, 0],
                                  a2=c_chi[ii, 1],
                                  a3=c_chi[ii, 2],
                                  a4=c_chi[ii, 3],
                                  altitude_low_boundary=parameters["altitude_lower"].to(u.m).value)
        if np.min(model_results) < 0:
            raise ValueError(
                "non-physical initial condition (negative density for " + str(parameters["chemical_species"][ii+1]) +
                "). This is most likely due to the exponential model parameter fit. "
                "We recommend changing the initialization.")

        atomic_oxygen_model += - model_results
        atomic_oxygen_data += - chi[:, ii+1]

    if np.min(atomic_oxygen_model) < 0:
        raise ValueError("non-physical initial condition (negative for atomic oxygen). This is most likely due "
                         "to the exponential model parameter fit. "
                         "We recommend changing the initialization in the nonlinear least squares optimization.")
    return density_mean_total[0], T0, chi, c_chi
