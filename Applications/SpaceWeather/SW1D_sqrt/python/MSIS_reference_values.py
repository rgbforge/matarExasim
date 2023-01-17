""" Module to call MSIS from Python.
Latest update: Jan 13, 2023 [OI]
"""
from pymsis import msis
import numpy as np
import astropy.units as u


def get_MSIS_species(MSIS_output, parameters):
    """A function to get the requested MSIS species results.

    :param MSIS_output: MSIS output for the requested species.
    :param parameters: contains specified input parameters (dict).
    :return: MSIS tensor results dimensions: (nspecies, n_longitude, n_latitude, n_altitude)
    """
    # initialize the dataset we need.
    MSIS = np.zeros((len(parameters["chemical_species"]),
                     parameters["n_longitude_MSIS"],
                     parameters["n_latitude_MSIS"],
                     parameters["n_radial_MSIS"]))

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

def MSIS_reference_values(parameters, mass):
    """A function to get MSIS reference values,
    i.e. the reference density (rho0), temperature (T0), mass fraction (chi), and coefficients of the fit (cchi).

    :param parameters: dictionary defined in pdeapp.py with all the model parameters (solver and physical parameters).
    :param mass: molecular mass of each species (we need it to compute mass density from species number densities).

    :return: (1) rho0 : type: float, units: kilogram/meters^3]
                        initial density todo: verify with Jordi.
             (2) T0 : type: float, units: Kelvin
                        initial temperature at the lower altitude boundary.
             (3) chi : # todo describe size + units
                        mass fractions (rho_i/rho) over altitude
             (4) cchi : # todo describe size + units
                        coefficients ai of the fit: chi ~ a1*exp(a2*(h-H0)) + a3*exp(a4*(h-H0))
    """
    # we do not need the mesh here, we only need the position of the lower and upper boundary
    # we want to obtain certain quantities at the lower boundary + a distribution in space of the partial densities
    # to do that we can use any radial distribution of points we want, not necessarily linked to the mesh
    # Actually we want more points, so don't use resolution, but another parameter. I set it to have 101 points.
    # todo: looks like you use 5000 points... ?
    altitude_mesh = np.linspace(parameters["altitude_lower"].to(u.km).value,
                                parameters["altitude_upper"].to(u.km).value,
                                parameters["n_radial_MSIS"])

    # todo: NEW we also "average" in longitude and latitude
    #  (we then average to get  a "general" quantity, not an instantaneous picture)
    #   need to create a vector of latitudes and longitudes of interest too
    #   angleres = 5;
    #   lat = -90 + angleres:angleres: 90 - angleres;
    #   long = -180:angleres: 180 - angleres;
    #   https://swxtrec.github.io/pymsis/examples/plot_surface_animation.html#sphx-glr-examples-plot-surface-animation-py
    longitude_mesh = np.linspace(-180, 180, parameters["n_longitude_MSIS"])
    latitude_mesh = np.linspace(-90, 90, parameters["n_latitude_MSIS"])

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

    # initialize the dataset we need. dimensions: (nspecies, nlons, nlats, nalts)
    MSIS = get_MSIS_species(MSIS_output=MSIS_output, parameters=parameters)


    # # loop over all altitudes and compute the mean temperature and density.
    # temperature_altitude_mean = np.zeros(len(altitude_mesh))
    # number_density_mean_species = np.zeros((len(altitude_mesh), len(parameters["chemical_species"])))
    # for ii in range(len(altitude_mesh)):
    #     temperature_altitude_mean[ii] = np.mean(output_msis[0, :, :, ii, -1]) * u.K
    #     for jj in range(len(parameters["chemical_species"])):
    #         number_density_mean_species[ii, jj] = np.mean(MSIS[])
    #
    # # get total mass density and temperature.
    # msis_total_mass_density = output_msis[0, :, :, :, 0] * u.kg / u.m ** 3
    # msis_temperature = output_msis[0, :, :, :, -1] * u.K






    # todo: clarify computation of chi.
    #  MATLAB code:
    #  rhoh = nh * mass;
    #  chi = (nh. * mass')./rhoh;

    # # todo: Jordi, could you please help me convert these values? I not sure how to obtain them.
    # return rho0, T0, chi, cchi
    return 0, 0, 0, 0
