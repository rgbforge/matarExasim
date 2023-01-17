""" Module to call MSIS from Python.
Latest update: Jan 13, 2023 [OI]
"""
from pymsis import msis
import numpy as np
import astropy.units as u


def MSIS_reference_values(parameters, mass):
    """

    :param parameters: dictionary defined in pdeapp.py with all the model parameters.
    :param mass: molecular mass of each species (we need it to compute mass density from species number densities)
    :return:
    """
    # todo: Jordi, do you know how I can obtain the altitude mesh in "km"??
    # todo: I need to pass it into the msis function below.
    # todo: I think we need to get mesh.dgnodes but that means we should compute the mesh before calling this function.

    # we do not need the mesh here, we only need the position of the lower and upper boundary 
    # we want to obtain certain quantities at the lower boundary + a distribution in space of the partial densities
    # to do that we can use any radial distribution of points we want, not necessarily linked to the mesh
    # Actually we want more points, so don't use resolution, but another parameter. I set it to have 101 points.

    # to get the altitude (mesh) in km: ((dgnodes-R0)*H0 + altitude_lower)/1000
    # R0: inner radius physicsparam[15] , H0 reference scale height, physicsparam[17], altitude_lower = 100e3
    alt_mesh = np.linspace(parameters["altitude_lower"].to(u.km).value,
                           parameters["altitude_upper"].to(u.km).value,
                           parameters["resolution"])

    # NEW to do we also "average" in longitude and latitude (we then average to get  a "general" quantity, not an instantaneous picture)
    # need to create a vector of latitudes and longitudes of interest too

    # get data needed to run MSIS.
    f10p7_msis, f10p7a_msis, ap_msis = msis.get_f107_ap(dates=parameters["date"])

    # run MSIS, output is of dimensions: (ndates, nlons, nlats, nalts, 11)
    # 11 stands for each species in the following order:
    # [Total mass density(kg / m3),  N2  # density (m-3), O2  # density (m-3), O  # density (m-3),
    #  He  # density (m-3), H  # density (m-3), Ar  # density (m-3), N  # density (m-3),
    #  Anomalous oxygen  # density (m-3), NO  # density (m-3), Temperature(K)]
    output_msis = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                           lons=parameters["longitude"].value,  # (list of floats) – Longitudes of interest
                           lats=parameters["latitude"].value,  # (list of floats) – Latitudes of interest
                           alts=alt_mesh,  # todo: (list of floats) – Altitudes of interest
                           # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                           f107s=f10p7_msis + (parameters["F10p7_uncertainty"]*1E22).value,
                           # F10.7 running 81-day average centered on the given date(s)
                           f107as=f10p7a_msis + (parameters["F10p7-81_uncertainty"]*1E22).value,
                           # Daily Ap
                           aps=[ap_msis])
    # get total mass density and temperature.
    msis_total_mass_density = output_msis[0, 0, 0, :, 0] * u.kg/u.m**3
    msis_temperature = output_msis[0, 0, 0, :, -1] * u.kg/u.m**3

    # initialize the dataset we need.
    MSIS = np.zeros((len(parameters["chemical_species"]), parameters["resolution"]))

    # loop over chemical species
    for ii in range(len(parameters["chemical_species"])):
        if parameters["chemical_species"][ii] == "N2":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 1] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "O2":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 2] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "O":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 3] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "He":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 4] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "H":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 5] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "Ar":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 6] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "N":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 7] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "Anomalous oxygen":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 8] * 1/u.m**3
        elif parameters["chemical_species"][ii] == "NO":
            MSIS[ii, :] = output_msis[0, 0, 0, :, 9] * 1/u.m**3

    # todo: Jordi, could you please help me convert these values? I not sure how to obtain them.
    return rho0, T0, chi, cchi