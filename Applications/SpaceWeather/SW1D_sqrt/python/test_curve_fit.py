from MSIS_reference_values import MSIS_reference_values, exp_model
import astropy.units as u
from pandas import read_csv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)

# specify model input parameters for summer solstice.
parameters = {
    "date": "2022-02-01 00:00:00",  # read in data for this day, i.e. F10.7 measurements. year-month-day hr:min:sec
    "t_restart": 0,  # restart at given time step (discrete value)
    "longitude": -117.1611 * u.deg,  # longitude coordinates # todo: try San Diego coords (lat=32.7157, lon=-117.1611)
    "latitude": 32.7157 * u.deg,  # latitude coordinates
    "neutrals_input_file_directory": "../inputs/neutrals.csv",  # neutrals input file location
    "chemical_species": ["O", "N2", "O2", "He"],  # chemical species.
    "altitude_lower": (100 * u.km).to(u.m),  # computational domain altitude lower bound (meters)
    "altitude_upper": (600 * u.km).to(u.m),  # computational domain altitude upper bound (meters)
    "n_radial_MSIS": 101,  # number of mesh points in the radial direction for MSIS simulation
    "n_longitude_MSIS": 72,  # number of mesh points in the longitude direction for MSIS simulation
    "n_latitude_MSIS": 35,  # number of mesh points in the longitude direction for MSIS simulation
}

# read neutrals mass
neutrals = read_csv(parameters["neutrals_input_file_directory"], delimiter=";")

# set species information
i_species = np.zeros(len(parameters["chemical_species"]), dtype=int)
for ii in range(len(parameters["chemical_species"])):
    i_species[ii] = np.where(neutrals.values[:, 0] == parameters["chemical_species"][ii])[0]

# atomic mass unit
amu = 1.66e-27 * u.kg
# mass of neutrals (kg) vector of 4 components (length of species).
mass = np.array(neutrals.values[i_species, 1], dtype=float) * amu

altitude_mesh = np.linspace(parameters["altitude_lower"].to(u.km).value,
                            parameters["altitude_upper"].to(u.km).value,
                            parameters["n_radial_MSIS"])

rho0, T0, chi, c_chi = MSIS_reference_values(parameters=parameters, mass=mass)
atomic_oxygen = np.ones(len(altitude_mesh))
l2_error = np.zeros(len(parameters["chemical_species"]))

fig, ax = plt.subplots(figsize=(6, 5))
for ii in range(1, 4):
    if ii == 1:
        label = "N2"
        c = "black"
    elif ii == 2:
        label = "O2"
        c = "blue"
    elif ii == 3:
        label = "He"
        c = "pink"
    model = exp_model((altitude_mesh*u.km).to(u.m).value,
                      a1=c_chi[ii-1, 0],
                      a2=c_chi[ii-1, 1],
                      a3=c_chi[ii-1, 2],
                      a4=c_chi[ii-1, 3],
                      altitude_low_boundary=parameters["altitude_lower"].to(u.m).value)
    # evaluate l2 error for this iteration.
    l2_error[ii] = np.linalg.norm(chi[:, ii] - model, ord=2)
    # update atomic oxygen density
    atomic_oxygen += -model

    # plot
    ax.plot(chi[:, ii], altitude_mesh, c=c, label=label + str(" model"))
    ax.scatter(model, altitude_mesh, c=c, s=2, label=label + str(" data"))

l2_error[0] = np.linalg.norm(atomic_oxygen - chi[:, 0], ord=2)
ax.plot(chi[:, 0], altitude_mesh, c="r", label="O model")
ax.scatter(atomic_oxygen, altitude_mesh, c="r", s=2, label="O data")

ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(100, 600)
ax.set_xlabel(r"$X_{i}(\rho_{i}/\rho)$")
ax.set_ylabel(r"$h$ (km)")
ax.set_title(str(parameters["date"]))
plt.savefig("../figs/MSIS_coefficients_fit_" + str(parameters["date"]) + ".png", dpi=800)
plt.show()