"""Module to initialize the pressure profile using MSIS.
Latest update: Jan 30th, 2023 [OI]
"""
import numpy as np
from pymsis import msis
import astropy.units as u
from Applications.SpaceWeather.SW1D_sqrt.python.MSIS_reference_values import get_MSIS_species


def MSIS_initial_condition_1D_pressure(x_dg,
                                       altitude_mesh_grid,
                                       parameters,
                                       mass,
                                       T0,
                                       m,
                                       rho0,
                                       H0,
                                       Fr,
                                       R0,
                                       number_of_components=3,
                                       number_of_dimensions=1):
    """

    :param R0: (float)
                reference length scale ratio mesh lower. (units: dimensionless).
    :param H0: (float)
                reference scale height. (units: m).
    :param Fr: (float)
            Froude number. (units: dimensionless).
    :param rho0: (float)
            reference density at the lower altitude boundary (e.g. @ 100km) (units: kg/m^3).
    :param m: (float)
            mass of atomic oxygen. (units: kg)
    :param T0: (float)
            reference temperature at the lower altitude boundary (e.g. @ 100km). (units: Kelvin).
    :param x_dg: (1d ndarray)
                DG mesh nodes.
    :param parameters: (dictionary)
                    list of input parameters defined in pdeapp.py.
    :param number_of_components: (float)
                                default is 3.
    :param number_of_dimensions: (float)
                                default is 1.
    :param altitude_mesh_grid: (ndarray)
                                altitude dg mesh grid (units: km).
    :param mass: (ndarray)
                mass of neutrals. (units: kg).

    :return: (ndarray) :
            (1) log_rho   # log(rho)
            (2) sqrt_rho_temperature   # sqrt(rho) * T
            (3) central_log_rho    # dr/dx
            (4) central_sqrt_rho_temperature    # d(sqrt(rho) * T)/dx
            used for initialization.
    """
    # get data (F10.7, F10.7_81, Ap) needed to run MSIS.
    f10p7_msis, f10p7a_msis, ap_msis = msis.get_f107_ap(dates=parameters["date"])

    # run MSIS, output is a tensor of dimensions: (n_dates, n_longitude, n_latitude, n_altitude, 11)
    # 11 stands for each species in the following order:
    #  (1) Total mass density (kg / m3),  (2) N2 density (m-3), (3) O2 density (m-3), (4) O density (m-3),
    #  (5) He density (m-3), (6) H density (m-3), (7) Ar density (m-3), (8) N density (m-3),
    #  (9) Anomalous oxygen density (m-3), (10) NO density (m-3), (11) Temperature(K)
    MSIS_output_center = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                                  lons=parameters["longitude"].value,  # (list of floats) – Longitudes of interest
                                  lats=parameters["latitude"].value,  # (list of floats) – Latitudes of interest
                                  alts=altitude_mesh_grid.value,  # (list of floats) – Altitudes of interest
                                  # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                                  f107s=f10p7_msis,
                                  # F10.7 running 81-day average centered on the given date(s)
                                  f107as=f10p7a_msis,
                                  # Daily Ap
                                  aps=[ap_msis])

    MSIS_output_minus = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                                 lons=parameters["longitude"].value,  # (list of floats) – Longitudes of interest
                                 lats=parameters["latitude"].value,  # (list of floats) – Latitudes of interest
                                 alts=altitude_mesh_grid.value - parameters["initial_dr"].value,
                                 # (list of floats) – Altitudes of interest
                                 # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                                 f107s=f10p7_msis,
                                 # F10.7 running 81-day average centered on the given date(s)
                                 f107as=f10p7a_msis,
                                 # Daily Ap
                                 aps=[ap_msis])

    MSIS_output_plus = msis.run(dates=parameters["date"],  # (list of dates) – Dates and times of interest
                                lons=parameters["longitude"].value,  # (list of floats) – Longitudes of interest
                                lats=parameters["latitude"].value,  # (list of floats) – Latitudes of interest
                                alts=altitude_mesh_grid.value + parameters["initial_dr"].value,
                                # (list of floats) – Altitudes of interest
                                # list of floats, optional) – Daily F10.7 of the previous day for the given date(s)
                                f107s=f10p7_msis,
                                # F10.7 running 81-day average centered on the given date(s)
                                f107as=f10p7a_msis,
                                # Daily Ap
                                aps=[ap_msis])

    # todo:
    #  [TAll, rhoAll] = atmosnrlmsise00(h, lat, long, year, doy, sec, LST, F10p7a, F10p7, aph, flags);
    #  [TAm, rhoAm] = atmosnrlmsise00(h - dr, lat, long, year, doy, sec, LST, F10p7a, F10p7, aph, flags);
    #  [TAp, rhoAp] = atmosnrlmsise00(h + dr, lat, long, year, doy, sec, LST, F10p7a, F10p7, aph, flags);
    MSIS_center = get_MSIS_species(MSIS_output=MSIS_output_center, parameters=parameters)[:, 0, 0, :]
    MSIS_minus = get_MSIS_species(MSIS_output=MSIS_output_minus, parameters=parameters)[:, 0, 0, :]
    MSIS_plus = get_MSIS_species(MSIS_output=MSIS_output_plus, parameters=parameters)[:, 0, 0, :]

    # todo:
    #  rho = rhoAll(:, indices)*mass * m / rho0;
    #  rhom = rhoAm(:, indices)*mass * m / rho0;
    #  rhop = rhoAp(:, indices)*mass * m / rho0;

    # todo: Jordi, what are the dimensions of these values?
    rho_center = MSIS_center.T @ mass * m / rho0
    rho_minus = MSIS_minus.T @ mass * m / rho0
    rho_plus = MSIS_plus.T @ mass * m / rho0

    # todo:
    #  Jordi, can we go over the dimensions here as well?
    #  mass0 = (rhoAll(:, indices) * mass)./ sum(rhoAll(:, indices), 2);
    #  massm = (rhoAm(:, indices) * mass)./ sum(rhoAm(:, indices), 2);
    #  massp = (rhoAp(:, indices) * mass)./ sum(rhoAp(:, indices), 2);
    mass_center = MSIS_center.value.T @ mass.value / np.sum(MSIS_center.value, axis=0)
    mass_minus = MSIS_minus.value.T @ mass.value / np.sum(MSIS_minus.value, axis=0)
    mass_plus = MSIS_plus.value.T @ mass.value / np.sum(MSIS_plus.value, axis=0)

    T_center = MSIS_output_center[0, 0, 0, :, -1] / T0
    T_minus = MSIS_output_minus[0, 0, 0, :, -1] / T0
    T_plus = MSIS_output_plus[0, 0, 0, :, -1] / T0

    # todo:
    #  rT = rho. * T. / mass0;
    #  rTm = rhom. * Tm. / massm;
    #  rTp = rhop. * Tp. / massp;
    # vector operation --> return same size.
    rho_temperature_center = (rho_center * T_center) / mass_center
    rho_temperature_minus = (rho_minus * T_minus) / mass_minus
    rho_temperature_plus = (rho_plus * T_plus) / mass_plus

    # todo:
    #  drT = rTp - rTm;
    #  drTdr = H * drT / (2 * dr);
    central_rho_temperature = H0 * (rho_temperature_plus - rho_temperature_minus) / (2 * parameters["initial_dr"])

    # todo:
    #  acc = (Fr ^ 2 * xdg * cos(lat0) ^ 2 - (r0. / xdg). ^ 2);
    #  rho = drTdr. / acc;
    acc = (Fr ** 2) * x_dg * (np.cos(parameters["latitude"].to(u.rad).value) ** 2) - (R0 / x_dg) ** 2
    rho = central_rho_temperature / acc

    # todo:
    #  T = mass0. * rT. / rho;
    #  T = T - T(1) + 1;
    #  rho = rho / rho(1);
    #  r = log(rho);
    #  srT = sqrt(rho). * T;
    temperature = mass_center * rho_temperature_center / rho_center
    temperature = temperature.value - temperature[0].value + 1
    rho = rho / rho[0]
    log_rho = np.log(rho)
    sqrt_rho_temperature = np.sqrt(rho) * temperature

    # todo:
    #  drdx = gradient(r). / gradient(xdg);
    #  dsrTdx = gradient(srT). / gradient(xdg);
    central_log_rho = np.gradient(f=log_rho, edge_order=2) / np.gradient(f=x_dg, edge_order=2)
    central_sqrt_rho_temperature = np.gradient(f=sqrt_rho_temperature, edge_order=2) / np.gradient(f=x_dg, edge_order=2)

    # todo:
    #  u = zeros(npoints, nc * (nd + 1));
    #  iu = [1, nc, nc + 1, nc * (nd + 1)];
    #  Jordi, could you help me translate this line?
    #  u(:, iu) = [r, srT, drdx, dsrTdx];
    #  can we avoid the transposing in pdeparams and set it up in the current shape here?
    results = np.zeros((len(altitude_mesh_grid), number_of_components * (number_of_dimensions + 1)))

    index_results = np.array([1, number_of_components, number_of_components + 1,
                              number_of_components * (number_of_dimensions + 1)]) - 1

    results[:, index_results] = np.array([log_rho.value,  # log(rho)
                                          sqrt_rho_temperature.value,  # sqrt(rho) * T
                                          central_log_rho.value,   # dr/dx
                                          central_sqrt_rho_temperature.value]).T   # d(sqrt(rho) * T)/dx
    return results
