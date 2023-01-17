""" Module including the 1D GITM sqrt-MSIS formulation flux and source functions.
Latest update: Oct 13th, 2022. [OI]
"""
from numpy import *
from sympy import exp, sqrt, log, pi, tanh, sin, cos, asin


def mass(u, q, w, v, x, t, mu, eta):
    m = array([1.0, 1.0, 1.0])
    return m


def flux(u, q, w, v, x, t, mu, eta):
    fi = fluxInviscid(u, x, mu, eta)
    fv = fluxViscous(u, q, x, mu, eta)

    f = fi + fv
    return f


def fluxWall(u, q, w, v, x, t, mu, eta):
    fi = fluxInviscidWall(u, x, mu, eta)
    fv = fluxViscousWall(u, q, x, mu, eta)

    f = fi + fv
    return f


def source(u, q, w, v, x, t, mu, eta):
    x1 = x[0]

    gam = mu[0]
    gam1 = gam - 1
    Gr = mu[1]
    Pr = mu[2]
    c23 = 2.0 / 3.0

    mw, dmdr = weightedMass(x, mu, eta)

    r = u[0]
    srvx = u[1]
    srT = u[2]

    rho = exp(r)
    sr = sqrt(rho)
    sr1 = 1 / sr
    r_1 = r - 1

    vx = srvx * sr1
    T = srT * sr1
    p = srT / gam

    drdx = -q[0]
    drvxdx = -q[1]
    drTdx = -q[2]

    dvxdx = sr1 * drvxdx - 0.5 * drdx * vx
    dTdx = sr1 * drTdx - 0.5 * drdx * T

    # Viscosity
    expmu = mu[11]
    expKappa = mu[12]
    nuEddy = mu[13]
    alphaEddy = mu[14]

    mustar = T**expmu
    k0 = thermalConductivity(x,mu,eta)
    kstar = k0*T **expKappa

    nu = (mustar * sr1 + sr*nuEddy)/sqrt(gam * Gr)
    fc = (kstar * sr1 + sr*alphaEddy)*mw*sqrt(gam / Gr) / Pr

    trr = nu * c23 * 2 * dvxdx - 2 * c23 * vx / x1
    trd = nu * 4 * (dvxdx - vx / x1) / x1

    R0 = mu[15]
    gravity0 = 1 / gam
    gravity = gravity0 * R0 ** 2 / x1 ** 2
    Fr = mu[3]
    ar = -gravity + x1 * Fr ** 2 / gam

    trp = 2 * c23 * nu * (dvxdx ** 2 - 2 * vx * dvxdx / x1 + vx ** 2 / (x1 ** 2))
    SigmadV = gam * gam1 * trp

    q_EUV = EUVsource1D(u, x, t, mu, eta)

    s = array([r_1 * dvxdx - 2 * vx / x1,
               sr * ar + 0.5 * (dvxdx - 2 * vx / x1) * srvx - 0.5 * p * drdx + 0.5 * trr * drdx + 0.5 * trd,
               sr * q_EUV + (3 / 2 - gam) * srT * dvxdx + 2 * (1 / 2 - gam) * srT * vx / x1 + fc * dTdx * (
                       2 / x1 + 0.5 * drdx) + SigmadV])

    return s


def fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    tau = gettau(uhat, mu, eta, x, n)

    f = fluxWall(u, q, w, v, x, t, mu, eta)
    fw0 = f[0] * n[0] + tau[0] * (u[0] - uhat[0])
    fw1 = f[1] * n[0] + tau[1] * (u[1] - uhat[1])
    fw2 = f[2] * n[0] + tau[2] * (u[2] - uhat[2])
    fw = array([fw0, fw1, fw2])

    # Inviscid outer boundary
    fi = fluxInviscid(u, x, mu, eta)
    ft0 = fi[0] * n[0] + tau[0] * (u[0] - uhat[0])
    ft1 = fi[1] * n[0] + tau[1] * (u[1] - uhat[1])
    ft2 = fi[2] * n[0] + tau[2] * (u[2] - uhat[2])
    ft = array([ft0, ft1, ft2])

    fb = hstack((fw, ft))
    fb = reshape(fb, [3, 2], 'F')
    return fb


def ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau):
    Tbot = 1.0

    # Isothermal Wall
    r = u[0]
    rho = exp(r)
    sr = sqrt(rho)

    utw1 = array([r, 0.0, sr * Tbot])

    # Inviscid wall
    utw2 = u

    ub = hstack((utw1, utw2))
    ub = reshape(ub, (3, 2), 'F')
    return ub


def initu(x, mu, eta):
    """

    :param x:
    :param mu:
    :param eta:
    :return:
    """
    x1 = x[0]

    Fr = mu[3]

    mw = weightedMass(x, mu, eta)

    Tbot = 1.0
    Ttop = 6.0
    R0 = mu[15]
    Ldim = mu[18]
    h0 = 35000 / Ldim

    a0 = (-1 + (Fr ** 2) * R0)

    T = Ttop - (Ttop - Tbot) * exp(-(x1 - R0) / h0)
    logp_p0 = a0 * mw * h0 / Ttop * log(1 + Ttop / Tbot * (exp((x1 - R0) / h0) - 1))
    rtilde = logp_p0 - log(T) + log(mw)
    rho = exp(rtilde)
    srT = sqrt(rho) * T

    u0 = array([rtilde, 0.0, srT])
    return u0


def stab(u1, q1, w1, v1, x, t, mu, eta, uhat, n, tau, u2, q2, w2, v2):
    uhat = 0.5 * (u1 + u2)
    tau = gettau(uhat, mu, eta, x, n)

    ftau0 = tau[0] * (u1[0] - u2[0])
    ftau1 = tau[1] * (u1[1] - u2[1])
    ftau2 = tau[2] * (u1[2] - u2[2])
    ftau = array([ftau0, ftau1, ftau2])
    return ftau


def fluxInviscid(u, x, mu, eta):
    gam = mu[0]
    r = u[0]
    srvx = u[1]
    srT = u[2]

    rho = exp(r)
    sr = sqrt(rho)
    sr1 = 1 / sr

    vx = srvx * sr1

    mw = weightedMass(x, mu, eta)
    p = srT / (gam + mw)

    fi = array([r * vx, srvx * vx + p, srT * vx])
    return fi


def fluxInviscidWall(u, x, mu, eta):
    gam = mu[0]
    r = u[0]
    rho = exp(r)
    sr = sqrt(rho)

    Tbot = mu[7]
    mw = weightedMass(x, mu, eta)
    p = sr * Tbot / (gam * mw)

    fi = array([0.0, p, 0.0])
    return fi


def fluxViscous(u, q, x, mu, eta):
    x1 = x[0]

    gam = mu[0]
    Gr = mu[1]
    Pr = mu[2]
    c23 = 2.0 / 3.0
    mw = weightedMass(x, mu, eta)

    r = u[0]
    srvx = u[1]
    srT = u[2]

    rho = exp(r)
    sr = sqrt(rho)
    sr1 = 1 / sr

    vx = srvx * sr1
    T = srT * sr1

    drdx = -q[0]
    drvxdx = -q[1]
    drTdx = -q[2]

    dvxdx = sr1 * drvxdx - 0.5 * drdx * vx
    dTdx = sr1 * drTdx - 0.5 * drdx * T

    # Viscosity
    expmu = mu[11]
    expKappa = mu[12]
    nuEddy = mu[13]
    alphaEddy = mu[14]

    mustar = T**expmu
    k0 = thermalConductivity(x,mu,eta)
    kstar = k0*T **expKappa

    nu = (mustar * sr1 + sr*nuEddy)/sqrt(gam * Gr)
    fc = (kstar * sr1 + sr*alphaEddy)*mw*sqrt(gam / Gr) / Pr

    trr = nu * c23 * 2 * dvxdx - 2 * c23 * vx / x1

    fv = array([0, -trr, -fc * dTdx])
    return fv


def fluxViscousWall(u, q, x, mu, eta):
    x1 = x[0]

    gam = mu[0]
    Gr = mu[1]
    Pr = mu[2]
    c23 = 2.0 / 3.0
    mw = weightedMass(x, mu, eta)


    r = u[0]
    rho = exp(r)
    sr = sqrt(rho)
    sr1 = 1 / sr
    vx = 0.0
    T = 1.0

    drdx = -q[0]
    drvxdx = -q[1]
    drTdx = -q[2]

    dvxdx = sr1 * drvxdx - 0.5 * drdx * vx
    dTdx = sr1 * drTdx - 0.5 * drdx * T
    
    # Viscosity
    expmu = mu[11]
    expKappa = mu[12]
    nuEddy = mu[13]
    alphaEddy = mu[14]

    mustar = T**expmu
    k0 = thermalConductivity(x,mu,eta)
    kstar = k0*T **expKappa

    nu = (mustar * sr1 + sr*nuEddy)/sqrt(gam * Gr)
    fc = (kstar * sr1 + sr*alphaEddy)*mw*sqrt(gam / Gr) / Pr

    trr = nu * c23 * 2 * dvxdx - 2 * c23 * vx / x1

    fv = array([0, -trr, -fc * dTdx])
    return fv


def gettau(uhat, mu, eta, x, n):
    gam = mu[0]
    Gr = mu[1]
    Pr = mu[2]
    mw = weightedMass(x, mu, eta)

    r = uhat[0]
    srvx = uhat[1]
    srT = uhat[2]

    rho = exp(r)
    sr = sqrt(rho)
    sr1 = 1 / sr
    T = srT * sr1

    #     vx = srvx*sr1
    #     c = sqrt(T);
    #     tauA = sqrt(vx*vx) + c
    tauA = mu[21]

    # Viscosity
    expmu = mu[11]
    expKappa = mu[12]
    nuEddy = mu[13]
    alphaEddy = mu[14]

    mustar = T**expmu
    k0 = thermalConductivity(x,mu,eta)
    kstar = k0*T **expKappa

    tauDv = (mustar * sr1 + sr*nuEddy)/sqrt(gam * Gr)
    tauDT = (kstar * sr1 + sr*alphaEddy)*mw*sqrt(gam / Gr) / Pr

    tau = array([tauA, tauA + tauDv, tauA + tauDT])
    return tau


def EUVsource1D(u, x, t, mu, eta):

    nspecies = 4
    nWaves = 37

    # Computation
    r = x[0]

    gam = mu[0]
    gam1 = gam - 1.0

    Fr = mu[3]
    omega = Fr / sqrt(gam)
    K0 = mu[4]
    M0 = mu[5]

    R0 = mu[15]
    H0 = mu[17]

    latitude = mu[22] * pi / 180
    longitude = mu[23] * pi / 180
    declinationSun0 = mu[7] * pi / 180
    doy = mu[10]
    t0 = mu[20]
    seconds = t*t0
    Ndays = doy + seconds/86400

    declinationSun = asin(-sin(declinationSun0)*cos(2*pi*(Ndays+9)/365.24 + pi*0.0167*2*pi*(Ndays-3)/365.24))

    # computation of angles
    # define local time
    long_offset = omega*t + 2*pi*doy - 3*pi/4
    localTime = longitude + long_offset
    cosChi = sin(declinationSun) * sin(latitude) + cos(declinationSun) * cos(latitude) * cos(localTime)

    absSinChi = sqrt(1 - cosChi ** 2)

    # computation F10.7 (let's assume it constant at first, the variation is at another scale)
    F10p7 = mu[8]
    F10p7_81 = mu[9]
    F10p7_mean = 0.5 * (F10p7 + F10p7_81)

    rtilde = u[0]
    rho = exp(rtilde)
    T = u[2] / sqrt(rho)

    # Quantities
    gravity = (R0 ** 2 / (r ** 2)) / gam
    H = T / (gam * gravity)

    # Chapman integral
    Rp = rho * H
    Xp = r / H
    y = sqrt(Xp / 2) * abs(cosChi)

    Ierf = 0.5 * (1 + tanh(1000 * (8 - y)))
    # todo: should these values be specified in pdeparams?
    # No need to, they don't change, this is a numerical approximation to the error function to guarantee positiveness
    a_erf = 1.06069630
    b_erf = 0.55643831
    c_erf = 1.06198960
    d_erf = 1.72456090
    f_erf = 0.56498823
    g_erf = 0.06651874

    erfcy = Ierf * (a_erf + b_erf * y) / (c_erf + d_erf * y + y * y) + (1 - Ierf) * f_erf / (g_erf + y)

    IcosChi = 0.5 * (1 + tanh(100 * cosChi))
    IsinChi = 0.5 * (1 + tanh(100 * (r * absSinChi - R0)))

    alpha1 = Rp * erfcy * sqrt(0.5 * pi * Xp)
    auxXp = (1 - IcosChi) * IsinChi * Xp * (1 - absSinChi)
    Rg = rho * H * exp(auxXp)
    alpha2 = (2 * Rg - Rp * erfcy) * sqrt(0.5 * pi * Xp)

    alpha = IcosChi * alpha1 + (1 - IcosChi) * (IsinChi * alpha2 + (1 - IsinChi) * 1e2)

    Chi = zeros([nspecies,1])
    dChidr = zeros([nspecies,1])
    Chi[0] = 1.0

    for iSpecies in range(2, nspecies+1):
        coeffsDensity = eta[(3+nspecies)*nWaves+4*(iSpecies-2):(3+nspecies)*nWaves+4*(iSpecies-1)]
        Chi[iSpecies-1] = coeffsDensity[0]*exp(coeffsDensity[1]*(r-R0)*H0) + coeffsDensity[2]*exp(coeffsDensity[3]*(r-R0)*H0)
        Chi[0] = Chi[0] - Chi[iSpecies-1]

    mass = eta[(3+nspecies)*nWaves+4*(nspecies-1):(3+nspecies)*nWaves+4*(nspecies-1)+nspecies]
    #Alert: new change for consistency with total mass (not pushed yet in the Matlab version)
    mw = 0.0
    for iSpecies in range(0, nspecies):
        mw = mw + Chi[iSpecies]/mass[iSpecies]
    mw = 1/mw

    # Compute EUV
    s_EUV = 0
    lambdaw = eta[0:nWaves]
    AFAC = eta[nWaves:2*nWaves]
    F74113 = eta[2*nWaves:3*nWaves]

    for iSpecies in range(0,nspecies):
        crossSection = eta[(3+iSpecies-1)*nWaves:(3+iSpecies)*nWaves]
        tau = M0*Chi(iSpecies)*crossSection*alpha/mass(iSpecies)

        slope0 = 1 + AFAC * (F10p7_mean - 80)
        Islope = 0.5 * (1 + tanh(1000 * (slope0 - 0.8)))
        slopeIntensity = slope0 * Islope + 0.8 * (1 - Islope)
        Intensity0 = F74113 * slopeIntensity
        Intensity = Intensity0 * exp(-tau)

        Q = sum(crossSection * Intensity / lambdaw)
        s_EUV = s_EUV + Chi(iSpecies)*Q*mw/mass(iSpecies)

    eff = mu[6]
    s_EUV = gam * gam1 * eff * s_EUV / K0

    return s_EUV


def weightedMass(x, mu, eta):

    nspecies = 4
    nWaves = 37

    #Position
    r = x[0]

    R0 = mu[15]
    H0 = mu[18]

    #Compute weighted density compositions (n_i/rho = Chi/mi)
    Chi = zeros([nspecies,1])
    dChidr = zeros([nspecies,1])
    Chi[0] = 1.0

    for iSpecies in range(2, nspecies+1):
        coeffsDensity = eta[(3+nspecies)*nWaves+4*(iSpecies-2):(3+nspecies)*nWaves+4*(iSpecies-1)]
        Chi[iSpecies-1] = coeffsDensity[0]*exp(coeffsDensity[1]*(r-R0)*H0) + coeffsDensity[2]*exp(coeffsDensity[3]*(r-R0)*H0)
        Chi[0] = Chi[0] - Chi[iSpecies-1]

        dChidr[iSpecies-1] = (coeffsDensity[0]*coeffsDensity[1]*exp(coeffsDensity[1]*(r-R0)*H0) + coeffsDensity[2]*coeffsDensity[3]*exp(coeffsDensity[3]*(r-R0)*H0))*H0
        dChidr[0] = dChidr[0] - dChidr[iSpecies-1]

    mass = eta[(3+nspecies)*nWaves+4*(nspecies-1):(3+nspecies)*nWaves+4*(nspecies-1)+nspecies]

    mw = 0.0
    dmdr = 0.0
    #Alert: new change for consistency with total mass (not pushed yet in the Matlab version)
    for iSpecies in range(0, nspecies):
        mw = mw + Chi[iSpecies]/mass[iSpecies]
        dmdr = dmdr + dChidr[iSpecies]/mass[iSpecies]
    mw = 1/mw
    dmdr = -mw*mw*dmdr

    return mw, dmdr

def thermalConductivity(x, mu, eta):
    nspecies = 4
    nWaves = 37

    #Position
    r = x[0]

    R0 = mu[15]
    H0 = mu[18]

    #Compute weighted density compositions (n_i/rho = Chi/mi)
    Chi = zeros([nspecies,1])
    dChidr = zeros([nspecies,1])
    Chi[0] = 1.0

    for iSpecies in range(2, nspecies+1):
        coeffsDensity = eta[(3+nspecies)*nWaves+4*(iSpecies-2):(3+nspecies)*nWaves+4*(iSpecies-1)]
        Chi[iSpecies-1] = coeffsDensity[0]*exp(coeffsDensity[1]*(r-R0)*H0) + coeffsDensity[2]*exp(coeffsDensity[3]*(r-R0)*H0)
        Chi[0] = Chi[0] - Chi[iSpecies-1]

    mass = eta[(3+nspecies)*nWaves+4*(nspecies-1):(3+nspecies)*nWaves+4*(nspecies-1)+nspecies]

    kappa = 0.0
    ckappai = eta[(3+nspecies)*nWaves+4*(nspecies-1)+nspecies:(3+nspecies)*nWaves+4*(nspecies-1)+2*nspecies]
    for iSpecies in range(0, nspecies):
        kappa = kappa + ckappai[iSpecies]*Chi[iSpecies]

    return kappa