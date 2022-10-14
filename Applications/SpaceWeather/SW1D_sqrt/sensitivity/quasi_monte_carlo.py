import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# number of samples
N = 500

# number of parameters
d = 4

# reference density
rho0 = 1.3279e-8

# reference dynamic viscosity
mu0 = 8.063632e-5

# reference thermal conductivity
kappa0 = 0.02167

# lower and upper bounds
l_bounds = [0.2, 1E-2, 1E-2, 1E-1]
u_bounds = [1.5, 1E2, 1E2, 1E1]

# scale and sample using Latin Hypercube Sampling (LHS)
sample = qmc.scale(sample=qmc.LatinHypercube(d=d).random(n=N),
                   l_bounds=l_bounds,
                   u_bounds=u_bounds)

np.save(file="sensitivity/samples/samples", arr=sample)

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 6))
ax[0, 0].scatter(sample[:, 0], sample[:, 1], s=2)
ax[0, 0].set_xlabel(r"EUV efficiency $\epsilon$")
ax[0, 0].set_ylabel(r"Reference Thermal conductivity $\kappa_{0}$")

ax[0, 1].scatter(sample[:, 0], sample[:, 2], s=2)
ax[0, 1].set_xlabel(r"EUV efficiency $\epsilon$")
ax[0, 1].set_ylabel(r"Reference Dynamic Viscosity $\mu_{0}$")
ax[0, 1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))

ax[0, 2].scatter(sample[:, 0], sample[:, 3], s=2)
ax[0, 2].set_xlabel(r"EUV efficiency $\epsilon$")
ax[0, 2].set_ylabel(r"Reference Neutral Density $\rho_{0}$")

ax[1, 0].scatter(sample[:, 1], sample[:, 2], s=2)
ax[1, 0].set_xlabel(r"Reference Thermal conductivity $\kappa_{0}$")
ax[1, 0].set_ylabel(r"Reference Dynamic Viscosity $\mu_{0}$")
ax[1, 0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))

ax[1, 1].scatter(sample[:, 1], sample[:, 3], s=2)
ax[1, 1].set_xlabel(r"Reference Thermal conductivity $\kappa_{0}$")
ax[1, 1].set_ylabel(r"Reference Neutral Density $\rho_{0}$")

ax[1, 2].scatter(sample[:, 2], sample[:, 3], s=2)
ax[1, 2].set_xlabel(r"Reference Dynamic Viscosity $\mu_{0}$")
ax[1, 2].set_ylabel(r"Reference Neutral Density $\rho_{0}$")
#ax[1, 2].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig("sensitivity/figs/samples.png", dpi=500)
plt.show()
