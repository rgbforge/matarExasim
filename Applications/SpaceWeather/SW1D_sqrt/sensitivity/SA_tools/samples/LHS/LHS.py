import os

import numpy as np
from Applications.SpaceWeather.SW1D_sqrt.sensitivity.SA_tools.sobol import generate_tensor_c, generate_tensor_d
import matplotlib.pyplot as plt
from scipy.stats import qmc
import matplotlib

font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


# dimensionality setting: number of samples and number of parameters
N = int(1e4)
d = 8
folder = "LHS"

# sample 2N of the input parameter.
sampler = qmc.LatinHypercube(d=d)
sample = sampler.random(n=2*N)

A = sample[:N, :]
B = sample[N:, :]

# lower and upper bounds
#          mu0, kappa0, epsilon, F_{10.7}, F_{10.7}^{81}, Ap, nu_{eddy}, alpha_{eddy}
l_bounds = [1.3E-5, 5.6E-5, 0.05, -10, -2, -2/3, 0, 0]
u_bounds = [1.3E-3, 5.6E-3, 0.7, 10, 2, 2/3, 1000, 100]

# scale and sample
A_scaled = qmc.scale(sample=A,
                     l_bounds=l_bounds,
                     u_bounds=u_bounds)

B_scaled = qmc.scale(sample=B,
                     l_bounds=l_bounds,
                     u_bounds=u_bounds)


# save samples [A, B]
np.save(file=os.getcwd() + "/" + str(folder) + "/A_sample_scaled_" + str(N), arr=A_scaled)
np.save(file=os.getcwd() + "/" + str(folder) + "/B_sample_scaled_" + str(N), arr=B_scaled)

# generate matrix C=[C1, C2, C3, ..., C11].
C = generate_tensor_c(A=A_scaled, B=B_scaled, d=d)
np.save(file=os.getcwd() + "/" + str(folder) + "/C_sample_scaled_" + str(N), arr=C)

# generate matrix D=[D1, D2, D3, ..., D11].
D = generate_tensor_d(A=A_scaled, B=B_scaled, d=d)
np.save(file=os.getcwd() + "/" + str(folder) + "/D_sample_scaled_" + str(N), arr=D)

fig, ax = plt.subplots(ncols=d-1, nrows=d-1, figsize=(30, 30))

for jj in range(d-1):
    for ii in range(d-1):
        if jj <= ii:
            ax[ii, jj].scatter(A_scaled[:, jj], A_scaled[:, ii+1], s=2, c="r", alpha=0.2, label="A")
            ax[ii, jj].scatter(B_scaled[:, jj], B_scaled[:, ii+1], s=2, c="b", alpha=0.2, label="B")

        else:
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
            ax[ii, jj].spines['top'].set_visible(False)
            ax[ii, jj].spines['right'].set_visible(False)
            ax[ii, jj].spines['bottom'].set_visible(False)
            ax[ii, jj].spines['left'].set_visible(False)

ax[0, 0].legend()
ax[-1, 0].set_xlabel(r"$\mu_{0}$")
ax[-1, 1].set_xlabel(r"$\kappa_{0}$")
ax[-1, 2].set_xlabel(r"$\epsilon$")
ax[-1, 3].set_xlabel(r"$F_{10.7}$")
ax[-1, 4].set_xlabel(r"$F_{10.7}^{81}$")
ax[-1, 5].set_xlabel(r"$A_{p}$")
ax[-1, 6].set_xlabel(r"$\nu_{eddy}$")


ax[0, 0].set_ylabel(r"$\kappa_{0}$")
ax[1, 0].set_ylabel(r"$\epsilon$")
ax[2, 0].set_ylabel(r"$F_{10.7}$")
ax[3, 0].set_ylabel(r"$F_{10.7}^{81}$")
ax[4, 0].set_ylabel(r"$A_{p}$")
ax[5, 0].set_ylabel(r"$\nu_{eddy}$")
ax[6, 0].set_ylabel(r"$\alpha_{eddy}$")


fig.suptitle("Random Samples N = " + str(N))
plt.tight_layout()
plt.savefig(os.getcwd() + "/" + str(folder) + "/random_samples_" + str(N) + ".png", dpi=500)
plt.show()