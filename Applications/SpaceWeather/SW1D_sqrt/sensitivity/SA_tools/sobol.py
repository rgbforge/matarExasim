import numpy as np
import copy


def generate_tensor_c(A, B, d):
    """Generate Random variables in a matrix A, B, C.
    :param d: (int) number of random variables.
    :param A: (ndarray) matrix of size (N x d).
    :param B: (ndarray) matrix of size (N x d).
    :return: (ndarray) C tensor, size (d x N x d).
    """
    # copy matrix B and only modify one column.
    C = np.zeros((d, np.shape(B)[0], np.shape(B)[1]))
    for ii in range(d):
        C[ii, :, :] = copy.deepcopy(B)
        C[ii, :, ii] = A[:, ii]
    return C


def generate_tensor_d(A, B, d):
    """Generate random variables in a matrix A, B, D.
    :param d: (int) number of random variables.
    :param A: (ndarray) matrix of size (N x d).
    :param B: (ndarray) matrix of size (N x d).
    :return: (ndarray) C tensor, size (d x N x d).
    """
    # copy matrix A and only modify one column.
    D = np.zeros((d, np.shape(A)[0], np.shape(A)[1]))
    for ii in range(d):
        D[ii, :, :] = copy.deepcopy(A)
        D[ii, :, ii] = B[:, ii]
    return D


def sobol_mc(A, B, C, D, f):
    """Evaluate the QoI given input parameters.
    :param A: (ndarray) sampled matrix A.
    :param B: (ndarray) sampled matrix B.
    :param C: (ndarray) sampled matrix C.
    :param D: (ndarray) sampled matrix D.
    :param f: model function (mapping from input to output)-- output is scalar.
    :return: (ndarray) f(A) size (N x 1), (ndarray) f(B) size (N x 1), (ndarray) f(C) size (N x d),
     (ndarray) f(D) size (N x d)
    """
    # initialize matrices.
    N, d = np.shape(A)
    YA = np.zeros(N)
    YB = np.zeros(N)
    YC = np.zeros((N, d))
    YD = np.zeros((N, d))

    # evaluate function for each sample.
    for ii in range(N):
        YA[ii] = f(z=A[ii, :])
        YB[ii] = f(z=B[ii, :])
        for jj in range(d):
            YC[ii, jj] = f(z=C[jj, ii, :])
            YD[ii, jj] = f(z=D[jj, ii, :])
    return YA, YB, YC, YD


def estimate_sobol(YA, YB, YC, type="sobol"):
    """Estimate Sobol' indices (main effect) Vi/V and (total effect) Ti/T.
    :param YA: (ndarray) output of input sampled matrix A.
    :param YB: (ndarray) output of input sampled matrix B.
    :param YC: (ndarray) output of input sampled matrix C.
    :param type: (str) type of estimator ("sobol", "owen", "saltelli_I", "saltelli_II", "janon", etc...)
    :return: (ndarray) S (main effect) size (d x 1), (ndarray) T (total effect) size (d x 1)
    """
    main_effect = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=len(YA), type_estimator=type)
    total_effect = estimator_total_effect(YA=YA, YB=YB, YC=YC, N=len(YA), type_estimator=type)
    return main_effect, total_effect


def estimator_main_effect(YA, YB, YC, N, type_estimator="sobol"):
    """Computes the main effect sobol indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: (int) number of samples.
    :param type_estimator: (str) type of estimator ("sobol", "owen", "saltelli", "janon", etc...)
    :return: (ndarray) sobol indices main effect, size (d x 1).
    """
    if type_estimator == "owen":
        """
        A. B. Owen, Variance components and generalized Sobol’ indices, SIAM/ASA Journal on Uncertainty
        Quantification, 1 (2013), pp. 19–41, https://doi.org/10.1137/120876782, http://dx.doi.org/10.1137/
        120876782, https://arxiv.org/abs/http://dx.doi.org/10.1137/120876782.
        """
        V = np.mean(YA ** 2) - np.mean(YA) ** 2
        return ((2 * N) / (2 * N - 1) * (1 / N * YA.T @ YC -
                                         ((np.mean(YA) + np.mean(YC)) / 2) ** 2 +
                                         (np.var(YA) + np.var(YC)) / (4 * N))) / V
    if type_estimator == "sobol":
        """
        A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        y = np.r_[YA, YB]
        f02 = np.mean(YA) ** 2
        return (1 / N * YA.T @ YC - f02) / np.var(y)

    if type_estimator == "saltelli":
        """
        Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, Marco Ratto, Stefano Tarantola,
        Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index,
        Computer Physics Communications
        """
        f02 = np.mean(YA * YB)
        y = np.r_[YA, YB]
        return ((1 / N) * YA.T @ YC - f02) / np.var(y)

    if type_estimator == "janon":
        """ 
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        f02 = np.mean(np.r_(YA, YB))
        return (1 / N * YA.T @ YC - f02) / (np.mean(YA ** 2) - f02)


def estimator_second_effect(ii, jj, YA, YB, YC, YD, N, type_estimator="sobol"):
    """Computes the second "interation" sobol indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param YD: (ndarray) output of sampled matrix D size (N x d).
    :param type_estimator: (str) type of estimator ("sobol", "owen", "saltelli", "janon", etc...)
    :param N: (int) number of samples.
    :return: (ndarray) sobol indices main effect, size (d x 1).
    """
    """A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
    Computer Physics Communications, 145 (2002), pp. 280 – 297.
    """
    y = np.r_[YA, YB]
    Vij = np.mean(YC[:, jj] * YD[:, ii] - YA * YB) / np.var(y)
    Si = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=N, type_estimator=type_estimator)[ii]
    Sj = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=N, type_estimator=type_estimator)[jj]
    return Vij - Si - Sj


def estimator_total_effect(YA, YB, YC, N, type_estimator="sobol"):
    """Computes the total effect sobol indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: number of samples.
    :param type_estimator: "sobol", "owen", "saltelli_I", "saltelli_II", "janon", etc...
    :return: (ndarray) sobol total effect indices, size (d times 1).
    """
    if type_estimator == "sobol":
        """unbiased
        A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) ** 2
        return 1 - (1 / N * YB.T @ YC - f02) / np.var(YA)

    if type_estimator == "saltelli_I":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA) * np.mean(YB)
        return 1 - (1 / (N - 1) * YB.T @ YC - f02) / (np.mean(YA ** 2) - np.mean(YA) ** 2)

    if type_estimator == "saltelli_II":
        """bias O(1/n)
        A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        f02 = np.mean(YA * YB)
        return 1 - (1 / (N - 1) * YB.T @ YC - f02) / (np.mean(YA ** 2) - np.mean(YA) ** 2)

    if type_estimator == "janon":
        """ bias O(1/n)
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        f02 = np.mean(np.append(YA, YB))
        return 1 - (1 / N * YB.T @ YC - f02) / (np.mean(YA ** 2) - f02)

    if type_estimator == "owen":
        """unbaised
        A. B. Owen, Variance components and generalized Sobol’ indices, SIAM/ASA Journal on Uncertainty
        Quantification, 1 (2013), pp. 19–41, https://doi.org/10.1137/120876782, http://dx.doi.org/10.1137/
        120876782, https://arxiv.org/abs/http://dx.doi.org/10.1137/120876782.
        """
        T = np.zeros(np.shape(YC)[1])
        for ii in range(np.shape(YC)[1]):
            T[ii] = (1 / (2 * N) * np.sum((YB - YC[:, ii]) ** 2)) / (np.mean(YA ** 2) - np.mean(YA) ** 2)
        return T