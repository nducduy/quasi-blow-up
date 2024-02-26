import math

import numpy as np
import scipy
from scipy.special import gamma as gamma_func
from scipy.special import hyp2f1 as hyp_2f1
import mpmath as mp #with special hypergeometric functions
from scipy.special import gammainc as gamma_incomp_func #lower incomplete gamma func

#from utils import utils



class rsteinstein3(object):
    """
    Class for generating paths of the rStein-Stein model -Version3.
    """
    def __init__(self, n_steps = 100, N = 1000, T = 1.00,H=0.5,X0=1,S0=1,r=0,rho=0):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n_steps # Granularity (#steps per year)
        self.dt = T/n_steps # Step size
        self.a = H-0.5 # Alpha
        self.N = N # number of sample Paths
        self.S0=S0
        self.X0=X0
        self.r=r
        self.H=H # Hurt index
        self.rho=rho #correlation coefficient

    def compute_covariance_RL(self,H=0.5, t=0.5, s=1):
        self.H=H
        minn = np.minimum(t, s)
        if minn == 0:
            return 0.
        else:
            maxx = np.maximum(s, t)
            alpha = H + 0.5
            factor = minn ** (alpha) / (alpha * maxx ** (1 - alpha))
            # Hypergeometric function defined for |z|<1 and anlytically continued beyond
            return factor * scipy.special.hyp2f1(1, 1 - alpha, 1 + alpha, minn / maxx)

    def simul_paths(self):
        """
                return the variance X and log of stock prices columnwise
                That the correct order should be np.tranpose(X) and np.transpose(S)
        """
        X0=self.X0
        H=self.H
        rho=self.rho

        T=self.T
        S0=self.S0
        r=self.r
        n_steps=self.n



        ####Generate Brownian motions
        MMM = int(self.N/2)  # Monte Carlo Simulations
        NN = self.n  # Time steps
        np.random.seed(1)
        W1 = np.random.normal(0, 1, (NN, MMM))
        W2 = np.random.normal(0, 1, (NN, MMM))

        W1_tilde = np.concatenate((np.zeros(MMM)[np.newaxis, :], W1))
        W2_tilde = np.concatenate((np.zeros(MMM)[np.newaxis, :], W2))

        # Antithetic variates
        W1_anti, W2_anti = - W1_tilde, -W2_tilde
        W = np.concatenate((W1_tilde, W1_anti), axis=1)
        W_perp = np.concatenate((W2_tilde, W2_anti), axis=1)
        #####


        dt = self.dt
        tt = np.linspace(0., T, n_steps + 1)
        ttt = tt[:, np.newaxis]

        w1 = W[:(n_steps + 1), :]
        w2 = W_perp[:(n_steps + 1), :]
        BB = rho * w1 + np.sqrt(1 - rho * rho) * w2

        # Cholesky
        t_vec = np.linspace(dt, T, n_steps)
        cov_matrix = [[self.compute_covariance_RL(H, t, s) for t in t_vec] for s in t_vec]
        L = np.linalg.cholesky(cov_matrix)
        fbm_path = L @ w1[1:, :]

        alpha = H + 0.5
        g0 = X0/np.sqrt(1+ t_vec ** (2*H))

        X = g0[:, np.newaxis]*(1+ fbm_path / gamma_func(alpha))
        X = np.concatenate((X0 * np.ones(2 * MMM)[np.newaxis, :], X))
        x = np.array(X[:-1])
        volatility = x
        volatility = np.concatenate((np.zeros(2 * MMM)[np.newaxis, :], volatility))

        drift_logS = np.log(S0) + r * ttt - 0.5 * dt * np.cumsum(volatility * volatility, axis=0)
        martingale_logS = np.sqrt(dt) * np.cumsum(volatility * BB, axis=0)
        logS = drift_logS + martingale_logS

        return X, logS

