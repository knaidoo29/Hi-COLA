import numpy as np


class Likelihood():

    
    def __init__(self):
        """
        Initialises class and set ups arrays used in likelihood calculations later.
        """
        self.mu_planck = np.array([0.01041, 0.02223, 0.14208])
        self.cov_planck = 1e-9 * np.array(
            [[0.006621, 0.12444, -1.1929],
             [0.12444,  21.344,  -94.001],
             [-1.1929,  -94.001, 1488.4]]
        )
        self.inv_cov_planck = np.linalg.inv(self.cov_planck)
        _, logdet = np.linalg.slogdet(self.cov_planck)
        self.log_norm_planck = (len(self.mu_planck) * np.log(2 * np.pi) + logdet)


    def loglike_Planck(self, theory):
        """
        Computes the Planck compressed likelihdood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for [theta*, w_b, w_bc].

        Returns
        -------
        loglike : float
            Planck compressed log likelihood values.
        """
        delta = theory - self.mu_planck
        chi2 = float(delta @ self.inv_cov_planck @ delta)
        loglike = -0.5*self.log_norm_planck  - 0.5*chi2
        return loglike


