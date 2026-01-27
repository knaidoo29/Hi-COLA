import numpy as np
import sympy as sym

from pathlib import Path
import warnings

from scipy.interpolate import interp1d

from . import redshift

from .model_standard import StandardModel
from .model_horndeski import HorndeskiModel
from .model_cubic_galileon import CubicGalileon
from .model_cubic_galileon_extensions import CubicGalileonExtensions
from .model_ess import ESS


class Sampler():

    """
    A class for constraining cosmological models.
    """
    
    def __init__(self):
        """
        Initialise the sampler class.
        """
        super().__init__()
        self.settings = None
        self.model = None
        self.fixed_params = {}
        self.varied_params = {}
        self.params_info = {}
        self.varied_param2idx = {}
        self.varied_idx2param = {}
        self.Nvaried = None
        self.constraints = None
        self.interp = {}
        self.solver = {'variable1': None, 'variable2': None}
        self.verbose = True
        self.derived_keys = [
            'H0', 
            'Omega_g0', 
            'Omega_b0', 
            'Omega_c0', 
            'Omega_l0',
            'Omega_nu_ur0',
            'Omega_nu_nr0',
            'fphi0',
            'z_star', 
            'r_star', 
            'theta_star', 
            'r_drag',
            'c_s_sq_gt_0', 
            'Q_s_gt_0'
        ]
        self.derived_labels = [
            r'H_{0}', 
            r'\Omega_{g,0}', 
            r'\Omega_{b,0}', 
            r'\Omega_{c,0}', 
            r'\Omega_{\Lambda,0}', 
            r'\Omega_{\nu,0}^{\mathrm{UR}}', 
            r'\Omega_{\nu,0}^{\mathrm{NR}}', 
            r'f_{\phi,0}',
            r'z_{*}', 
            r'r_{*}', 
            r'\theta_{*}', 
            r'r_{d}',
            r'c_{s}^{2}>0', 
            r'Q_{s}>0'
        ]


    # Setup the sampler

    def _check_param_settings(self, param):
        """
        Checks a parameter in settings and adds it either to the fixed parameter dictionary or
        the varied parameter dictionary.

        Parameters
        ----------
        param : str
            Parameter key.
        """
        
        if 'fixed' in self.settings[param]:
            self.fixed_params[param] = self.settings[param]['fixed']
            self.params_info[param] = 'fixed'
        else:
            self.varied_params[param] = {
                'init': self.settings[param]['init'],
                'type': self.settings[param]['prior'][0],
                'prior': [self.settings[param]['prior'][1],self.settings[param]['prior'][2]],
            }
            self.params_info[param] = 'varied'
            self.varied_param2idx[param] = self.Nvaried
            self.Nvaried += 1
    

    def setup(self, settings, model_verbose=False):
        """
        Setups the sampler fixed and varied parameters.
        
        Parameters
        ----------
        settings : dict
            Settings parameter dictionary.
        """

        self.settings = settings

        if self.settings['sampler'] in ['dynesty', 'emcee', 'pocoMC']:
            self.sampler_method = self.settings['sampler']
        else:
            assert False, "Unknown sampler %s." % self.settings['sampler']
        
        if self.verbose:
            print('Using sampler:', self.sampler_method)

        assert 'model' in self.settings, "Must define model."

        if self.settings['model'] == 'GR':
            self.model = StandardModel()
        elif self.settings['model'] == 'Horndeski':
            self.model = HorndeskiModel()
        elif self.settings['model'] == 'CubicGalileon':
            self.model = CubicGalileon()
        elif self.settings['model'] == 'CubicGalileonExtensions':
            self.model = CubicGalileonExtensions()
        elif self.settings['model'] == 'ESS':
            self.model = ESS()
        else:
            assert False, "Unknown model %s." % self.settings['model']

        self.model.verbose = model_verbose
        self.Nvaried = 0

        # check H0_ini
        assert 'H0_ini' in self.settings, "Parameter 'H0_ini' must be defined in settings dictionary."
        self._check_param_settings('H0_ini')

        # check for Omega_b_ini or w_b_ini
        assert 'Omega_b_ini' in self.settings or 'w_b_ini' in self.settings, \
            "Parameter 'Omega_b_ini' or 'w_b_ini' must be defined in settings dictionary."
        if 'Omega_b_ini' in self.settings:
            self._check_param_settings('Omega_b_ini')
        elif 'w_b_ini' in self.settings:
            self._check_param_settings('w_b_ini')

        # check for Omega_c_ini or w_c_ini
        assert 'Omega_c_ini' in self.settings or 'w_c_ini' in self.settings, \
            "Parameter 'Omega_c_ini' or 'w_c_ini' must be defined in settings dictionary."
        if 'Omega_c_ini' in self.settings:
            self._check_param_settings('Omega_c_ini')
        elif 'w_c_ini' in self.settings:
            self._check_param_settings('w_c_ini')
        
        # check w0
        assert 'w0' in self.settings, "Parameter 'w0' must be defined in settings dictionary."
        self._check_param_settings('w0')

        # check wa
        assert 'wa' in self.settings, "Parameter 'wa' must be defined in settings dictionary."
        self._check_param_settings('wa')

        # Mnu 
        assert 'Mnu' in self.settings, "Parameter 'Mnu' must be defined in settings dictionary."
        self._check_param_settings('Mnu')

        if self.settings['model'] != 'GR':

            # check fphi_ini
            assert 'fphi_ini' in self.settings, "Parameter 'fphi_ini' must be defined in settings dictionary."
            self._check_param_settings('fphi_ini')
        
        if self.settings['model'] != 'GR':

            # check phi_ini
            if 'phi_ini' in self.settings:
                self._check_param_settings('phi_ini')
                self.solver['variable1'] = None
            else:
                self.solver['variable1'] = 0
            
             # check phi_ini
            if 'phi_prime_ini' in self.settings:
                self._check_param_settings('phi_prime_ini')
                self.solver['variable2'] = None
            else:
                self.solver['variable2'] = 1
            
            if self.solver['variable1'] is None:
                if self.solver['variable2'] is not None:
                    self.solver['variable1'] = self.solver['variable2']
                    self.solver['variable2'] = None
                else:
                    assert False, "You cannot define both phi_ini and phi_prime_ini since once or both must be defined via the closure equation."

            # check solver method
            if 'method' in self.settings['solver']:
                self.solver['method'] = self.settings['solver']['method']
            else:
                self.solver['method'] = 'Radau'
            
            # check which_root
            if 'which_root' in self.settings['solver']:
                self.solver['which_root'] = self.settings['solver']['which_root']
            else:
                self.solver['which_root'] = 0

        if self.settings['model'] != 'GR':
            
            if self.settings['model'] == 'Horndeski':

                assert 'K' in self.settings, "Definition for K required."
                assert 'G3' in self.settings, "Definition for G3 required."
                assert 'G4' in self.settings, "Definition for G4 required."

                self.model.define_K(self.settings['K']['func'], self.settings['K']['params'])
                self.model.define_G3(self.settings['G3']['func'], self.settings['G3']['params'])
                self.model.define_G4(self.settings['G4']['func'], self.settings['G4']['params'])

            elif self.settings['model'] == 'CubicGalileonExtensions':

                self.model.define_extension(self.settings['extension'])

            self.model.construct_model(lambdify=False)

        else:

            print()
        
        if self.settings['model'] == 'Horndeski':
            
            # run through Horndeski parameters

            for param in self.model.sym['K_G3_G4_syms']:
                assert str(param) in self.settings, "Parameter %s must be defined in settings dictionary." % param
                self._check_param_settings(str(param))
        
        ### Add conditions for parameters from other models...

        if self.settings['model'] == 'CubicGalileonExtensions':

            if self.settings['extension'] == 1 or self.settings['extension'] == 2:

                # check phi_0
                assert 'phi_0' in self.settings, "Parameter 'phi_0' must be defined in settings dictionary."
                self._check_param_settings('phi_0')


        if self.settings['model'] == 'ESS':

            # check f_k2
            assert 'f_k2' in self.settings, "Parameter 'f_k2' must be defined in settings dictionary."
            self._check_param_settings('f_k2')

            # check f_g2
            assert 'f_g2' in self.settings, "Parameter 'f_g2' must be defined in settings dictionary."
            self._check_param_settings('f_g2')
        

        for param in self.varied_params: 
            self.varied_idx2param[self.varied_param2idx[param]] = param

        # Store prior ranges
        self.init_value = np.array([self.varied_params[self.varied_idx2param[i]]['init'] for i in range(0, self.Nvaried)])
        self.prior_min = np.array([self.varied_params[self.varied_idx2param[i]]['prior'][0] for i in range(0, self.Nvaried)])
        self.prior_max = np.array([self.varied_params[self.varied_idx2param[i]]['prior'][1] for i in range(0, self.Nvaried)])
    
        # Constraints information

        self.constraints = []
        allowed_constraints = [
            'Planck', 'H0_LOCAL_ALL', 'H0_LOCAL_SHOES', 'H0_LOCAL_MCP', 'H0_LOCAL_TRGB', 'H0_LOCAL_Type2SN',
            'DESI_DR2_BAO_FULL', 'DESI_DR2_BAO_BGS', 'DESI_DR2_BAO_LRG1', 'DESI_DR2_BAO_LRG2',
            'DESI_DR2_BAO_LRG_ELG', 'DESI_DR2_BAO_ELG', 'DESI_DR2_BAO_QSO', 'DESI_DR2_BAO_LyA',
            'DES_SN_Dovekie'
        ]
        self.constraint2idx = {}

        self.likelihood_switch = {
        }
        for constraint in allowed_constraints:
            self.likelihood_switch[constraint] = False

        for constraint in self.settings['constraints']:
            assert constraint in allowed_constraints, "Constraint %s not supported." % constraint
            if constraint == 'H0_LOCAL_ALL':
                for H0_LOCAL in ['H0_LOCAL_SHOES', 'H0_LOCAL_MCP', 'H0_LOCAL_TRGB', 'H0_LOCAL_Type2SN']:
                    if H0_LOCAL in self.settings['constraints']:
                        assert False, "If H0_LOCAL_ALL specified do not include %s" % H0_LOCAL
                
            if constraint == 'DESI_DR2_BAO_FULL':
                for BAO in ['DESI_DR2_BAO_BGS', 'DESI_DR2_BAO_LRG1', 'DESI_DR2_BAO_LRG2','DESI_DR2_BAO_LRG_ELG', 'DESI_DR2_BAO_ELG', 'DESI_DR2_BAO_QSO', 'DESI_DR2_BAO_LyA']:
                    if BAO in self.settings['constraints']:
                        assert False, "If DESI_DR2_BAO_FULL specified do not include %s" % BAO
            self.likelihood_switch[constraint] = True
            self.constraints.append(constraint)
        
        for (i, constraint) in enumerate(self.settings['constraints']):
            self.constraint2idx[constraint] = i

        assert 'fname' in self.settings, "Must define fname for outputs."
        self.fname = self.settings['fname']

        if self.verbose:

            print('Sampler: Settings')
            print(' - Fixed parameters:')

            for key in self.fixed_params.keys():
                print(' -- %s = %0.6f' % (key, self.fixed_params[key]))

            print(' - Varied parameters:')
            for key in self.varied_params.keys():
                print(' -- %s' % key)
                print(' --- init = %0.6f' % (self.varied_params[key]['init']))
                print(' --- prior_type = %s' % (self.varied_params[key]['type']))
                print(' --- prior_bounds = %s' % (self.varied_params[key]['prior']))
            
            print(' - Constraints:')
            for constraint in self.constraints:
                print(' -- %s' % constraint)

            self.fname = self.settings['fname']
            for constraint in self.constraints:
                self.fname += '_%s' % constraint

        
    # run a model with a set of parameters

    def run_model(self, params):
        """
        Run model with particular parameter settings.
        """
        if self.params_info['H0_ini'] == 'fixed':
            H0_value = self.fixed_params['H0_ini']
        else:
            H0_value = params[self.varied_param2idx['H0_ini']]
        
        if 'Omega_b_ini' in self.params_info:
            if self.params_info['Omega_b_ini'] == 'fixed':
                Omega_b_value = self.fixed_params['Omega_b_ini']
            else:
                Omega_b_value = params[self.varied_param2idx['Omega_b_ini']]
        else:
            if self.params_info['w_b_ini'] == 'fixed':
                Omega_b_value = self.fixed_params['w_b_ini']/((1e-2*H0_value)**2)
            else:
                Omega_b_value = params[self.varied_param2idx['w_b_ini']]/((1e-2*H0_value)**2)
        
        if 'Omega_c_ini' in self.params_info:
            if self.params_info['Omega_c_ini'] == 'fixed':
                Omega_c_value = self.fixed_params['Omega_c_ini']
            else:
                Omega_c_value = params[self.varied_param2idx['Omega_c']]
        else:
            if self.params_info['w_c_ini'] == 'fixed':
                Omega_c_value = self.fixed_params['w_c_ini']/((1e-2*H0_value)**2)
            else:
                Omega_c_value = params[self.varied_param2idx['w_c_ini']]/((1e-2*H0_value)**2)
        
        if self.params_info['w0'] == 'fixed':
            w0_value = self.fixed_params['w0']
        else:
            w0_value = params[self.varied_param2idx['w0']]

        if self.params_info['wa'] == 'fixed':
            wa_value = self.fixed_params['wa']
        else:
            wa_value = params[self.varied_param2idx['wa']]
        
        if self.params_info['Mnu'] == 'fixed':
            Mnu_value = [self.fixed_params['Mnu']]
        else:
            Mnu_value = [params[self.varied_param2idx['Mnu']]]
        
        if self.settings['model'] != 'GR':

            if self.params_info['fphi_ini'] == 'fixed':
                fphi_value = self.fixed_params['fphi_ini']
            else:
                fphi_value = params[self.varied_param2idx['fphi_ini']]

            if self.settings['model'] == 'Horndeski':
                
                K_G3_G4_value = []
                for param in self.model.sym['K_G3_G4_syms']:
                    if self.params_info[str(param)] == 'fixed':
                        K_G3_G4_value.append(self.fixed_params[str(param)])
                    else:
                        K_G3_G4_value.append(params[self.varied_param2idx[str(param)]])
            
            if self.settings['model'] != 'GR':

                if (self.solver['variable1'] == 0 and self.solver['variable2'] == 1) or (self.solver['variable1'] == 1 and self.solver['variable2'] == 0):
                    phi_ini_value = 1e-9
                    phi_prime_ini_value = 1e-9
                else:
                    if self.solver['variable1'] == 0 or self.solver['variable2'] == 0:
                        if self.params_info['phi_prime_ini'] == 'fixed':
                            phi_prime_ini_value = self.fixed_params['phi_prime_ini']
                        else:
                            phi_prime_ini_value = params[self.varied_param2idx['phi_prime_ini']]
                    else:
                        phi_prime_ini_value = 1e-9

                    if self.solver['variable1'] == 1 or self.solver['variable2'] == 1:
                        if self.params_info['phi_ini'] == 'fixed':
                            phi_ini_value = self.fixed_params['phi_ini']
                        else:
                            phi_ini_value = params[self.varied_param2idx['phi_ini']]
                    else:
                        phi_ini_value = 1e-9
                
            # Reset some of these parameters i

            if self.settings['model'] == 'CubicGalileon':
                
                self.solver['variable1'] = 1
                self.solver['variable2'] = None
                phi_ini_value = 1e-9
                phi_prime_ini_value = 1e-9            
            
            elif self.settings['model'] == 'ESS':
                
                self.solver['variable1'] = 1
                self.solver['variable2'] = None
                phi_ini_value = 1e-9
                phi_prime_ini_value = 1e-9

            if self.settings['model'] == 'CubicGalileonExtensions':
                
                if self.settings['extension'] == 1 or self.settings['extension'] == 2:
                    if self.params_info['phi_0'] == 'fixed':
                        f_g2_value = self.fixed_params['phi_0']
                    else:
                        f_g2_value = params[self.varied_param2idx['phi_0']]

            elif self.settings['model'] == 'ESS':
                   
                if self.params_info['f_k2'] == 'fixed':
                    f_k2_value = self.fixed_params['f_k2']
                    f_g2_value = self.fixed_params['f_g2']
                else:
                    f_k2_value = params[self.varied_param2idx['f_k2']]
                    f_g2_value = params[self.varied_param2idx['f_g2']]
        
        if self.settings['model'] == 'GR':
            self.model.set_cosmo_params(
                H0_value, Omega_c_value, Omega_b_value, w0=w0_value, wa=wa_value, mnu=Mnu_value
            )
        elif self.settings['model'] == 'Horndeski':
            self.model.set_cosmo_params(
                H0_value, Omega_c_value, Omega_b_value, fphi_value, K_G3_G4_value, 
                w0=w0_value, wa=wa_value, mnu=Mnu_value
            )
        elif self.settings['model'] == 'CubicGalileon':
            self.model.set_cosmo_params(
                H0_value, Omega_c_value, Omega_b_value, fphi_value,
                w0=w0_value, wa=wa_value, mnu=Mnu_value
            )
        elif self.settings['model'] == 'CubicGalileonExtensions':
            self.model.set_cosmo_params(
                H0_value, Omega_c_value, Omega_b_value, fphi_value, [f_g2_value],
                w0=w0_value, wa=wa_value, mnu=Mnu_value
            )
        elif self.settings['model'] == 'ESS':
            self.model.set_cosmo_params(
                H0_value, Omega_c_value, Omega_b_value, fphi_value, f_k2_value, f_g2_value,
                w0=w0_value, wa=wa_value, mnu=Mnu_value
            )
        
        if 'z_max' in self.settings['solver']:
            z_max_value = self.settings['solver']['z_max']
        else:
            z_max_value = 1200.
        
        if 'Npoints' in self.settings['solver']:
            Npoints_value = self.settings['solver']['Npoints']
        else:
            Npoints_value = 200
        
        if 'forwards' in self.settings['solver']:
            forwards_value = self.settings['solver']['forwards']
        else:
            forwards_value = True

        if 'HS_correction' in self.settings['solver']:
            HS_correction_value = self.settings['solver']['HS_correction']
        else:
            HS_correction_value = True

        if self.settings['model'] == 'GR':
            self.model.run_solver(
                z_max_value, Npoints_value, forwards=forwards_value, HS_correction=HS_correction_value, method=self.solver['method'], 
                which_root=self.solver['which_root']
            )
        else:
            self.model.run_solver(
                z_max_value, Npoints_value, forwards=forwards_value, HS_correction=HS_correction_value,
                variable1=self.solver['variable1'], phi_ini=phi_ini_value, variable2=self.solver['variable2'], 
                phi_prime_ini=phi_prime_ini_value, method=self.solver['method'], which_root=self.solver['which_root']
            )
    
    # Get derived products

    def _get_derived_product(self, key, root=0):
        """
        Returns a specific derived product. This will convert None outputs to -4043 and True and False to 1 and 0.
        
        Parameters
        ----------
        key : str
            Derived product key.
        root : int
            The solution of the numerical solver to look at.
        """
        None_value = -4043.
        if self.settings['model'] == 'GR':
            value = self.model.output[key]
        else:
            value = self.model.output[key][root]
        if value is None:
            return None_value
        elif value == True:
            return 1.
        elif value == False:
            return 0.
        else:
            return value

    
    def get_derived(self, root=0):
        """
        Returns derived products for a given model.

        Parameters
        ----------
        root : int
            The solution of the numerical solver to look at.
        
        Returns
        -------
        derived : array 
            Derived data product for each model.
        """
        derived = np.array([self._get_derived_product(key, root=root) for key in self.derived_keys])
        return derived
    

    # Planck Compressed Likelihood

    def init_Planck(self):
        """
        Initialises Planck compressed likelihood.
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
    

    def theory_Planck_compressed(self, root=0):
        """
        Compute Planck theoretical products for the compressed likelihood.
        """
        theory = np.zeros(3)
        if self.settings['model'] == 'GR':
            theory[0] = self.model.output['theta_star']
        else:
            theory[0] = self.model.output['theta_star'][root]
        if self.settings['model'] == 'GR':
            theory[1] = self.model.output['Omega_b0']*(1e-2*self.model.output['H0'])**2
        else:
            theory[1] = self.model.output['Omega_b0'][root]*(1e-2*self.model.output['H0'][root])**2
        if self.settings['model'] == 'GR':
            theory[2] = self.model.output['Omega_b0']
            theory[2] += self.model.output['Omega_c0']
            theory[2] *= (1e-2*self.model.output['H0'])**2
        else:
            theory[2] = self.model.output['Omega_b0'][root]
            theory[2] += self.model.output['Omega_c0'][root]
            theory[2] *= (1e-2*self.model.output['H0'][root])**2
        return theory
    

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
        loglike = -0.5*self.log_norm_planck - 0.5*chi2
        return loglike
    
    # H0 observations and likelihoods

    def init_H0(self):
        """
        Initialise H0 constraints.
        """
        # SHOES H0 measurements
        self.SHOES_H0 = 73.17
        self.SHOES_H0_err = 0.86
        # Megamaser Cosmology Project
        self.MCP_H0 = 73.9 
        self.MCP_H0_err = 3.
        # TRGB-SBF
        self.TRGB_H0 = 73.8
        self.TRGB_H0_err = 2.4
        # Type II SN
        self.Type2SN_H0 = 74.9 
        self.Type2SN_H0_err = 2.7
        self.log_norm_H0_SHOES = np.log(2*np.pi*self.SHOES_H0_err**2)
        self.log_norm_H0_MCP = np.log(2*np.pi*self.MCP_H0_err**2)
        self.log_norm_H0_TRGB = np.log(2*np.pi*self.TRGB_H0_err**2)
        self.log_norm_H0_Type2SN = np.log(2*np.pi*self.Type2SN_H0_err**2)


    def theory_H0(self, root=0):
        """
        Returns the H0 value of the model.
        """
        if self.settings['model'] == 'GR':
            return self.model.output['H0']
        else:
            return self.model.output['H0'][root]


    def loglike_H0_LOCAL_SHOES(self, theory):
        """
        Computes the H0 likelihood for SHOES measurements.

        Parameters
        ----------
        theory : array
            Theoretical predictions for H0.

        Returns
        -------
        loglike : float
            H0 log likelihood values.
        """
        chi2 = ((theory - self.SHOES_H0)/self.SHOES_H0_err)**2
        loglike = -0.5*self.log_norm_H0_SHOES  - 0.5*chi2
        return loglike
    

    def loglike_H0_LOCAL_MCP(self, theory):
        """
        Computes the H0 likelihood for SHOES measurements.

        Parameters
        ----------
        theory : array
            Theoretical predictions for H0.

        Returns
        -------
        loglike : float
            H0 log likelihood values.
        """
        chi2 = ((theory - self.MCP_H0)/self.MCP_H0_err)**2
        loglike = -0.5*self.log_norm_H0_MCP  - 0.5*chi2
        return loglike
    

    def loglike_H0_LOCAL_TRGB(self, theory):
        """
        Computes the H0 likelihood for SHOES measurements.

        Parameters
        ----------
        theory : array
            Theoretical predictions for H0.

        Returns
        -------
        loglike : float
            H0 log likelihood values.
        """
        chi2 = ((theory - self.TRGB_H0)/self.TRGB_H0_err)**2
        loglike = -0.5*self.log_norm_H0_TRGB  - 0.5*chi2
        return loglike
    

    def loglike_H0_LOCAL_Type2SN(self, theory):
        """
        Computes the H0 likelihood for SHOES measurements.

        Parameters
        ----------
        theory : array
            Theoretical predictions for H0.

        Returns
        -------
        loglike : float
            H0 log likelihood values.
        """
        chi2 = ((theory - self.Type2SN_H0)/self.Type2SN_H0_err)**2
        loglike = -0.5*self.log_norm_H0_Type2SN  - 0.5*chi2
        return loglike
    

    # DESI BAO likelihoods

    def init_DESI_BAO_DR2(self):
        """
        Initialises DESI BAO DR2 likelihood.
        """
        self.DESI_z_BGS = 0.29500000
        self.DESI_z_LRG1 = 0.51000000
        self.DESI_z_LRG2 = 0.70600000
        self.DESI_z_LRG3_ELG1 = 0.93400000
        self.DESI_z_ELG2 = 1.32100000
        self.DESI_z_QSO = 1.48400000
        self.DESI_z_LyA = 2.33

        self.DESI_type_BGS = 'DV_over_rs'
        self.DESI_type_LRG1 = np.array(['DM_over_rs', 'DH_over_rs'])
        self.DESI_type_LRG2 = np.array(['DM_over_rs', 'DH_over_rs'])
        self.DESI_type_LRG2_ELG1 = np.array(['DM_over_rs', 'DH_over_rs'])
        self.DESI_type_ELG2 = np.array(['DM_over_rs', 'DH_over_rs'])
        self.DESI_type_QSO = np.array(['DM_over_rs', 'DH_over_rs'])
        self.DESI_type_LyA = np.array(['DH_over_rs', 'DM_over_rs'])

        self.mu_DESI_BGS = 7.94167639
        self.cov_DESI_BGS = 5.78998687e-03

        self.inv_cov_DESI_BGS = 1/self.cov_DESI_BGS
        self.log_norm_DESI_BGS = np.log(2 * np.pi * self.cov_DESI_BGS)

        self.mu_DESI_LRG1 = np.array([13.58758434, 21.86294686])
        self.cov_DESI_LRG1 = np.array(
            [[2.83473742e-02, -3.26062007e-02],
             [-3.26062007e-02, 1.83928040e-01]]
        )

        self.inv_cov_DESI_LRG1 = np.linalg.inv(self.cov_DESI_LRG1)
        _, logdet = np.linalg.slogdet(self.cov_DESI_LRG1)
        self.log_norm_DESI_LRG1 = (len(self.mu_DESI_LRG1) * np.log(2 * np.pi) + logdet)

        self.mu_DESI_LRG2 = np.array([17.35069094, 19.45534918])
        self.cov_DESI_LRG2 = np.array(
            [[3.23752442e-02, -2.37445646e-02],
             [-2.37445646e-02, 1.11469198e-01]]
        )

        self.inv_cov_DESI_LRG2 = np.linalg.inv(self.cov_DESI_LRG2)
        _, logdet = np.linalg.slogdet(self.cov_DESI_LRG2)
        self.log_norm_DESI_LRG2 = (len(self.mu_DESI_LRG2) * np.log(2 * np.pi) + logdet)

        self.mu_DESI_LRG3_ELG1 = np.array([21.57563956, 17.64149464])
        self.cov_DESI_LRG3_ELG1 = np.array(
            [[2.61732816e-02, -1.12938006e-02],
             [-1.12938006e-02, 4.04183878e-02]]
        )

        self.inv_cov_DESI_LRG3_ELG1 = np.linalg.inv(self.cov_DESI_LRG3_ELG1)
        _, logdet = np.linalg.slogdet(self.cov_DESI_LRG3_ELG1)
        self.log_norm_DESI_LRG3_ELG1 = (len(self.mu_DESI_LRG3_ELG1) * np.log(2 * np.pi) + logdet)

        self.mu_DESI_ELG2 = np.array([27.60085612, 14.17602155])
        self.cov_DESI_ELG2 = np.array(
            [[1.05336516e-01, -2.90308418e-02],
             [-2.90308418e-02, 5.04233092e-02]]
        )

        self.inv_cov_DESI_ELG2 = np.linalg.inv(self.cov_DESI_ELG2)
        _, logdet = np.linalg.slogdet(self.cov_DESI_ELG2)
        self.log_norm_DESI_ELG2 = (len(self.mu_DESI_ELG2) * np.log(2 * np.pi) + logdet)

        self.mu_DESI_QSO = np.array([30.51190063, 12.81699964])
        self.cov_DESI_QSO = np.array(
            [[5.83020277e-01, -1.95215562e-01],
             [-1.95215562e-01, 2.68336193e-01]]
        )

        self.inv_cov_DESI_QSO = np.linalg.inv(self.cov_DESI_QSO)
        _, logdet = np.linalg.slogdet(self.cov_DESI_QSO)
        self.log_norm_DESI_QSO = (len(self.mu_DESI_QSO) * np.log(2 * np.pi) + logdet)

        self.mu_DESI_LyA = np.array([8.631545674846294, 38.988973961958784])
        self.cov_DESI_LyA = np.array(
            [[1.02136194e-02, -2.31395216e-02],
             [-2.31395216e-02, 2.82685779e-01]]
        )

        self.inv_cov_DESI_LyA = np.linalg.inv(self.cov_DESI_LyA)
        _, logdet = np.linalg.slogdet(self.cov_DESI_LyA)
        self.log_norm_DESI_LyA = (len(self.mu_DESI_LyA) * np.log(2 * np.pi) + logdet)
    

    def prep4BAO(self, root=0):
        """
        Constructs interpolation functions.
        """
        if self.settings['model'] == 'GR':
            DH = self.model.const['c[km/s]']/self.model.output['H']
            DM = self.model.output['Dc']/(1e-2*self.model.output['H0'])
            DV = (self.model.output['z']*DH*DM**2)**(1/3)
        else:
            DH = self.model.const['c[km/s]']/self.model.output['H'][root]
            DM = self.model.output['Dc'][root]/(1e-2*self.model.output['H0'][root])
            DV = (self.model.output['z']*DH*DM**2)**(1/3)
        self.interp['DH_vs_x'] = interp1d(self.model.output['x'], DH) 
        self.interp['DM_vs_x'] = interp1d(self.model.output['x'], DM)
        self.interp['DV_vs_x'] = interp1d(self.model.output['x'], DV)

    
    def theory_DESI_BAO_DR2_BGS(self, root=0):
        """
        Compute theoretical BAO measurements for BGS.
        """
        if self.settings['model'] == 'GR':
            theory = self.interp['DV_vs_x'](redshift.z2x(self.DESI_z_BGS))/self.model.output['r_drag']
        else:
            theory = self.interp['DV_vs_x'](redshift.z2x(self.DESI_z_BGS))/self.model.output['r_drag'][root]
        return theory
    

    def theory_DESI_BAO_DR2_LRG1(self, root=0):
        """
        Compute theoretical BAO measurements for LRG1.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG1))/self.model.output['r_drag']
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG1))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG1))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG1))/self.model.output['r_drag'][root]
        return theory
    

    def theory_DESI_BAO_DR2_LRG2(self, root=0):
        """
        Compute theoretical BAO measurements for LRG2.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG2))/self.model.output['r_drag']
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG2))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG2))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG2))/self.model.output['r_drag'][root]
        return theory

    
    def theory_DESI_BAO_DR2_LRG3_ELG1(self, root=0):
        """
        Compute theoretical BAO measurements for LRG3 and ELG1.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG3_ELG1))/self.model.output['r_drag']
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG3_ELG1))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LRG3_ELG1))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LRG3_ELG1))/self.model.output['r_drag'][root]
        return theory
    

    def theory_DESI_BAO_DR2_ELG2(self, root=0):
        """
        Compute theoretical BAO measurements for LRG3 and ELG1.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_ELG2))/self.model.output['r_drag']
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_ELG2))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_ELG2))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_ELG2))/self.model.output['r_drag'][root]
        return theory
    

    def theory_DESI_BAO_DR2_QSO(self, root=0):
        """
        Compute theoretical BAO measurements for QSO.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_QSO))/self.model.output['r_drag']
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_QSO))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_QSO))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_QSO))/self.model.output['r_drag'][root]
        return theory


    def theory_DESI_BAO_DR2_LyA(self, root=0):
        """
        Compute theoretical BAO measurements for LyA.
        """
        theory = np.zeros(2)
        if self.settings['model'] == 'GR':
            theory[0] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LyA))/self.model.output['r_drag']
            theory[1] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LyA))/self.model.output['r_drag']
        else:
            theory[0] = self.interp['DH_vs_x'](redshift.z2x(self.DESI_z_LyA))/self.model.output['r_drag'][root]
            theory[1] = self.interp['DM_vs_x'](redshift.z2x(self.DESI_z_LyA))/self.model.output['r_drag'][root]
        return theory
    

    def loglike_DESI_BAO_DR2_BGS(self, theory):
        """
        Computes the DESI BAO DR2 BGS likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.

        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_BGS
        chi2 = float(delta**2 / self.inv_cov_DESI_BGS)
        loglike = -0.5*self.log_norm_DESI_BGS  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_LRG1(self, theory):
        """
        Computes the DESI BAO DR2 LRG1 likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_LRG1
        chi2 = float(delta @ self.inv_cov_DESI_LRG1 @ delta)
        loglike = -0.5*self.log_norm_DESI_LRG1  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_LRG2(self, theory):
        """
        Computes the DESI BAO DR2 LRG2 likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_LRG2
        chi2 = float(delta @ self.inv_cov_DESI_LRG2 @ delta)
        loglike = -0.5*self.log_norm_DESI_LRG2  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_LRG3_ELG1(self, theory):
        """
        Computes the DESI BAO DR2 LRG ELG likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_LRG3_ELG1
        chi2 = float(delta @ self.inv_cov_DESI_LRG3_ELG1 @ delta)
        loglike = -0.5*self.log_norm_DESI_LRG3_ELG1  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_ELG2(self, theory):
        """
        Computes the DESI BAO DR2 ELG likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_ELG2
        chi2 = float(delta @ self.inv_cov_DESI_ELG2 @ delta)
        loglike = -0.5*self.log_norm_DESI_ELG2  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_QSO(self, theory):
        """
        Computes the DESI BAO DR2 QSO likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_QSO
        chi2 = float(delta @ self.inv_cov_DESI_QSO @ delta)
        loglike = -0.5*self.log_norm_DESI_QSO  - 0.5*chi2
        return loglike
    

    def loglike_DESI_BAO_DR2_LyA(self, theory):
        """
        Computes the DESI BAO DR2 LyA likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for DESI BAO.
        
        Returns
        -------
        loglike : float
            BAO log likelihood values.
        """
        delta = theory - self.mu_DESI_LyA
        chi2 = float(delta @ self.inv_cov_DESI_LyA @ delta)
        loglike = -0.5*self.log_norm_DESI_LyA - 0.5*chi2
        return loglike
    

    def init_SN4PantheonPlus(self):
        pass
    

    def init_SN4Union3(self):
        pass


    def init_SN4DES_Dovekie(self):
        """
        Initialise the data for DES SN Dovekie data set.
        """

        import csv

        try:
            from importlib.resources import files  # Python ≥ 3.9
        except ImportError:
            from importlib_resources import files  # Python 3.8
        
        csv_path = files("HiCOLA.Frontend.obs.DES_Dovekie") / "DES-Dovekie_HD.csv"
        
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data_dict = [row for row in reader]
        
        zHD = np.array([float(row['zHD']) for row in data_dict])
        cond = np.where(zHD > 0.00)
        zCMB = np.array([float(row['zHD']) for row in data_dict])[cond]
        zHEL = np.array([float(row['zHEL']) for row in data_dict])[cond]
        mu = np.array([float(row['MU']) for row in data_dict])[cond]

        self.z_CMB_DES_SN_Dovekie = zCMB
        self.z_HEL_DES_SN_Dovekie = zHEL
        self.mu_DES_SN_Dovekie = mu

        npz_path = files("HiCOLA.Frontend.obs.DES_Dovekie") / "STAT+SYS.npz"

        data = np.load(npz_path)
        n = data[data.files[0]][0]
        invcov = np.zeros((n, n))
        invcov[np.triu_indices(n)] = data[data.files[1]]

        # Reflect to lower triangular part to make it symmetric
        i_lower = np.tril_indices(n, -1)
        invcov[i_lower] = invcov.T[i_lower]

        self.inv_cov_DES_SN_Dovekie = invcov

        self.cov_DES_SN_Dovekie = np.linalg.inv(self.inv_cov_DES_SN_Dovekie)

        _, logdet = np.linalg.slogdet(self.cov_DES_SN_Dovekie)
        self.log_norm_DES_SN_Dovekie = (len(self.mu_DES_SN_Dovekie) * np.log(2 * np.pi) + logdet)
        
        self.C_DES_SN_Dovekie = np.sum(self.inv_cov_DES_SN_Dovekie)


    def prep4SN(self, root=0):
        """
        Constructs interpolation functions.

        Parameters
        ----------
        root : int, optional
            The solution of the numerical solver to look at.
        """
        if self.settings['model'] == 'GR':
            DA = self.model.output['Dc']/((1+self.model.output['z'])*1e-2*self.model.output['H0'])
        else:
            DA = self.model.output['Dc'][root]/((1+self.model.output['z'])*1e-2*self.model.output['H0'][root])
        self.interp['DA_vs_x'] = interp1d(self.model.output['x'], DA)


    def theory_SN4PantheonPlus(self):
        pass
    

    def theory_SN4Union3(self):
        pass


    def theory_SN4DES_Dovekie(self):
        """
        Computes the theoretical predictions for the distance modulus for DES SN Dovekie.

        Returns
        -------
        theory : array
            Theoretical predictions for the distance modulus for DES SN Dovekie.
        """
        if self.settings['model'] == 'GR':
            DL = self.interp['DA_vs_x'](redshift.z2x(self.z_CMB_DES_SN_Dovekie))
            DL *= (1 + self.z_CMB_DES_SN_Dovekie)*(1 + self.z_HEL_DES_SN_Dovekie)
            mu_theory = 5.*np.log10(DL)
        else:
            DL = self.interp['DA_vs_x'](redshift.z2x(self.z_CMB_DES_SN_Dovekie))
            DL *= (1 + self.z_CMB_DES_SN_Dovekie)*(1 + self.z_HEL_DES_SN_Dovekie)
            mu_theory = 5.*np.log10(DL)
        mu_theory += 25.
        return mu_theory
    

    def loglike_SN4PantheonPlus(self):
        pass


    def loglike_SN4Union3(self):
        pass
    

    def loglike_SN4DES_Dovekie(self, theory):
        """
        Computes the DES SN Dovekie likelihood.

        Parameters
        ----------
        theory : array
            Theoretical predictions for the distance modulus for DES SN Dovekie.
        
        Returns
        -------
        loglike : float
            DES SN Dovekie log likelihood values.
        """

        delta = theory - self.mu_DES_SN_Dovekie
        chit2 = delta @ self.inv_cov_DES_SN_Dovekie @ delta.T
        B = np.sum(delta @ self.inv_cov_DES_SN_Dovekie)
        chi2 = chit2 - (B**2 / self.C_DES_SN_Dovekie)
        loglike = -0.5*self.log_norm_DES_SN_Dovekie - 0.5*chi2
        return loglike


    def initialise(self):
        """
        Initialise observations and likelihood quantities.
        """
        if 'Planck' in self.constraints:
            self.init_Planck()
        if 'H0' in self.constraints:
            self.init_H0()
        DESI_constraints = [
            'DESI_DR2_BAO_FULL', 
            'DESI_DR2_BAO_BGS',
            'DESI_DR2_BAO_LRG1',
            'DESI_DR2_BAO_LRG2',
            'DESI_DR2_BAO_LRG_ELG', 
            'DESI_DR2_BAO_ELG', 
            'DESI_DR2_BAO_QSO', 
            'DESI_DR2_BAO_LyA', 
        ]
        if any(item in self.constraints for item in DESI_constraints):
            self.init_DESI_BAO_DR2()
        if 'DES_SN_Dovekie' in self.constraints:
            self.init_SN4DES_Dovekie()
    

    def ptform(self, u):
        """
        Transform a uniform random to the prior range of sampled variables, for use with dynesty.

        Parameters
        ----------
        u : array
            Random variables.
        """
        param_values = (self.prior_max-self.prior_min)*u + self.prior_min 
        return param_values
    

    def log_prior(self, param_values):
        """
        Computes the log prior, for use with emcee.

        Parameters
        ----------
        param_values : array
            Parameter values.
        """
        if np.all((param_values >= self.prior_min) & (param_values <= self.prior_max)):
            return 0.
        else:
            return -np.inf
    

    def _loglike(self, param_values):
        """
        Internal loglikelihood function.

        Parameters
        ----------
        param_values : array
            Parameter value array.
        
        Returns
        -------
        loglike : float
            Likelihood output.
        """

        if self.sampler_method == 'emcee':
            loglike = self.log_prior(param_values)
        else:
            loglike = 0.
        
        if not np.isfinite(loglike):
            return -np.inf, np.ones(len(self.derived_keys))
        else:
            
            if self.debug:
                self.run_model(param_values)
            else:
                # suppresses divide by zero errors sometimes seen in some of the lambdified functions.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.run_model(param_values)

            blob = self.get_derived(root=self.root)
            
            if any('DESI' in c for c in self.constraints):
                self.prep4BAO()
            
            if 'DES_SN_Dovekie' in self.constraints:
                self.prep4SN(root=self.root)

            if self.likelihood_switch['Planck']:
                theory = self.theory_Planck_compressed()
                loglike += self.loglike_Planck(theory)

            if self.likelihood_switch['H0_LOCAL_ALL'] or self.likelihood_switch['H0_LOCAL_SHOES']:
                theory = self.theory_H0()
                loglike += self.loglike_H0_LOCAL_SHOES(theory)
                
            if self.likelihood_switch['H0_LOCAL_ALL'] or self.likelihood_switch['H0_LOCAL_MCP']:
                theory = self.theory_H0()
                loglike += self.loglike_H0_LOCAL_MCP(theory)

            if self.likelihood_switch['H0_LOCAL_ALL'] or self.likelihood_switch['H0_LOCAL_TRGB']:
                theory = self.theory_H0()
                loglike += self.loglike_H0_LOCAL_TRGB(theory)

            if self.likelihood_switch['H0_LOCAL_ALL'] or self.likelihood_switch['H0_LOCAL_Type2SN']:
                theory = self.theory_H0()
                loglike += self.loglike_H0_LOCAL_Type2SN(theory)
            
            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_BGS']:
                theory = self.theory_DESI_BAO_DR2_BGS(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_BGS(theory)
            
            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_LRG1']:
                theory = self.theory_DESI_BAO_DR2_LRG1(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_LRG1(theory)

            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_LRG2']:
                theory = self.theory_DESI_BAO_DR2_LRG2(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_LRG2(theory)

            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_LRG3_ELG1']:
                theory = self.theory_DESI_BAO_DR2_LRG3_ELG1(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_LRG3_ELG1(theory)

            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_ELG2']:
                theory = self.theory_DESI_BAO_DR2_ELG2(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_ELG2(theory)
            
            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_QSO']:
                theory = self.theory_DESI_BAO_DR2_QSO(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_QSO(theory)
            
            if self.likelihood_switch['DESI_DR2_BAO_FULL'] or self.likelihood_switch['DESI_DR2_BAO_LyA']:
                theory = self.theory_DESI_BAO_DR2_LyA(root=self.root)
                loglike += self.loglike_DESI_BAO_DR2_LyA(theory)
            
            if self.likelihood_switch['DES_SN_Dovekie']:
                theory = self.theory_SN4DES_Dovekie()
                loglike += self.loglike_SN4DES_Dovekie(theory)
            
            return loglike, blob
    

    def loglike(self, param_values):
        """
        Log-likelihood function for various parameters.

        Parameters
        ----------
        param_values : array
            Parameter value array.
        """

        if self.settings['model'] != 'GR':
            if self.model._lambdified == False:
                self.model._lambdify_symbolic()
        
        if self.debug:
            loglike, blob = self._loglike(param_values)
        else:
            try:
                loglike, blob = self._loglike(param_values)
            except Exception:
                if self.debug:
                    print('Exception reached in model computation...')
                blob = np.zeros(len(self.derived_keys))
                loglike = -np.inf
        
        if self.sampler_method == 'dynesty':
            if not np.isfinite(loglike):
                loglike = -1e300  # huge negative log-likelihood
        
        if self.derived:
            return loglike, blob
        else:
            return loglike
        

    def time_loglike(self, size, debug=False, root=0, derived=True):
        """
        Computes the time per loglikelihood evaluation.

        Parameters
        ----------
        size : int
            Number of samples to average loglikelihood evaluations.
        derived : bool, optional
            Sets whether derived data products should be included.
        root : int, optional
            The solution of the numerical solver to look at.
        debug : bool, optional
            Runs in debug mode to enable better diagnostics of errors.
        """

        import time

        self.root = root
        self.debug = debug
        self.derived = derived

        u = np.random.random_sample((size+1, self.Nvaried))

        param_values = self.ptform(u[0])
        self.loglike(param_values)

        t1 = time.time()
        [self.loglike(self.ptform(u[i])) for i in range(1, len(u))]
        t2 = time.time()

        print('Time per likelihood call:', (t2-t1)/size)


    def set_dynesty_settings(self, nlive=100, sample='rslice', slices=5, walks=20, update_interval=0.5, dlogz=0.5, checkpoint_interval=60):
        """
        Defined dynesty settings.
        
        Parameters
        ----------
        nlive : int
            Live points in the sampler.
        sample : str
            Sampling type.
        slices : int
            If using a slicing methods, this defines the number of slices to use.
        walks : int
            The number of walks to use if the sampling method is 'rwalks'.
        update_interval : float
            The update interval used for parallel processed jobs, if the code is not parallelised then this is ignored.
        dlogz : float
            The stopping criteria for nested sampling.
        checkpoint_interval : int
            The time in seconds between checkpoint saves.
        """
        self.dynesty_settings = {
            'nlive': nlive,
            'update_interval': update_interval,
            'dlogz': dlogz,
            'checkpoint_interval': checkpoint_interval
        }

        self.dynesty_settings['sample'] = sample

        if sample == 'rwalks':
            self.dynesty_settings['walks'] = walks
            self.dynesty_settings['slices'] = None
        else:
            self.dynesty_settings['slices'] = slices
            self.dynesty_settings['walks'] = None


    def set_emcee_settings(self, walkers=62, burnin=100, steps=1000):
        """
        Defined emcee settings.
        
        Parameters
        ----------
        nwalkers : int
            Number of walkers in the ensemble.
        burnin : int
            The number of burn in steps.
        steps: int
            Number of steps for the walkers to move. 
        """
        self.emcee_settings = {
            'walkers': walkers,
            'burnin': burnin,
            'steps': steps
        }

    
    def set_pocomc_settings(self, n_effective=512, n_active=512, dynamic=True, checkpoint_iter=4):
        """
        Defined emcee settings.
        
        Parameters
        ----------
        nwalkers : int, optional
            Number of walkers in the ensemble.
        burnin : int, optional
            The number of burn in steps.
        steps: int, optional
            Number of steps for the walkers to move. 
        checkpoint_iter : int, optional
        """
        self.pocomc_settings = {
            'n_effective': n_effective,
            'n_active': n_active,
            'dynamic': dynamic,
            'checkpoint_iter': checkpoint_iter
        }


    def run_mcmc(
            self, processes=1, derived=True, checkpoint=True, whichcheckpoint=0, resume=False, root=0, debug=False
        ):
        """
        Runs the mcmc or sampling method to sample the parameter space.
        
        Parameters
        ----------
        processes : int, optional
            The number of processes to use for the 
        derived : bool, optional
            Sets whether derived data products should be included.
        checkpoint : bool, optional
            Tells the code to save checkpoint outputs incase of crashes and allows resume already started evaluations.
        whichcheckpoint : bool, optional
            Tells the code which checkpoint output to resume from, if there are many.
        resume : bool, optional
            Resume from an already started checkpoint file.
        root : int, optional
            The solution of the numerical solver to look at.
        debug : bool, optional
            Runs in debug mode to enable better diagnostics of errors.
        """

        self.root = root
        self.debug = debug
        self.derived = derived

        self.initialise()
        
        if self.sampler_method == 'dynesty':

            from dynesty import NestedSampler

            if checkpoint:
                if resume == False:
                    if Path(self.fname + '.save').exists():
                        fname_idx = 1
                        while Path(self.fname + '_%i.save' % fname_idx).exists():
                            fname_idx += 1
                        checkpoint_fname = self.fname + '_%i.save' % fname_idx
                    else:
                        checkpoint_fname = self.fname + '.save'
                else:
                    if whichcheckpoint == 0:
                        assert Path(self.fname + '.save').exists(), 'Checkpoint file %s.save does not exist.' % self.fname
                        checkpoint_fname = self.fname + '.save'
                    else:
                        fname_idx = whichcheckpoint
                        assert Path(self.fname + '_%i.save' % fname_idx).exists(), 'Checkpoint file %s.save does not exist.' % self.fname
                        checkpoint_fname = self.fname + '_%i.save' % fname_idx
            else:
                checkpoint_fname = None
            
            if processes == 1:
                if resume:
                    sampler = NestedSampler.restore(checkpoint_fname)
                    sampler.run_nested(resume=True, dlogz=self.dynesty_settings['dlogz'], checkpoint_file=checkpoint_fname, checkpoint_every=self.dynesty_settings['checkpoint_interval'])
                else:
                    sampler = NestedSampler(
                        self.loglike, self.ptform, self.Nvaried,
                        nlive=self.dynesty_settings['nlive'],
                        sample=self.dynesty_settings['sample'],
                        slices=self.dynesty_settings['slices'],
                        walks=self.dynesty_settings['walks'],
                        blob=self.derived
                    )
                    sampler.run_nested(dlogz=self.dynesty_settings['dlogz'], checkpoint_file=checkpoint_fname, checkpoint_every=self.dynesty_settings['checkpoint_interval'])

            else:
                
                if self.settings['model'] != 'GR':
                    if self.model._lambdified:
                        self.model._delambdify()

                from dynesty.pool import Pool
            
                with Pool(processes, self.loglike, self.ptform) as pool:
                    if resume:
                        sampler = NestedSampler.restore(checkpoint_fname)
                        sampler.run_nested(resume=True, dlogz=self.dynesty_settings['dlogz'], checkpoint_file=checkpoint_fname, checkpoint_every=self.dynesty_settings['checkpoint_interval'])
                    else:
                        sampler = NestedSampler(
                            pool.loglike, pool.prior_transform, self.Nvaried, pool=pool,
                            nlive=self.dynesty_settings['nlive'],
                            sample=self.dynesty_settings['sample'],
                            slices=self.dynesty_settings['slices'],
                            walks=self.dynesty_settings['walks'],
                            blob=self.derived,
                            update_interval=self.dynesty_settings['update_interval']
                        )
                        sampler.run_nested(dlogz=self.dynesty_settings['dlogz'], checkpoint_file=checkpoint_fname, checkpoint_every=self.dynesty_settings['checkpoint_interval'])

            self.sampler = sampler

            self.samples = self.sampler.results.samples
            self.weights = self.sampler.results.importance_weights()
            if self.derived:
                self.blob = self.sampler.results.blob

        elif self.sampler_method == 'emcee':

            import emcee

            if checkpoint:
                if resume == False:
                    if Path(self.fname + '.h5').exists():
                        fname_idx = 1
                        while Path(self.fname + '_%i.h5' % fname_idx).exists():
                            fname_idx += 1
                        checkpoint_fname = self.fname + '_%i.h5' % fname_idx
                    else:
                        checkpoint_fname = self.fname + '.h5'
                else:
                    if whichcheckpoint == 0:
                        assert Path(self.fname + '.h5').exists(), 'Checkpoint file %s.h5 does not exist.'
                        checkpoint_fname = self.fname + '.h5'
                    else:
                        fname_idx = whichcheckpoint
                        assert Path(self.fname + '_%i.h5' % fname_idx).exists(), 'Checkpoint file %s.h5 does not exist.'    
                        checkpoint_fname = self.fname + '_%i.h5' % fname_idx
            else:
                checkpoint_fname = None

            if checkpoint_fname is not None:
                backend = emcee.backends.HDFBackend(checkpoint_fname)
                if resume == False:
                    backend.reset(self.emcee_settings['walkers'], self.Nvaried)
            else:
                backend = None

            if processes == 1:

                sampler = emcee.EnsembleSampler(self.emcee_settings['walkers'], self.Nvaried, self.loglike, backend=backend)

                if resume:
                    pos_ini = None
                else:
                    pos_ini = self.init_value  + 1e-4 * np.random.randn(self.emcee_settings['walkers'], self.Nvaried)

                sampler.run_mcmc(pos_ini, self.emcee_settings['steps'], progress=True)
                
            else:

                if self.settings['model'] != 'GR':
                    if self.model._lambdified:
                        self.model._delambdify()
                
                import multiprocessing

                with multiprocessing.Pool(processes=processes) as pool:

                    sampler = emcee.EnsembleSampler(
                        self.emcee_settings['walkers'],
                        self.Nvaried,
                        self.loglike,
                        pool=pool,
                        backend=backend
                    )

                    if resume:
                        pos_ini = None
                    else:
                        pos_ini = self.init_value + 1e-4 * np.random.randn(self.emcee_settings['walkers'], self.Nvaried)

                    sampler.run_mcmc(pos_ini, self.emcee_settings['steps'], progress=True)

            self.sampler = sampler

            self.samples = self.sampler.get_chain(flat=True, discard=self.emcee_settings['burnin'])
            self.weights = np.ones(len(self.samples))
            if self.derived:
                self.blob = self.sampler.get_blobs(flat=True, discard=self.emcee_settings['burnin'])

        elif self.sampler_method == 'pocoMC':
            
            from scipy.stats import uniform

            import pocomc as pc

            prior_list = [uniform(loc=self.prior_min[i], scale=self.prior_max[i]-self.prior_min[i]) for i in range(0, self.Nvaried)]

            prior = pc.Prior(prior_list)

            if resume:
                assert Path('pocoMC_states/'+self.fname + '_%i.state' % whichcheckpoint).exists(), "File pocoMC_states/%s_%i.state does not exist." % (self.fname, whichcheckpoint)
                checkpoint_fname = 'pocoMC_states/' + self.fname + '_%i.state' % whichcheckpoint
            else:
                checkpoint_fname = None
            
            if checkpoint:
                save_every = self.pocomc_settings['checkpoint_iter']
            else:
                save_every = None

            if processes == 1:

                sampler = pc.Sampler(
                    prior=prior,
                    likelihood=self.loglike,
                    output_dir='pocoMC_states',
                    output_label=self.fname
                )

                sampler.run(resume_state_path=checkpoint_fname, save_every=save_every)

            else:

                if self.settings['model'] != 'GR':
                    if self.model._lambdified:
                        self.model._delambdify()
                
                sampler = pc.Sampler(
                    prior=prior,
                    likelihood=self.loglike,
                    pool=processes,
                    output_dir='pocoMC_states',
                    output_label=self.fname
                )

                sampler.run(resume_state_path=checkpoint_fname, save_every=save_every)
            
            self.sampler = sampler

            if self.derived:
                self.samples, self.weights, _, _, self.blob = self.sampler.posterior(return_blobs=True)
            else:
                self.samples, self.weights, _, _ = self.sampler.posterior()
    

    def get_MLE(self, root=0, debug=False, derived=False):
        """
        Runs iminuit to find the maximum likelihood parameter values.

        Parameters
        ----------
        root : int, optional
            The solution of the numerical solver to look at.
        debug : bool, optional
            Runs in debug mode to enable better diagnostics of errors.
        derived : bool, optional
            Sets whether derived data products should be included.

        Returns
        -------
        samples_MLE : array
            Best fit parameter values.
        samples_MLE_errors : array
            Best fit parameter value errors.
        blob_MLE : array
            Best fit parameter derived products.
        """
        
        self.root = root
        self.debug = debug
        self.derived = False

        from iminuit import Minuit

        param_names = [f"p{i}" for i in range(self.Nvaried)]

        def nll_wrapped(*params):
            return -self.loglike(np.array(params))
        
        nll_wrapped._parameters = {
            name: None for name in param_names
        }

        m = Minuit(nll_wrapped, *self.init_value)
        m.errordef = 0.5
        m.limits = list(zip(self.prior_min, self.prior_max))

        m.migrad()
        m.minos()

        self.samples_MLE = np.array(m.values)
        self.samples_MLE_errors = np.array(m.errors)

        self.derived = derived

        if self.derived == True:
            _, self.blob_MLE = self.loglike(self.samples_MLE)
            return self.samples_MLE, self.samples_MLE_errors, self.blob_MLE
        else:
            return self.samples_MLE, self.samples_MLE_errors
        
    
    def get_proflike(self, param, size=10, bounds=None, root=0, debug=False, derived=False):
        """
        Returns the profile likelihood for a given parameter.

        Parameters
        ----------
        root : int, optional
            The solution of the numerical solver to look at.
        debug : bool, optional
            Runs in debug mode to enable better diagnostics of errors.
        derived : bool, optional
            Sets whether derived data products should be included.
        """

        self.root = root
        self.debug = debug
        self.derived = False

        from iminuit import Minuit

        param_names = [f"p{i}" for i in range(self.Nvaried)]

        def nll_wrapped(*params):
            return -self.loglike(np.array(params))
        
        nll_wrapped._parameters = {
            name: None for name in param_names
        }

        m = Minuit(nll_wrapped, *self.init_value)
        m.errordef = 0.5
        m.limits = list(zip(self.prior_min, self.prior_max))

        m.migrad()

        profile_index = self.varied_param2idx[param]

        if bounds is None:
            param_grid = np.linspace(self.prior_min[profile_index], self.prior_max[profile_index], size)
        else:
            param_grid = np.linspace(bounds[0], bounds[1], size)

        profile = m.mnprofile(profile_index, grid=param_grid)
        
        self.proflike_param = param
        self.proflike_x = profile[0]
        self.proflike_NLL = profile[1]

        return self.proflike_x, self.proflike_NLL
    

    def _get_param_info4chains(self):
        """
        Returns parameter names, labels and ranges.

        Return
        ------
        param_names : list
            List of parameter ranges.
        param_labels : list
            List of parameter labels
        param_ranges : list
            List parameter ranges and prior.
        """
        param_names = [self.varied_idx2param[i] for i in range(self.Nvaried)]
        param_labels = [self.settings[self.varied_idx2param[i]]['label'] for i in range(self.Nvaried)]
        param_ranges = []
        for param in param_names:
            param_ranges.append(self.varied_params[param]['prior'])
        return param_names, param_labels, param_ranges
    
    
    def save_chains(self):
        """
        Save chains to a file.
        """
        param_names, param_labels, param_ranges = self._get_param_info4chains()
        if self.derived:
            np.savez(
                self.fname + '_chains.npz', 
                samples=self.samples, weights=self.weights, 
                param_names=param_names, param_labels=param_labels, param_ranges=param_ranges,
                blob=self.blob, derived_keys=self.derived_keys, derived_labels=self.derived_labels
            )
        else:
            np.savez(
                self.fname + '_chains.npz', 
                samples=self.samples, weights=self.weights, 
                param_names=param_names, param_labels=param_labels, param_ranges=param_ranges
            )
    

    def save_MLE(self):
        """
        Save best fit values to a file.
        """
        param_names, param_labels, param_ranges = self._get_param_info4chains()
        if self.derived:
            np.savez(
                self.fname + '_MLE.npz', 
                samples_MLE=self.samples_MLE, samples_MLE_errors=self.samples_MLE_errors,
                param_names=param_names, param_labels=param_labels, param_ranges=param_ranges,
                blob_MLE=self.blob_MLE, derived_keys=self.derived_keys, derived_labels=self.derived_labels
            )
        else:
            np.savez(
                self.fname + '_MLE.npz', 
                samples_MLE=self.samples_MLE, samples_MLE_errors=self.samples_MLE_errors,
                param_names=param_names, param_labels=param_labels, param_ranges=param_ranges
            )
    

    def save_NLL(self):
        """
        Save best fit values to a file.
        """
        param_names, param_labels, param_ranges = self._get_param_info4chains()
        np.savez(
            self.fname + '_%s_NLL.npz' % self.proflike_param, 
            proflike_x=self.proflike_x, proflike_NLL=self.proflike_NLL,
            param_names=param_names, param_labels=param_labels, param_ranges=param_ranges
        )

    
    def add2dict(self, sample_dict, param, chain, label, ranges=None):
        """
        Add parameter chains to the dictionary.
        
        Parameters
        ----------
        sample_dict : list
            Sample dictionary with parameter chains, labels and priors.
        param : str
            Parameter name.
        chain : array
            Parameter chain.
        label : str
            Parameter label.
        ranges : list, optional
            Minimum and maximum ranges for the parameter being added, can be set to None.
        """
        assert len(sample_dict['weights']) == len(chain), 'New chain must match dimension of original weights.'
        assert param not in sample_dict.keys(), 'New parameter already exists in sample dictionary.'
        _dict = {}
        _dict['chains'] = chain
        _dict['label'] = label
        _dict['prior'] = ranges
        sample_dict[param] = _dict
        return sample_dict


    def sample2dict(self, fname=None, derived=True, derived_limits=[['fphi0', [0., 1.]]]):
        """
        Construct dictionary for samples.

        Parameters
        ----------
        fname : str
            To load samples from a file.
        derived : bool
            Whether to include derived data products.
        derived_limits : list
            List of limits for derived parameters, only specify those with boundaries.
        
        Returns
        -------
        sample_dict : dict
            Sample dictionary with parameter chains, labels and priors.
        """

        if fname is None:
            samples, weights = self.samples, self.weights
            param_names, param_labels, param_ranges = self._get_param_info4chains()
        else:
            data = np.load(fname + '_chains.npz')
            samples = data['samples']
            weights = data['weights']
            param_names = list(data['param_names'])
            param_labels = list(data['param_labels'])
            param_ranges = list(data['param_ranges'])
        
        ranges_dict = {}
        for (i, param) in enumerate(param_names):
            ranges_dict[param] = param_ranges[i]

        if derived:
            
            for i in range(0, len(derived_limits)):
                ranges_dict[derived_limits[i][0]] = derived_limits[i][1]
            
            if fname is None:
                blob = self.blob
                samples = np.column_stack([samples, blob])
                param_names += self.derived_keys
                param_labels += self.derived_labels

                for param in list(self.derived_keys):
                    if param in ranges_dict.keys():
                        param_ranges.append(ranges_dict[param])
                    else:
                        param_ranges.append(None)

            else:
                data = np.load(fname + '_chains.npz')
                blob = data['blob']
                samples = np.column_stack([samples, blob])
                param_names += list(data['derived_keys'])
                param_labels += list(data['derived_labels'])

                for param in list(data['derived_keys']):
                    if param in ranges_dict.keys():
                        param_ranges.append(ranges_dict[param])
                    else:
                        param_ranges.append(None)

        sample_dict = {}
        sample_dict['weights'] = weights
        for (i, param) in enumerate(param_names):
            sample_dict = self.add2dict(sample_dict, param, samples[:,i], param_labels[i], param_ranges[i])
        
        return sample_dict
    

    def MLE2dict(self, fname=None, derived=True):
        """
        Construct dictionary for samples.

        Parameters
        ----------
        fname : str
            To load samples from a file.
        derived : bool
            Whether to include derived data products.
        
        Returns
        -------
        MLE_dict : dict
            MLE dictionary containing MLE best fit values and errors.
        """

        if fname is None:
            samples_MLE = self.samples_MLE
            samples_MLE_errors = self.samples_MLE_errors
            param_names, param_labels, param_ranges = self._get_param_info4chains()
        else:
            data = np.load(fname + '_MLE.npz')
            samples_MLE = data['samples_MLE']
            samples_MLE_errors = data['samples_MLE_errors']
            param_names = list(data['param_names'])
            param_labels = list(data['param_labels'])

        if derived:
            
            if fname is None:
                blob_MLE = self.blob_MLE
                samples_MLE = np.concatenate([samples_MLE, blob_MLE])
                samples_MLE_errors = np.concatenate([samples_MLE_errors, -np.ones_like(blob_MLE)])
                param_names += self.derived_keys
                param_labels += self.derived_labels
            else:
                data = np.load(fname + '_MLE.npz')
                blob_MLE = data['blob_MLE']
                samples_MLE = np.concatenate([samples_MLE, blob_MLE])
                samples_MLE_errors = np.concatenate([samples_MLE_errors, -np.ones_like(blob_MLE)])
                param_names += list(data['derived_keys'])
                param_labels += list(data['derived_labels'])

        MLE_dict = {}
        for (i, param) in enumerate(param_names):
            MLE_dict[param] = {'MLE': samples_MLE[i], 'MLE_err': samples_MLE_errors[i]}
        
        return MLE_dict
    

    def sample2MCSamples(self, sample_dict=None, fname=None, derived=False, derived_limits=[['fphi0', [0., 1.]]]):
        """
        Converts sample chains into the getdist MCSamples object.

        Parameters
        ----------
        sample_dict : dict
            Sample dictionary with parameter chains, labels and priors.
        fname : str
            To load samples from a file.
        derived : bool
            Whether to include derived data products.
        derived_limits : list
            List of limits for derived parameters, only specify those with boundaries.
        
        Yeild
        -----
        Returns the MCSamples object for a given chain sample.
        """
        from getdist import MCSamples

        if sample_dict is None:
            sample_dict = self.sample2dict(fname=fname, derived=derived, derived_limits=derived_limits)
        
        weights = sample_dict['weights']
        param_names = []
        param_labels = []
        samples = []
        ranges_dict = {}

        for param in sample_dict.keys():
            if param != 'weights':
                param_names.append(param)
                param_labels.append(sample_dict[param]['label'])
                samples.append(sample_dict[param]['chains'])
                if sample_dict[param]['prior'] is not None:
                    ranges_dict[param] = sample_dict[param]['prior']

        return MCSamples(samples=samples, names=param_names, labels=param_labels, weights=weights, ranges=ranges_dict)
    
    
    def quickplot(self, sample_dict=None, fname=None, derived=False, params=None, MLE_dict=None):
        """
        Quickly plots parameter constraints using getdist.

        Parameters
        ----------
        sample_dict : dict
            Sample dictionary with parameter chains, labels and priors.
        fname : str
            To load samples from a file.
        derived : bool
            Whether to include derived data products.
        params : list, optional
            List of parameters to plot.
        MLE_dict : dict, optional
            MLE dictionary.
        """
        
        import matplotlib.pylab as plt
        from getdist import plots

        chains = self.sample2MCSamples(fname=fname, sample_dict=sample_dict, derived=derived)

        if params is None:
            params = self.varied_params
        
        markers_dict = {}

        if MLE_dict is not None:
            for param in params:
                if param in MLE_dict:
                    markers_dict[param] = MLE_dict[param]['MLE']
        
        g = plots.get_subplot_plotter()
        g.triangle_plot([chains], params, filled=True, title_limit=1, markers=markers_dict, marker_args={"lw": 1, 'color':'k', 'ls':'--'})


    def clean(self):
        """
        Reinitialise a class.
        """
        self.__init__()
        
                

