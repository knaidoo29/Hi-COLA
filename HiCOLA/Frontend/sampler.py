import numpy as np
import sympy as sym

from . import redshift, utils

from .likelihood import Likelihood

from .model_standard import StandardModel
from .model_horndeski import HorndeskiModel
from .model_cubic_galileon import CubicGalileon


class Sampler(Likelihood):


    def __init__(self):
        super().__init__()
        self.settings = None
        self.model = None
        self.fixed_params = {}
        self.varied_params = {}
        self.verbose = True


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
        else:
            self.varied_params[param] = {
                'init': self.settings[param]['init'],
                'type': self.settings[param]['prior'][0],
                'prior': [self.settings[param]['prior'][1],self.settings[param]['prior'][2]],
            }


    def setup(self, settings):
        """
        Setups the sampler fixed and varied parameters.
        
        Parameters
        ----------
        settings : dict
            Settings parameter dictionary.
        """

        self.settings = settings

        if self.settings['model'] == 'GR':
            self.model = StandardModel()
        elif self.settings['model'] == 'Horndeski':
            self.model = HorndeskiModel()
        elif self.settings['model'] == 'CubicGalileon':
            self.model = CubicGalileon()
        else:
            assert False, "Unknown model %s" % self.settings['model']

        # check H0
        assert 'H0' in self.settings, "Parameter 'H0' must be defined in settings dictionary."
        self._check_param_settings('H0')

        # check for Omega_b or w_b
        assert 'Omega_b' in self.settings or 'w_b' in self.settings, \
            "Parameter 'Omega_b' or 'w_b' must be defined in settings dictionary."
        if 'Omega_b' in self.settings:
            self._check_param_settings('Omega_b')
        elif 'w_b' in self.settings:
            self._check_param_settings('w_b')

        # check for Omega_c or w_c
        assert 'Omega_c' in self.settings or 'w_c' in self.settings, \
            "Parameter 'Omega_c' or 'w_c' must be defined in settings dictionary."
        if 'Omega_c' in self.settings:
            self._check_param_settings('Omega_c')
        elif 'w_c' in self.settings:
            self._check_param_settings('w_c')
        
        if self.settings['model'] != 'GR':

            # check fphi
            assert 'fphi' in self.settings, "Parameter 'fphi' must be defined in settings dictionary."
            self._check_param_settings('fphi')
        
        if self.settings['model'] == 'Horndeski':

            assert 'K' in self.settings, "Definition for K required."
            assert 'G3' in self.settings, "Definition for G3 required."
            assert 'G4' in self.settings, "Definition for G4 required."

            self.model.define_K(self.settings['K']['func'], self.settings['K']['params'])
            self.model.define_G3(self.settings['G3']['func'], self.settings['G3']['params'])
            self.model.define_G4(self.settings['G4']['func'], self.settings['G4']['params'])

            self.model.construct_model()

        else:

            print()
        
        if self.settings['model'] == 'Horndeski':
            
            # run through Horndeski parameters

            for param in self.model.sym['K_G3_G4_syms']:
                assert str(param) in self.settings, "Parameter %s must be defined in settings dictionary." % param
                self._check_param_settings(str(param))
         
        ### Add conditions for parameters from other models...

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
    