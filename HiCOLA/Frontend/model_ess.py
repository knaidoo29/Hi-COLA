from .model_horndeski import HorndeskiModel


class ESS(HorndeskiModel):

    """
    A class for constructing background expansion and growth functions for the 
    Cubic Galieon model.
    """

    def __init__(self):
        """
        Initialises the Horndeski model class.
        """
        super().__init__()

        self.define_K('k_1*X + k_2*X**2', 'k_1, k_2')
        self.define_G3('g_31*X + g_31*X**2', 'g_31, g_32')
        self.define_G4('0.5', None)
    
    
    def set_cosmo_params(
            self, H0_ref, Omega_c0_ref, Omega_b0_ref, fphi, f_k2, f_g2,  w0=-1., wa=0., Tcmb=2.7255, Tnu0=1.9518, mnu=[], Neff=3.044):
        """
        Set cosmological and Horndeski parameters.

        Parameters
        ----------
        H0_ref : float
            Reference LCDM Hubble constant.
        Omega_c0_ref : float
            Reference LCDM cold dark matter density at redshift zero.
        Omega_b0_ref : float
            Reference LCDM baryon density at redshift zero.
        fphi : float
            The fraction of the scalar field density as a fraction of the full dark energy density (including a cosmological constant).
        f_k2 : float
            The ration between k_2 and k_1.
        f_g2 : float
            The ration between g_32 and g_31.
        w0 : float, optional
            Set dark energy equation of state today.
        wa : float, optional
            Set dark energy equation of state gradient.
        Tcmb : float, optional
            The cmb temperature today, set to 2.7255.
        Tnu0 : float, optional
            The relic neutrino temperature today, set to 1.9518.
        mnu : list, optional
            Neutrino mass for each massive species.
        Neff : float, optional
            Effective number of neutrino species used for computing the drag epoch.
        """

        self.params['H0_ref'] = H0_ref
        self.params['Omega_c0_ref'] = Omega_c0_ref
        self.params['Omega_b0_ref'] = Omega_b0_ref
        # photon density set by CMB temperature
        self.params['Tcmb0'] = Tcmb
        self.params['Tnu0'] = Tnu0
        self._get_C_gamma()
        self.params['Omega_g0_ref'] = self.get_Omega_g(1., 1e-2*self.params['H0_ref'], 1.)
        # neutrino density set by CMB temperature
        self._get_C_nu()
        self.params['mnu'] = mnu
        self.params['N_ur'] = 3-len(mnu)
        self.params['Neff'] = Neff
        self.params['Omega_nu_ur0_ref'] = self.get_Omega_nu_ur(1., 1e-2*self.params['H0_ref'], 1.)
        self.params['Omega_nu_nr0_ref'] = 0.
        for _mnu in self.params['mnu']: 
            self.params['Omega_nu_nr0_ref'] = self.get_Omega_nu_nr(1., 1e-2*self.params['H0_ref'], _mnu, 1.) 
        self.params['fphi'] = fphi
        self.params['Omega_l0_LCDM'] = 1. - self.params['Omega_g0_ref'] - self.params['Omega_nu_ur0_ref'] - self.params['Omega_nu_nr0_ref'] - self.params['Omega_c0_ref'] - self.params['Omega_b0_ref']
        self.params['Omega_phi0_ref'] = fphi*self.params['Omega_l0_LCDM']
        self.params['Omega_l0_ref'] = self.params['Omega_l0_LCDM'] - self.params['Omega_phi0_ref']
        self.params['fH'] = 1.
        self.params['w0'] = w0
        self.params['wa'] = wa

        Omega_g0_ref = self.params['Omega_g0_ref']
        Omega_n0_ref = self.params['Omega_nu_ur0_ref']
        Omega_r0_ref = Omega_g0_ref + Omega_n0_ref
        Omega_m0_ref =  self.params['Omega_c0_ref'] + self.params['Omega_b0_ref'] + self.params['Omega_nu_nr0_ref']

        k1_Tracker = 12*fphi*(Omega_r0_ref+Omega_m0_ref-1)/(2+f_k2)
        g31_Tracker = 4*fphi*(1-Omega_r0_ref-Omega_m0_ref)/(2+f_g2)
        k2_Tracker = f_k2*k1_Tracker
        g32_Tracker = f_g2*g31_Tracker
        
        K_G3_G4_values = [k1_Tracker, k2_Tracker, g31_Tracker, g32_Tracker]

        assert len(K_G3_G4_values) == len(self.sym['K_G3_G4_syms']), "Length of Horndeski K_G3_G4_values must match number of defined K, G3, G4 variables."
        self.params['K_G3_G4_values'] = K_G3_G4_values
        