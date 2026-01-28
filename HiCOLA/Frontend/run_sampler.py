import sys
from os import environ

# Set thread environment variables FIRST
N_THREADS = '1'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS

import HiCOLA
import yaml


def main():

    print(' Hi-COLA: Running Sampler')
    print()

    NPROCESSES = int(sys.argv[1])
    YAML_FNAME = str(sys.argv[2])

    print(' - processes: %i' % NPROCESSES)
    print(' - yaml: %s' % YAML_FNAME)
    print()

    with open(YAML_FNAME, mode="r", encoding="utf-8") as file_data:
        settings = yaml.safe_load(file_data)

    samp = HiCOLA.Sampler()
    samp.setup(settings)
    samp.initialise()

    if settings['sampler'] == 'dynesty':
        samp.set_dynesty_settings(**settings['dynesty'])
    elif settings['sampler'] == 'emcee':
        samp.set_emcee_settings(**settings['emcee'])
    elif settings['sampler'] == 'pocoMC':
        samp.set_pocomc_settings(**settings['pocoMC'])

    debug = settings['run'].get('debug', True)
    resume = settings['run'].get('resume', False)
    whichcheckpoint = settings['run'].get('whichcheckpoint', 0)

    print()
    print(' - Running MCMC...')
    print()

    samp.run_mcmc(
        processes=NPROCESSES,
        derived=True,
        debug=debug,
        resume=resume,
        whichcheckpoint=whichcheckpoint,
    )

    print()
    print(' - Saving chains to %s_chains.npz' % samp.fname)

    samp.save_chains()

    if settings['run'].get('MLE', False):
        
        print()
        print(' - Running MLE...')

        samp.get_MLE(root=0, debug=debug, derived=True)
        
        print()
        print(' - Saving MLE to %s_MLE.npz' % samp.fname)

        samp.save_MLE()

    print()
    print(' - Done!')

if __name__ == "__main__":
    main()
