import numpy as np

def set_batch_parameters(BETA=0.96, Npoints=48, \
                                    tau_specifics=[], shortsell=True):
    """Set modifiable model parameters here. This setting is common to these other scripts:
    * MAIN.PY
    * TRANSITION.PY
    * COMPARESTEADYSTATE.PY
    """
    Npoints = Npoints - len(tau_specifics)
    # labor supply elasticity (convexity)
                         # chi-ZERO [ 0.0 ]
    chi_array = [ 0.03 ]  # chi-LO [ 0.1 ]
                         # chi-HI [ 0.4 ]

    # CRRA (CM)
    sigma_array = [ 2.0 ]
    # Transfer rate
    if shortsell == True:
        # Natural borrowing limit economy
        tau_array = np.linspace(BETA-1.0+1e-4, 0.025, Npoints)
        # tau_array = np.linspace(0.0, 0.05, Npoints)
    elif shortsell == False:
        # Zero borrowing constraint tau <= 0 does not seem to have equilibrium
        tau_array = np.linspace(0.001, 0.025, Npoints)
    # Add two reference points for experiments later
    tau_array = np.concatenate((tau_array, tau_specifics))
    tau_array = np.sort(np.unique(tau_array))
    # Cartesian product
    param_batch = [ [i,j,k] for i in chi_array
                            for j in sigma_array
                            for k in tau_array   ]
    # Add new line above to create new experiment. That's all you need to do.

    return param_batch
