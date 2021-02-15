import set_batch_parameters as batchparams

def setparameters(rank, experiment, N_lists=50):
    """Set modifiable model parameters here. This setting is common to these other scripts:
    * MAIN.PY

    Dependencies:
    * SET_BATCH_PARAMETERS.PY

    User Notes:
    Step 1: Open up and modify SET_BATCH_PARAMETERS.PY as required.
            You may vary definition of parameters to your liking. Know the theory first!
    Step 2: In MAIN.PY set EXPERIMENT as an integer from 0 to some max integer,
            N := len(param_batch)

    EXAMPLE usage in MAIN.PY
        -   e.g., we're varying parameters {chi, sigma, tau}
        -   There are Nlists=50 such combinations
        - So example usage is:

        >   # Pick EXPERIMENT:
        >   experiment = 23
        >   paramlist = setparameters(rank, experiment, Nlists=50)

        >   # Name and instantiate elements of paramlist:
        >   chi, sigma, tau = paramlist


    """

    # Parameter combinations. USER: please modify SET_BATCH_PARAMETERS()
    param_batch = batchparams.set_batch_parameters(Npoints=N_lists)

    # Instance EXPERIMENT from param_batch:
    paramlist = param_batch[experiment]

    return paramlist
