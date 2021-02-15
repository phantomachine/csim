## MASHA

Monetary Asset Search Heterogenous Agent model.

## README.md

Python 2.7/3.6 code for solving and simulating monetary equilibria in the model of Kam and Lee (2017). Code uses MPI4Py and OpenMPI for parallel computation.
(c) Copyright 2017-, Timothy Kam and Junsang Lee. Contact: tcy.kam@gmail.com

## Code Example

To run the MAIN.PY file in normal mode (serial):

> ``python main.py``

or, in IPython:

> ``run main``

To run the MAIN.PY file using OpenMPI:

> ``mpirun -n YourNumberOfCPUs python main.py``

or, if you have chmod MAIN.PY to be BASH Executable you can also invoke as

> ``mpirun -n YourNumberOfCPUs ./main.py``

Same goes for CALIBRATE.PY or COMPARESTEADYSTATES.PY

## Motivation

The project builds on the model of Menzio, Shi and Sun (2013) in JET by making it quantitative and introducing the option for agents to visit a centralized market subject to a real fixed participation cost.

## Scripts and Classes

* Main model class is CSSEGMOD.PY

* Run single instance using MAIN.PY

* Batch experiments:

  * COMPARESTEADYSTATE.PY (main script)

  * SET_BATCH_PARAMETERS.PY (parameter settings for COMPARESTEADYSTATE.PY)

* Save plotting results of COMPARESTEADYSTATE:

  * COMPAREBATCHRUNS.PY: Run this first to generate at least two economies across different inflation rates

  * COMPARESHORTSELL.PY: Run this after having saved results from COMPAREBATCHRUNS

## Contributors

Written by Timothy Kam and Junsang Lee.

## License

The GNU GPL v.3.0 terms apply everywhere to these codes and subsequent uses and modifications of them (see LICENCE). Please cite this original source in all your work (i.e., your own paper and source codes) as:
      * Paper: T. Kam and J. Lee (2017): "TITLE OF PAPER", *Journal Details*
      * Code: https://github.com/phantomachine/cssegmart/

  See CHANGELOG file for recent updates/bug-fix details.
