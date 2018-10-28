# csim

**C**ompetitive **S**earch with endogenous **I**ncomplete **M**arkets

Python codes and (Data) Jupyter Notebook for paper *Inflationary Redistribution vs. Trading Opportunities* by Timothy Kam and Junsang Lee

Stationary monetary equilibrium in a competitive search environment with limited and endogenous asset-market participation 

Code solves and simulates stationary equilibrium behavior and non-degenerate distribution of agents

A current version of the paper can be found here

* CSSEGMOD: Class file

* MAIN: Script for running an instance of the model

* SETOPS: Some Python tools for set operations (custom coded)

* SETOUTPATHS: Determine where to save results (.NPZ files) and figures (.PNG and .EPS files)
 * Users need to customize to own computer setting
 
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

## Contributors

Written by Timothy Kam, with Junsang Lee.

## License

The GNU GPL v.3.0 terms apply everywhere to these codes and subsequent uses and modifications of them (see LICENCE). Please cite this original source in all your work (i.e., your own paper and source codes) as:
      * Paper: T. Kam and J. Lee (2017): **Inflationary Redistribution vs Trading Opportunities**: *A New-Monetarist Heterogeneous-agent Quantitative Theory*, *Journal Details TBA*
      * Source: https://github.com/phantomachine/csim/

  See CHANGELOG file for recent updates/bug-fix details.

