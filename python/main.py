#!/usr/bin/env python

""" MAIN.PY:
    ===========================================================================
    *   Use this MAIN file for single-instance ad-hoc solution and simulation.

    *   Use COMPARESTEADYSTATE instead if you want to vary parametric instances
        and do batch solution and simulation runs. Then use COMPAREBATCHRUNS and COMPARESHORTSELL to do comparative equilibria (More info: README.md)

    *   Use ESTIMATE to perform formal Method of Simulated Moments (MSM)
        estimation of key model parameters

    This file is BASH executable. Run as

        ./main.py

    as a shortcut to the usual statement: python main.py

    For OpenMPI jobs, run in terminal as:

        mpirun -n NPROC ./main.py

    where NPROC is the number of processing nodes you desire on your workstation or on a supercomputing grid/cluster.

    Windoze users: Replace the command mpirun with mpiexec

    (c) 2017- T.Kam (tcy.kam@gmail.com) and J.Lee (junsanglee@skku.edu).

    The GNU GPL v.3.0 terms apply everywhere to these codes and subsequent uses and modifications of them (see LICENCE). Please cite this original source in all your work (i.e., your own paper and source codes) as:
        * Paper: T. Kam and J. Lee (2017): "TITLE OF PAPER", *Journal Details*
        * Code: github.com/phantomachine/cssegmart/

    See CHANGELOG file for recent updates/bug-fix details.
    ===========================================================================
"""


from __future__ import print_function, division, absolute_import
import sys, os
import socket
import time
import numpy as np
#import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#from numba import jit, autojit, double, float64
#import numba
from mpi4py import MPI
#-- Custom Classes ----
from cssegmod import basemod as model
from setoutpaths import *
import scipy.stats as stats


# Close any open FIGURES
plt.close("all")

# Temporary option to fix l_sim bug. Source of bug eliminated. Stored data was contaminated, hence temporaryhack below:
# temporaryhack = True

## ============ SETTINGS CONTROL PANEL =====================================
#SERVER_RUN = 0 # Are you running remotely on server? (== 1)

# Interpolating basis family of your choice (slinear: B-spline family)
Interpolant = 'slinear'
# Output graphics (at the end)
DO_PLOTS = 0
# Load previous results?
LOAD_OLD = False
FRESH_GUESS = None
SAVE_RESULTS = True
# Store sequence of approximation results? (Set to FALSE to speed up!)
STORE_APPROXSEQ = True

# Vary parameters here (you can create new EXPERIMENTS!)
## =======================================================##
#experiment = 69               # READ SETPARAMETERS.PY first!
#Nlists = 80                   # See SET_BATCH_PARAMETERS.PY
## =======================================================##

# MPI rank and cores:
rank = MPI.COMM_WORLD.rank
num_cpus = MPI.COMM_WORLD.size
ncpu_custom = num_cpus

# YOU CAN'T TOUCH THIS, BELOW! ---------------------------------------------
# Automatic Override-assumes you're PLOTTING on existing saved results!
if num_cpus == 1 and rank == 0:
    DO_PLOTS = 1
    LOAD_OLD = True
    ncpu_custom = 24  # Set this number to your previous OpenMPI run's NPROC
# YOU CAN'T TOUCH THIS, ABOVE! ---------------------------------------------

# Set parameters here. Rename EXPERIMENT number for each new instance.

experiment = 1    # (1, NBL) (2, ZBL)

beta_freq = 'Q'

# ===========================================================================
# Inflation and interest rates from annual data
# Lucas and Nicolini, annual 3-mth T-bill-rate mean = 3.4% p/a
# FRED CPI inflation rate 3.9% p/a
# Sample: 1915 to 2007
tau_data_annual = 0.034329539689549854
i_data_annual = 0.03854204301075268
beta_annual  = (1. + tau_data_annual)/(1. + i_data_annual)
if beta_freq == 'Q':
    tau_data = (1.0 + 0.034)**0.25 - 1.0
    i_data = (1.0 + 0.039)**0.25 - 1.0
elif beta_freq == 'A':
    tau_data = tau_data_annual
    i_data = i_data_annual
beta = (1. + tau_data)/(1. + i_data)

# Prob. free access to CM ex ante
alpha = 0.0

### VERSION: 29 July, 08:55 ===================
# UbarDM = 480.2432186
# chi = 0.07834567
# kappa = 0.024968437

# UbarDM = 405.02462
# chi = 0.03
# kappa = 0.003601745

# # V Calibration: JS, 2018-08-07 ------------------------------------------------
# UbarDM = 406.0
# chi = 0.03
# kappa = 0.00399765
# # ^ Calibration: JS, 2018-08-07 ------------------------------------------------
CALIBRATE = 'tokyo'

if CALIBRATE=='tokyo':
    # Calibration: JS, 2018-08-07 -----------------------------------------
    UbarDM = 407.7725
    chi = 0.03
    kappa = 0.0039970
elif CALIBRATE=='canberra':
# Calibration -- 06/10/2018 -----------------------------------------------
    UbarDM = 406.211700000
    chi = 0.021000000
    kappa = 0.004997650

ShortSell = True
# ===========================================================================
tau = tau_data

# Check with settings in COMPARESTEADYSTATE.PY
if experiment == 1:
    if rank == 0:
        print("\nRUNNING: Benchmark Calibration")
        ShortSell = True
elif experiment==10:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing AND Zero Fixed Cost")
    chi = 0.000001
    ShortSell=True
elif experiment==11:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing AND High Fixed Cost")
    chi = 2.0*chi
    ShortSell=True
if experiment == 12:
    if rank == 0:
        print("\nRUNNING: Benchmark Calibration - High inflation")
        ShortSell = True
    tau = 0.025
elif experiment==2:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing")
    ShortSell=False
elif experiment==20:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing AND Zero Fixed Cost")
    chi = 0.000001
    ShortSell=False
elif experiment==21:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing AND High Fixed Cost")
    chi = 2.0*chi
    ShortSell=False
elif experiment==3:
    if rank == 0:
        print("\nRUNNING: Benchmark + Free CM acess")
    alpha = 0.025
    ShortSell = True
## =======================================================================##

## =========================================================================
# Begin stopwatch
if (rank==0):
    tic = time.time()
# Instatiate CSSEGMOD Model Class
klmod = model(  BETA = beta, UbarCM=1.0, UbarDM=UbarDM, KAPPA = kappa,
                CHIprop = chi, ALPHA = alpha, A = 1.0,
                SIGMA_CM = 2.0, SIGMA_DM = 1.01, TAU = tau,
                m_gridsize_ltk = 24, # card.: finite set of m < KAPPA
                m_gridsize_gtk = 48, # card.: grid of states to loop
                m_gridsize_fine=200, # choice set, table-lookup: initial crude
                m_gridsize_refine=1e4, # choice set, table-lookup: later, finer
                MAXITER1=3000,      # iterations over price (OMEGA)
                MAXITER2=120,      # iterations over Bellman functionals
                SMOOTH=0.9,        # secant algorithm smoother: omega loop
                TOL0=1e-4,         # %-precision, %-change omega convergence
                TOL1=1e-5,         # Precision for convergence value functions
                TOL2=1e-5,         # Precision for identifying IsExist(lottery)
                Tsim=10000,        # length of monte carlo sims
                DirtyHack=True,    # check exist lotteries up TOL2 Precision
                InterpMethod=Interpolant, # basis function class, interpolant
                TESTMODE=False,     # Developer's troubleshoot setting
                ShortSell = ShortSell, # Assumption on borrowing, fixed cost CHI
                ymax_adhoc = 0.0, # Relevant only to ad-hoc borrowing limit
                NCPU = ncpu_custom, RANK = rank
            )

# Define paths --------------------------------------------------------------
outpath = 'benchmark/'
figure_directory,outfilename,previous_outfilename,rootdir =\
                                        filefind(experiment,rootdir=outpath)

# Messages and Information for User -----------------------------------------
if (rank==0):
    print("\nRunning Experiment %i"  %(experiment))
    print("\nSHORT SELL? %s" % (str(ShortSell)) )
    print("\n\tchi=%0.6f, CHI=%0.6f, A=%0.3f, UbarCM=%0.6f, UbarDM=%0.6f,  TAU=%0.6f, KAPPA=%0.6f, ALPHA=%0.6f" % (chi, klmod.CHI, klmod.A, klmod.UbarCM, klmod.UbarDM, klmod.TAU, klmod.KAPPA, klmod.ALPHA))
    if klmod.TESTMODE == True:
        print("\n\tYou have enabled verbose display as self.TESTMODE=True!")
    print("\n\tNumber of CPUs:" + str(num_cpus))
    print("\n\tParallel grids >> N_ltk = %i, N_gtk = %i, m_gridsize = %i" \
            % (klmod.m_gridsize_ltk, klmod.m_gridsize_gtk, klmod.m_gridsize) )

    print("\n\tGiven A = %0.4f, upper bound on m: %0.4f" %(klmod.A, klmod.m_max))
    if LOAD_OLD == False or (LOAD_OLD == True and FRESH_GUESS == False):
        print("\n\tResults will be saved to: " + outfilename+'.npy ...')
        print("\t> Computing now. One moment please ........")

# Solve for SME --------------------------------------------------------------
results = klmod.SolveSteadyState( outfilename,
                                LOAD_OLD=LOAD_OLD,
                                FRESH_GUESS=FRESH_GUESS,
                                SAVE_RESULTS=SAVE_RESULTS,
                                STORE_APPROXSEQ=STORE_APPROXSEQ,
                            )

if (rank==0):
    Tstop = time.time() - tic
    print("Time elapsed (s): %6.5g seconds" % (Tstop))

# Analyze results ------------------------------------------------------------
if (rank == 0):
    if (SAVE_RESULTS==True) and (LOAD_OLD==False or (LOAD_OLD==True and FRESH_GUESS==True)):
        print("\nNow saving results to .NPY file ...")
        filename = outfilename + '.npy'
        np.save(filename, results)

    # Unpack dictionary RESULTS from above for use in plotting and etc. below
    # -----------------------------------------------------------------------
    # Stored results along approx. sequence
    V_seq = results['diagnostics']['V_seq']
    Vbar_seq = results['diagnostics']['Vbar_seq']
    V_tilde_seq = results['diagnostics']['Vtilde_seq']
    lottery_seq = results['diagnostics']['lottery_seq']
    errorOmega_seq = results['diagnostics']['errorOmega_seq']
    errorV_seq = results['diagnostics']['errorV_seq']
    # value function evaluated on grid
    Vbar = results['valuegrid']['Vbar']
    V = results['valuegrid']['V']
    Vtilde = results['valuegrid']['Vtilde']
    B = results['valuegrid']['B']
    W = results['valuegrid']['W']
    # Fitted value functions
    Vbarfit = results['valuefit']['Vbar']
    Vfit = results['valuefit']['V']
    Bfit = results['valuefit']['B']
    Wfit = results['valuefit']['W']
    # policy functions evaluated on grid
    lstar = results['policygrid']['lstar']
    ystar = results['policygrid']['ystar']
    bstar = results['policygrid']['bstar']
    qstar = results['policygrid']['qstar']
    xstar = results['policygrid']['xstar']
    # lotteries
    lottery_supports = results['lottery']['supports']
    lottery_payoffs = results['lottery']['payoffs']
    # Monte Carlo sims: distribution
    simhistory = results['distro']['simhistory']
    # # This is a manual hack (2018-09-14) Python dict seems to
    # # mess up 'q' in simhistory! Note self.Q() also updated in CSSEGMOD.PY
    # qs = self.Q(simhistory['x'], simhistory['b'])
    # simhistory['q'] = qs
    omega = results['distro']['omega']

    # Consumption distribution
    partsim = results['distro']['simhistory']['parti'] # CM/DM participation
    # Get history of DM's q, then fill with CM C whenever location says in CM:
    consim = simhistory['q'].copy()
    consim[partsim==2] = klmod.Cstar

    # Graphics Department -----------------------------------------------------
    if (DO_PLOTS == 1) and (num_cpus == 1):
        print('\n\n\tNow plotting and saving to EPS ... Value functions')
        plt = klmod.DrawValuefun(V,B,W-klmod.CHI,Vtilde,Vfit,\
                lottery_supports,lottery_payoffs, PlotSeparate=False)
        plt.savefig(figure_directory+'values.eps')
        plt.savefig(figure_directory+'values.png')

        print('\n\n\tNow plotting and saving to EPS ... Decision functions')
        policies = [lstar, bstar, qstar, xstar]
        policynames = ['$l^{*}$', '$b^{*}$', '$q^{*}$', '$x^{*}$']
        plt = klmod.DrawPolicyfun(policies, policynames, lottery_supports)
        plt.savefig(figure_directory+'policies.eps')
        plt.savefig(figure_directory+'policies.png')

        print('\n\n\tNow plotting and saving to EPS ... Distribution')
        plt = klmod.DrawDistro(simhistory['m'], lottery_supports, bins=100)
        plt.savefig(figure_directory+'distro.eps')
        plt.savefig(figure_directory+'distro.png')

        print('\n\n\tNow plotting and saving to EPS ... Simulations')
        T_sim = len(simhistory['b'])
        T_plt = 10000
        start = max(T_sim - T_plt, 0)
        stop = T_sim
        # Loop to plot Simulations
        ydata={ 'sims_m'       :   [ 'm'      , '$m$' ]  ,
                'sims_l'       :   [ 'l'      , '$l$' ]  ,
                'sims_b'       :   [ 'b'      , '$b$' ]  ,
                'sims_x'       :   [ 'x'      , '$x$' ]  ,
                'sims_q'       :   [ 'q'      , '$q$' ]  ,
                'sims_match'   :   [ 'match'  , 'Match Status' ]  ,
                'sims_lottery' :   [ 'lottp'  , 'Lottery Won, $z_{j}(m)$' ],
                'sims_particpate' :   [ 'parti'  , 'Market status' ]
                }
        for idx_fig, key in enumerate(ydata.keys()):
            y = ydata[key]
            series = simhistory[y[0]][start:stop]
            plt = klmod.DrawTimeSeries(series, y[1])
            plt.savefig(figure_directory + key + '.eps')
            plt.savefig(figure_directory + key + '.png')

        # Render plots on screen
        plt.show(block=False)
        # plt.show()

    # Statistics Department ---------------------------------------------------
    if (num_cpus == 1):
        # Print some results
        print("\n=====================================================")
        print("\nSteady state equilibrium: Allocations")
        print("\n-----------------------------------------------------")
        omega = klmod.MarketClearing(simhistory)
        inflation = klmod.MoneyGrowthFactor(omega)
        print("\nomega = %3.4f, inflation rate  = %3.4f" \
                                        %(omega, inflation-1.0) )
        print("\nLottery prizes (supports):")
        print(lottery_supports)

        # Distribution of money - moments
        print("\n=====================================================")
        print("\nSteady state equilibrium: Money Distribution")
        print("\n-----------------------------------------------------")
        msim = simhistory['m']
        print("\n* Mean Asset = %3.2f" % (np.mean(msim)) )
        print("\n* Std Asset = %3.2f" % (np.std(msim)) )
        print("\n* Skewness Asset = %3.2f" % (stats.skew(msim)) )
        print("\n* Skewness Test (Z-value): %3.2f" % (stats.skewtest(msim)[0]) )
        print("\n* Skewness Test (p-value): %3.2f" % (stats.skewtest(msim)[1]) )
        print("\n* Gini (Money/Asset Holding) = %3.6f" \
                                    % (klmod.gini_rawdata(msim)) )

        # Welfare Distribution
        print("\n=====================================================")
        print("\nSteady state equilibrium: Welfare (CEV) Distribution")
        print("\n-----------------------------------------------------")
        Vbarsim = Vbarfit(msim) # ex-ante welfare simulation history
        #cevsim = klmod.invU_CM(Vbarsim)
        cevsim = np.absolute(((1-klmod.SIGMA_CM)*Vbarsim + 1.0)**(1.0/(1.0-klmod.SIGMA_CM)))
        print("\n* Mean CEV = %3.6f" % (np.mean(cevsim)) )
        print("\n* Std CEV = %3.6f" % (np.std(cevsim)) )
        print("\n* Skewness CEV = %3.6f" % (stats.skew(cevsim)) )
        print("\n* Skewness Test (Z-value): %3.6f" \
                                    % (stats.skewtest(cevsim)[0]) )
        print("\n* Skewness Test (p-value): %3.6f" \
                                    % (stats.skewtest(cevsim)[1]) )

        # Consumption Distribution
        print("\n=====================================================")
        print("\nSteady state equilibrium: Consumption Distribution")
        print("\n-----------------------------------------------------")
        print("\n* Mean Consumption = %3.6f" % (np.mean(consim)) )
        print("\n* Std Consumption = %3.6f" % (np.std(consim)) )
        print("\n* Skewness Consumption = %3.6f" % (stats.skew(consim)) )
        print("\n* Skewness Test (Z-value): %3.6f" \
                                    % (stats.skewtest(consim)[0]) )
        print("\n* Skewness Test (p-value): %3.6f" \
                                    % (stats.skewtest(consim)[1]) )
        print("\n* Gini (Consumption) = %3.6f" \
                                    % (klmod.gini_rawdata(consim)) )

        # Pricing distribution
        print("\n=====================================================")
        print("\nSteady state equilibrium: Pricing Distribution")
        print("\n-----------------------------------------------------")
        xsim = simhistory['x']
        xsim[partsim==2] = 0.0
        xsim[np.isnan(xsim)] = 0.0
        qsim = simhistory['q']
        qsim[partsim==2] = 1000.0
        qsim[np.isnan(qsim)] = 100.0
        qsim[qsim==0.0] = 10.0
        xsim[qsim==0.0] = 0.0
        # print(xsim)
        # print(qsim)
        # DM prices
        pricesim = xsim/qsim
        # CM price
        pricesim[partsim==2] = 1.0
        print("\n* Mean Prices = %3.6f" % (np.mean(pricesim)) )
        print("\n* Std Prices = %3.6f" % (np.std(pricesim)) )

    if (num_cpus > 1):
        print("\n\n\nFinishing up ... GOODBYE!\n\n")
        MPI.COMM_WORLD.Abort()
