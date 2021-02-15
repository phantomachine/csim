#!/usr/bin/env python

from __future__ import print_function, division, absolute_import
import sys, os
import socket
import time
import numpy as np
# import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from tabulate import tabulate
#from numba import jit, autojit, double, float64
#import numba
from mpi4py import MPI
import scipy.stats as stats

#-- Custom Classes and Modules ----
from cssegmod import basemod as model   # Model class
from set_batch_parameters import *      # Define parameter combinations here
from setoutpaths_batch import *         # Define output path for saving files
import send                             # for sending auto email

# ================= USER DEFINE BATCH EXPERIMENT SETTINGS HERE ================
# Do you want to bug your coauthors with email notifications?
SENDMAIL = True

# x-axis variable (TAU or INFLATION)
PLOT_xaxis_TAU = True

# Use DM good as basis for CEV calculation?
DM_GOOD_CEV = False

# Specify email receipients here:
# email_list = ["tcy.kam@gmail.com", "junsang.skku@gmail.com"]
email_list = ["tcy.kam@gmail.com"]

# MPI rank and cores:
rank = MPI.COMM_WORLD.rank
num_cpus = MPI.COMM_WORLD.size

# Custom setting for defining grids in self.m_grid
ncpu_custom = num_cpus

# ****************************************************************************
# Set BATCH experiment ID here:
Batch_ID = 30

# Number of equilibrium points
N_eqm = 24

# Number of Monte-Carlo paths (also number cross-section individuals)
Hsim = 1000

#  NOTES: ShortSell **********************************************************
# Borrowing Constraint Assumption: shortsell =
# True (natural limit)
# False (Zero borrowing limit)
# None (ad-hoc +ve limit)
# ****************************************************************************

# ================= EXTERNALLY CALIBRATED PARAMETERS HERE ==============
beta_freq = 'Q'

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

# override experiment settings: User customisable

alpha = 0.0
# V Calibration: JS, 2018-08-07 ------------------------------------------------
# UbarDM = 404.25 # 406.0
# chi = 0.032
# kappa = 0.00399765 # 0.00399765
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



# ------------------------------------------------

if Batch_ID == 1:
    if rank == 0:
        print("\nRUNNING: Benchmark Calibration")
    batch = 'benchmark'
    ShortSell = True
elif Batch_ID == 10:
    if rank == 0:
        print("\nRUNNING: Zero Fixed Cost")
    batch = 'benchmarkzerofixcost'
    chi = 0.000001
    ShortSell = True
elif Batch_ID == 11:
    if rank == 0:
        print("\nRUNNING: High Fixed Cost")
    batch = 'benchmarkhighfixcost'
    chi = 2.0*chi
    ShortSell = True
elif Batch_ID == 2:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing Economy")
    batch = 'zbl'
    ShortSell = False
elif Batch_ID == 20:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing Economy - Zero Fix Cost")
        chi = 0.000001
    batch = 'zblzerofixcost'
    ShortSell = False
elif Batch_ID == 21:
    if rank == 0:
        print("\nRUNNING: Zero Borrowing Economy - High Fix Cost")
        chi = 2.0*chi
    batch = 'zblhighfixcost'
    ShortSell = False
elif Batch_ID == 3:
    if rank == 0:
        print("\nRUNNING: Benchmark + Free CM acess")
        alpha = 0.025
    batch = 'freeaccess'
    ShortSell = True
elif Batch_ID == 30:
    if rank == 0:
        print("\nRUNNING: Benchmark + High Free CM acess")
        alpha = 0.25
    batch = 'freeaccess-hialpha'
    ShortSell = True
# elif Batch_ID == 4:
#     if rank == 0:
#         print("\nRUNNING: Very High Fixed Cost")
#     batch = 'veryhighfixcost'
#     chi = 0.30
#     ShortSell = True

# Reference economies for BEFORE vs AFTER welfare (CEV) comparisons
#
# baseline calibration
# tau_data = 0.010252
# Experiment at ZERO inflation
if (ShortSell == True) or (ShortSell == None):
    tau_before = 0.0
elif (ShortSell == False):
    tau_before = 0.001

if alpha > 0.0:
    # For precision reasons (LotteryTest), tau = 0.0 may fail to Compute
    # when alpha > 0 ... So hack it here!
    tau_before = 0.00001

# Experiment BEFORE at ...
# tau_before = 0.0

# Experiment AFTER at ...
tau_after = 0.025

# List to be appended to PARAMS below
tau_specifics = [tau_data, tau_before, 0.0025, 0.0125, tau_after]

# Set or create directory
homename = "comparemonty2future"
homedir = homename + '/'
if not os.path.exists(homedir):
    os.makedirs(homedir)

# Patch to results of current batch run
rootdir = homedir + batch + '/'

# Set up table of parameters lists: Customisable (use for TAU grid)
# beta = 0.99

params = set_batch_parameters(BETA=beta, Npoints=N_eqm,\
                                            tau_specifics=tau_specifics,\
                                                        shortsell=ShortSell)

# For sampling from agent's Monte Carlo history
Tsim = 10000
start = 2           # Start equilibrium economy
stop = 139           # End equilibrium economy

# ================= USER DEFINE BATCH EXPERIMENT SETTINGS HERE ================

# Close any open FIGURES
plt.close("all")
plt.rcParams.update({'axes.titlesize': 'large'})

# Load previous results?
LOAD_OLD = True
HOMOTOPY = True
FRESH_GUESS = True
SAVE_FIGURES = True
STORE_APPROXSEQ = False
SAVE_RESULTS = True
DO_TABLES = False
DO_PLOTS = True
# Saving to plt0 along each loop - value and policy function plots
DO_FUNCTION_PLOTS = False

# Automatic Override-assumes you're PLOTTING on existing saved results!
if num_cpus == 1 and rank == 0:
    LOAD_OLD = True
    DO_PLOTS = True
    # DO_TABLES = True
    DO_FUNCTION_PLOTS = False
    ncpu_custom = 20
    HOMOTOPY = False
    FRESH_GUESS = False

if num_cpus > 1 and rank == 0:
    emailstart = "STARTED > "
    # Send an automatic email to subscribers: see SEND.PY
    if SENDMAIL:
        send.message( rootdir, custom_message=emailstart, toaddr=email_list )

# Table lists for aggegrate outcomes/statistics
table_omega, table_omega_temp = [],[]
table_omega_five, table_omega_ninefive = [], []

table_inflation, table_inflation_temp = [], []
table_inflation_five, table_inflation_ninefive = [], []

table_moneystat0, table_moneystat0_temp = [], []
table_moneystat0_five, table_moneystat0_ninefive = [], []

table_moneystat1, table_moneystat1_temp = [], []
table_moneystat1_five, table_moneystat1_ninefive = [], []

table_moneystat2, table_moneystat2_temp = [], []
table_moneystat2_five, table_moneystat2_ninefive = [], []

table_moneystat3, table_moneystat3_temp = [], []
table_moneystat3_five, table_moneystat3_ninefive = [], []

table_moneystat4, table_moneystat4_temp = [], []
table_moneystat4_five, table_moneystat4_ninefive = [], []

table_moneystat5, table_moneystat5_temp = [], []
table_moneystat5_five, table_moneystat5_ninefive = [], []

table_welfarestat0, table_welfarestat0_temp = [], []
table_welfarestat0_five, table_welfarestat0_ninefive = [], []

table_welfarestat1, table_welfarestat1_temp = [], []
table_welfarestat1_five, table_welfarestat1_ninefive = [], []

table_welfarestat2, table_welfarestat2_temp = [], []
table_welfarestat2_five, table_welfarestat2_ninefive = [], []

table_welfarestat3, table_welfarestat3_temp = [], []
table_welfarestat3_five, table_welfarestat3_ninefive = [], []

table_welfarestat4, table_welfarestat4_temp = [], []
table_welfarestat4_five, table_welfarestat4_ninefive = [], []

# Table lists for average actions
table_lmean, table_lmean_temp = [], []
table_lmean_five, table_lmean_ninefive = [], []

table_ymean, table_ymean_temp = [], []
table_ymean_five, table_ymean_ninefive = [], []

table_xmean, table_xmean_temp = [], []
table_xmean_five, table_xmean_ninefive = [], []

table_bmean, table_bmean_temp = [], []
table_bmean_five, table_bmean_ninefive = [], []

table_qmean, table_qmean_temp = [], []
table_qmean_five, table_qmean_ninefive = [], []


# Table list for CM participation rate
table_parti, table_parti_temp = [], []
table_parti_five, table_parti_ninefive = [], []

# Table lists for b.x and b.q
table_xavg, table_xavg_temp = [], []
table_xavg_five, table_xavg_ninefive = [], []

table_qavg, table_qavg_temp = [], []
table_qavg_five, table_qavg_ninefive = [], []

# Table lists for b.x/y and b.q/y
table_xveloCM, table_xveloCM_temp = [], []
table_xveloCM_five, table_xveloCM_ninefive = [], []

table_qveloCM, table_qveloCM_temp = [], []
table_qveloCM_five, table_qveloCM_ninefive = [], []

# Table lists for b.x/M(DM) and b.q/M(DM)
table_xveloDM, table_xveloDM_temp = [], []
table_xveloDM_five, table_xveloDM_ninefive = [], []

table_qveloDM, table_qveloDM_temp = [], []
table_qveloDM_five, table_qveloDM_ninefive = [], []

# Approximation scheme -- see def Interp1d in src/mssmod9b.py
Interpolant = 'slinear'


# List of random seeds for Distribution()
randseed_list = [seed for seed in range(Hsim)]

# loop over steady states ...
for experiment, plist in enumerate(params):

    # Current loop experiment
    #chi, sigma, tau = plist
    tau = plist[2]     # we're fixing CHI via Batch_ID above!

    ## =======================================================================#
    # Instatiate MSS Model Class with approx. grid size 100. Defaults built in:
    klmod = model(  BETA = beta, UbarCM=1.0, UbarDM=UbarDM, KAPPA=kappa,
                CHIprop = chi, ALPHA = alpha, A = 1.0,
                SIGMA_CM = 2.0, SIGMA_DM = 1.01, TAU = tau,
                m_gridsize_ltk = 24, # card.grid m < KAPPA, NCPU-divisible
                # m_gridsize_gtk = 36, # card.grid m > KAPPA, NCPU-divisible
                m_gridsize_gtk = 48, # card.grid m > KAPPA, NCPU-divisible
                m_gridsize_fine=300, # choice set, table-lookup: initial, crude
                m_gridsize_refine=1e4, # choice set, table-lookup: later, finer
                MAXITER1=3000,     # iterations over price (OMEGA)
                MAXITER2=150,      # iterations over Bellman functionals
                SMOOTH=0.9,        # secant algorithm smoother: omega loop
                TOL0=1e-4,         # %-precision, %-change omega convergence
                TOL1=1e-5,         # Precision for convergence value functions
                TOL2=1e-5,         # Precision for identifying IsExist(lottery)
                Tsim=10000,        # length of monte carlo sims
                DirtyHack=True,    # check exist lotteries up TOL2 Precision
                InterpMethod=Interpolant, # basis function class, interpolant
                TESTMODE=False,     # Developer's troubleshoot setting
                ShortSell = ShortSell, # Assumption on borrowing, fixed cost CHI
                ymax_adhoc = 0.00, # Relevant only to ad-hoc borrowing limit
                NCPU = ncpu_custom, RANK = rank
            )


    # Define paths ------------------------------------------------------------
    # Figures stored in FIGURES/EXPERIMENTxx/
    # rootdir = 'results/compare_alpha_HI/'
    figure_directory, outfilename, previous_outfilename, rootdir =\
                                        createdir(experiment,rootdir=rootdir)


    # Begin stopwatch ---------------------------------------------------------
    tic = time.time()

    # Messages and Information for User ---------------------------------------
    if (rank == 0):
        print("\n\nRunning Experiment %i of %i"  %(experiment+1, len(params)) )
        # print("\n\tCHI=%0.3f, ALPHA=%0.3f, SIGMA_CM=%0.3f, TAU=%0.6f" \
        #       % (klmod.CHI, klmod.ALPHA, klmod.SIGMA_CM, klmod.TAU))
        print("\nSHORT SELL? %s" % (str(ShortSell)) )
        print("\nBATCH? %s" % (str(batch)) )
        print("\n\tchi=%0.6f, CHI=%0.6f, A=%0.3f, UbarDM=%0.6f, TAU=%0.6f, KAPPA=%0.6f" % (chi, klmod.CHI, klmod.A, UbarDM, klmod.TAU, klmod.KAPPA))
        if klmod.TESTMODE == True:
            print("\n\tYou have enabled verbose display as self.TESTMODE=True!")
        print("\n\tNumber of CPUs:" + str(num_cpus))
        print("\n\tParallel grids >> N_ltk = %i, N_gtk = %i, m_gridsize = %i" \
                % (klmod.m_gridsize_ltk, klmod.m_gridsize_gtk, klmod.m_gridsize) )

        print("\n\tGiven A = %0.4f, upper bound on m: %0.4f" \
                                                % (klmod.A, klmod.m_max))
        #if LOAD_OLD == False or (LOAD_OLD == True and FRESH_GUESS == False):
        print("\n\tResults will be saved to: " + outfilename+'.npy ...')
        print("\n\t> Computing now. One moment please ........")

    if (num_cpus > 1):
        if experiment == 0:
            LOAD_OLD = False
            HOMOTOPY = False
            SAVE_RESULTS = True
        elif experiment >= 1:
            LOAD_OLD = True
            HOMOTOPY = True
        if rank == 0:
            print("\n\t*** HOMOTOPY is set to " + str(HOMOTOPY) + " ***\n")

    # Solve instance of parameterized model. Get output as dictionary RESULTS:
    results = klmod.SolveSteadyState(   outfilename, previous_outfilename,
                                        LOAD_OLD=LOAD_OLD,
                                        HOMOTOPY=HOMOTOPY,
                                        FRESH_GUESS=FRESH_GUESS,
                                        SAVE_RESULTS=SAVE_RESULTS,
                                        STORE_APPROXSEQ = STORE_APPROXSEQ,
                                        randseed=randseed_list[0],
                                    )
    toc = time.time()
    print(toc - tic)

    # Save each run's RESULTS to an .NPY file ---------------------------------
    if (rank == 0):
        # if (SAVE_RESULTS==True) and (LOAD_OLD==False or (LOAD_OLD==True and FRESH_GUESS==True)):
        if (SAVE_RESULTS==True) and (num_cpus > 1):
            filename = outfilename + '.npy'
            print("\nNow saving results to " + filename)
            np.save(filename, results)

    # Unpack dictionary RESULTS above for plotting and tables below ----------
    if (rank == 0) and (num_cpus > 1):
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
        omega = results['distro']['omega']

        # Store current steady state equilibrium for tabulation...
        inflation = klmod.MoneyGrowthFactor(omega)

        tick = time.time()
        for idx_seed, seed in enumerate(randseed_list):

            # Simulate again for new randseed for idx_seed > 0
            if idx_seed > 0:
                simhistory, flag = klmod.Distribution(lottery_supports,
                                                Vfit,Wfit,Bfit,
                                                ystar,bstar,qstar,xstar,
                                                omega, seed=seed )

            # STORAGE of RESULTS ------>
            #table_mhat.append(mhat)
            table_omega_temp.append(omega)
            table_inflation_temp.append(inflation-1.0)

            # participation rate in CM
            participation = simhistory['parti']
            partrateCM = participation[participation==2].size/participation.size
            partrateDM =  1.0 - partrateCM
            table_parti_temp.append(partrateCM)

            # Conditional mean allocations of l, y (adjusted for transfer),q, x, b
            lmean = (simhistory['l'][participation==2]).mean()/klmod.l_max
            ymean = (simhistory['y'][participation==2]).mean()
            ymean = klmod.MoneyBalance(ymean, omega)
            xmean = simhistory['x'][participation==1].mean()
            bseries = simhistory['b']
            bs0 = bseries[participation==0]
            bs1 = bseries[participation==1].max() # take largest (initial match)
            wb0 = 0.0*bs0.size / (bs0.size + bs1.size)
            bmean = ( wb0*bs0.mean() + (1.0 - wb0)*bs1.mean() )
            qmean = simhistory['q'][participation==1].mean()
            table_lmean_temp.append(lmean)
            table_ymean_temp.append(ymean)
            table_xmean_temp.append(xmean)
            table_bmean_temp.append(bmean)
            table_qmean_temp.append(qmean)

            # Inverse of velocity
            inv_velocity = (1.0/omega)/(partrateDM*xmean \
                                        + partrateCM*klmod.Cstar)

            # # Average flow of q and x, conditional on DM
            # x_avg = (simhistory['b'][participation==1]).dot(simhistory['x'][participation==1])
            # q_avg = (simhistory['b'][participation==1]).dot(simhistory['q'][participation==1])
            bsim_DM = simhistory['b'][participation==1]
            xsim_DM = simhistory['x'][participation==1]
            qsim_DM = simhistory['q'][participation==1]
            ysim_CM = simhistory['y'][participation==2]

            # table_xavg.append(x_avg)
            # table_qavg.append(q_avg)
            table_xavg_temp.append(bsim_DM.dot(xsim_DM))
            table_qavg_temp.append(bsim_DM.dot(qsim_DM))

            # Velocity CM saving in DM spending
            # table_xveloCM.append(x_avg/ymean)
            # table_qveloCM.append(q_avg/ymean)

            bxm_CM = (bsim_DM*xsim_DM).mean() / ysim_CM.mean()
            table_xveloCM_temp.append(bxm_CM)
            bqm_CM = (bsim_DM*qsim_DM).mean() / ysim_CM.mean()
            table_qveloCM_temp.append(bqm_CM)

            # Velocity DM balance in DM spending
            # bsim_DM = simhistory['b'][participation==1]
            # xsim_DM = simhistory['x'][participation==1]
            # qsim_DM = simhistory['q'][participation==1]
            # Note: initial m before lottery (for ex-post DM entering agents)
            msim_DM = simhistory['m'][participation==1]

            bxm_DM = ( (bsim_DM*xsim_DM)/msim_DM ).mean()
            table_xveloDM_temp.append(bxm_DM)

            bqm_DM = ( (bsim_DM*qsim_DM)/msim_DM ).mean()
            table_qveloDM_temp.append(bqm_DM)

            # Distribution of money - moments
            msim = simhistory['m']
            gini = klmod.gini_rawdata(msim)

            table_moneystat0_temp.append(np.median(msim))
            table_moneystat1_temp.append(np.mean(msim))
            table_moneystat2_temp.append(np.std(msim))
            table_moneystat3_temp.append(stats.skew(msim))
            table_moneystat4_temp.append(gini)
            table_moneystat5_temp.append(inv_velocity)

            # Distribution of welfare - moments
            Vbarsim = Vbarfit(simhistory['m'])

            # table_welfarestat1.append(np.median(Vbarsim))
            # table_welfarestat1.append(np.mean(Vbarsim))
            # table_welfarestat2.append(np.std(Vbarsim))
            # table_welfarestat3.append(stats.skew(Vbarsim))

            # Equivalent consumption at each V(m) relative to first equilibrium
            if DM_GOOD_CEV == True:
                cevsim = np.exp(Vbarsim) - klmod.q_min
            else:
                #cevsim = -klmod.invU_CM(-Vbarsim)
                cevsim = np.absolute(((1-klmod.SIGMA_CM)*Vbarsim + 1.0)**(1.0/(1.0-klmod.SIGMA_CM)))
            CEV_mean = cevsim.mean()
            CEV_median = np.median(cevsim)
            if experiment == 0:
                CEV_mean0 = CEV_mean
                CEV_median0 = CEV_median
            CEV_mean = (CEV_mean / CEV_mean0 - 1.0)*100.
            CEV_mean = (CEV_median / CEV_median0 - 1.0)*100.

            #table_welfarestat4.append(q_G.tolist())
            table_welfarestat0_temp.append(CEV_median)
            table_welfarestat1_temp.append(CEV_mean)
            table_welfarestat2_temp.append(np.std(cevsim))
            table_welfarestat3_temp.append(stats.skew(cevsim))
            #table_welfarestat4.append(np.mean(Vbarsim))
            table_welfarestat4_temp.append(cevsim.mean())

            # Update status bar
            klmod.StatusBar_simple(idx_seed,len(randseed_list),"Monte Carlo...")

        tock = time.time()
        print(tock - tick)
        # END: For idx_seed (still rank=0)
        # ---------------------------------------------------------------------

        # STORE averages from Monte Carlo sims -------> MEAN
        table_omega.append(np.mean(table_omega_temp))
        table_inflation.append(np.mean(table_inflation_temp))
        table_moneystat0.append(np.mean(table_moneystat0_temp))
        table_moneystat1.append(np.mean(table_moneystat1_temp))
        table_moneystat2.append(np.mean(table_moneystat2_temp))
        table_moneystat3.append(np.mean(table_moneystat3_temp))
        table_moneystat4.append(np.mean(table_moneystat4_temp))
        table_moneystat5.append(np.mean(table_moneystat5_temp))
        table_welfarestat0.append(np.mean(table_welfarestat0_temp))
        table_welfarestat1.append(np.mean(table_welfarestat1_temp))
        table_welfarestat2.append(np.mean(table_welfarestat2_temp))
        table_welfarestat3.append(np.mean(table_welfarestat3_temp))
        table_welfarestat4.append(np.mean(table_welfarestat4_temp))

        # Table lists for average actions
        table_lmean.append(np.mean(table_lmean_temp))
        table_ymean.append(np.mean(table_ymean_temp))
        table_xmean.append(np.mean(table_xmean_temp))
        table_bmean.append(np.mean(table_bmean_temp))
        table_qmean.append(np.mean(table_qmean_temp))

        # Table list for CM participation rate
        table_parti.append(np.mean(table_parti_temp))

        # Table lists for b.x and b.q
        table_xavg.append(np.mean(table_xavg_temp))
        table_qavg.append(np.mean(table_qavg_temp))

        # Table lists for b.x/y and b.q/y
        table_xveloCM.append(np.mean(table_xveloCM_temp))
        table_qveloCM.append(np.mean(table_qveloCM_temp))

        # Table lists for b.x/M(DM) and b.q/M(DM)
        table_xveloDM.append(np.mean(table_xveloDM_temp))
        table_qveloDM.append(np.mean(table_qveloDM_temp))

        # STORE averages from Monte Carlo sims -------> 5 percentile
        table_omega_five.append(np.percentile(table_omega_temp, 5))
        table_inflation_five.append(np.percentile(table_inflation_temp, 5))
        table_moneystat0_five.append(np.percentile(table_moneystat0_temp, 5))
        table_moneystat1_five.append(np.percentile(table_moneystat1_temp, 5))
        table_moneystat2_five.append(np.percentile(table_moneystat2_temp, 5))
        table_moneystat3_five.append(np.percentile(table_moneystat3_temp, 5))
        table_moneystat4_five.append(np.percentile(table_moneystat4_temp, 5))
        table_moneystat5_five.append(np.percentile(table_moneystat5_temp, 5))
        table_welfarestat0_five.append(np.percentile(table_welfarestat0_temp,5))
        table_welfarestat1_five.append(np.percentile(table_welfarestat1_temp,5))
        table_welfarestat2_five.append(np.percentile(table_welfarestat2_temp,5))
        table_welfarestat3_five.append(np.percentile(table_welfarestat3_temp,5))
        table_welfarestat4_five.append(np.percentile(table_welfarestat4_temp,5))

        # Table lists for average actions
        table_lmean_five.append(np.percentile(table_lmean_temp, 5))
        table_ymean_five.append(np.percentile(table_ymean_temp, 5))
        table_xmean_five.append(np.percentile(table_xmean_temp, 5))
        table_bmean_five.append(np.percentile(table_bmean_temp, 5))
        table_qmean_five.append(np.percentile(table_qmean_temp, 5))

        # Table list for CM participation rate
        table_parti_five.append(np.percentile(table_parti_temp, 5))

        # Table lists for b.x and b.q
        table_xavg_five.append(np.percentile(table_xavg_temp, 5))
        table_qavg_five.append(np.percentile(table_qavg_temp, 5))

        # Table lists for b.x/y and b.q/y
        table_xveloCM_five.append(np.percentile(table_xveloCM_temp, 5))
        table_qveloCM_five.append(np.percentile(table_qveloCM_temp, 5))

        # Table lists for b.x/M(DM) and b.q/M(DM)
        table_xveloDM_five.append(np.percentile(table_xveloDM_temp, 5))
        table_qveloDM_five.append(np.percentile(table_qveloDM_temp, 5))

        # STORE averages from Monte Carlo sims -------> 95 percentile
        table_omega_ninefive.append(np.percentile(table_omega_temp,95))
        table_inflation_ninefive.append(np.percentile(table_inflation_temp,95))
        table_moneystat0_ninefive.append(np.percentile(table_moneystat0_temp,95))
        table_moneystat1_ninefive.append(np.percentile(table_moneystat1_temp,95))
        table_moneystat2_ninefive.append(np.percentile(table_moneystat2_temp,95))
        table_moneystat3_ninefive.append(np.percentile(table_moneystat3_temp,95))
        table_moneystat4_ninefive.append(np.percentile(table_moneystat4_temp,95))
        table_moneystat5_ninefive.append(np.percentile(table_moneystat5_temp,95))
        table_welfarestat0_ninefive.append(np.percentile(table_welfarestat0_temp,95))
        table_welfarestat1_ninefive.append(np.percentile(table_welfarestat1_temp,95))
        table_welfarestat2_ninefive.append(np.percentile(table_welfarestat2_temp,95))
        table_welfarestat3_ninefive.append(np.percentile(table_welfarestat3_temp,95))
        table_welfarestat4_ninefive.append(np.percentile(table_welfarestat4_temp,95))

        # Table lists for average actions
        table_lmean_ninefive.append(np.percentile(table_lmean_temp,95))
        table_ymean_ninefive.append(np.percentile(table_ymean_temp,95))
        table_xmean_ninefive.append(np.percentile(table_xmean_temp,95))
        table_bmean_ninefive.append(np.percentile(table_bmean_temp,95))
        table_qmean_ninefive.append(np.percentile(table_qmean_temp,95))

        # Table list for CM participation rate
        table_parti_ninefive.append(np.percentile(table_parti_temp,95))

        # Table lists for b.x and b.q
        table_xavg_ninefive.append(np.percentile(table_xavg_temp,95))
        table_qavg_ninefive.append(np.percentile(table_qavg_temp,95))

        # Table lists for b.x/y and b.q/y
        table_xveloCM_ninefive.append(np.percentile(table_xveloCM_temp,95))
        table_qveloCM_ninefive.append(np.percentile(table_qveloCM_temp,95))

        # Table lists for b.x/M(DM) and b.q/M(DM)
        table_xveloDM_ninefive.append(np.percentile(table_xveloDM_temp,95))
        table_qveloDM_ninefive.append(np.percentile(table_qveloDM_temp,95))

        ############ SAVE EACH FILE!!!!
        # Y-axis variable
        ydata_crosssection = { 'omega' :   [ table_omega_temp[1::]            ,
                                'Nominal Wage (Normalized), $\omega$' ]  ,
                'CMpart':   [   table_parti_temp[1::]            ,
                                'Participation Rate (status = 2, CM)' ]  ,
                'lmean' :   [   table_lmean_temp[1::],
                                "Labor," +r"$l^{\star}$"                   \
                                + " (status = 2, CM)"                    ] ,
                'ymean' :   [   table_ymean_temp[1::],
                                "Saving (with transfer)," +r"$y^{\star}$"    \
                                + " (status = 2, CM)"                    ] ,
                'xmean' :   [   table_xmean_temp[1::]            ,
                                'Avg. Buyer Spend, '+ r"$\bar{x}^{\star}$" \
                                + " (statu s= 1, DM-traded)"              ],
                'bmean' :   [   table_bmean_temp[1::],
                                'Avg. First-round Match, ' + r"$\bar{b}^{\star}$" \
                                + " (status = 1, DM-traded)"              ],
                'qmean' :   [   table_qmean_temp[1::],
                                'Avg. Consumption, ' + r"$\bar{q}^{\star}$" \
                                + " (status = 1, DM-traded)"              ],
                'bx' :   [   table_xavg_temp[1::],
                                'Avg. x, ' + r"$\bar{b}^{\star}\bar{x}^{\star}$" \
                                + " (status = 1, DM-traded)"              ],
                'bq' :   [   table_qavg_temp[1::],
                                'Avg. q, ' + r"$\bar{b}^{\star}\bar{q}^{\star}$" \
                                + " (status = 1, DM-traded)"              ],
                'xveloCM' : [   table_xveloCM_temp[1::],
                                'Avg. bx/y, '  \
                                + " (status = 1, DM-traded)"              ],
                'qveloCM' : [   table_qveloCM_temp[1::],
                                'Avg. bq/y, '  \
                                + " (status = 1, DM-traded)"              ],
                'xveloDM' : [   table_xveloDM_temp[1::],
                                'Avg. bx/m, '  \
                                + " (status = 1, DM-traded)"              ],
                'qveloDM' : [   table_qveloDM_temp[1::],
                                'Avg. bq/m, '  \
                                + " (status = 1, DM-traded)"              ],
                'money-median' : [    table_moneystat0_temp[1::]   ,
                                    'Median (Money Distro)' ] ,
                'money-mean' : [    table_moneystat1_temp[1::]   ,
                                    'Mean (Money Distro)' ] ,
                'money-skew' : [    table_moneystat3_temp[1::]   ,
                                    'Skewness (Money Distro)' ],
                'money-gini' : [    table_moneystat4_temp[1::],
                                    'Gini (Money Distro)'     ],
                'inv_velocity' : [    table_moneystat5_temp[1::],
                                    'Real Balance to GDP'     ],
                'cev-mean' : [  table_welfarestat4_temp[1::],
                                    'CEV (Welfare) %'          ],
                }

        if (SAVE_RESULTS==True):
            filename = outfilename + '-monty-ydata' + '.npy'
            print("\nNow saving results to " + filename)
            np.save(filename, ydata_crosssection)



        # Plot functions and time series
        #if (DO_FUNCTION_PLOTS == True):
        if (DO_FUNCTION_PLOTS==True) and (experiment==0 or experiment==N_eqm-1):
            print('\n\n\tNow plotting and saving to EPS ... Value functions')
            plt = klmod.DrawValuefun(V,B,W,Vtilde,Vfit,\
                    lottery_supports,lottery_payoffs, PlotSeparate=False)
            plt.savefig(figure_directory+'values.eps')
            plt.savefig(figure_directory+'values.png')    # Table of welfare - moments
            # table_welfarestat_relative4 = \
            # [(i/table_welfarestat4[1]-1.0)*100.0 for i in table_welfarestat4]
            # Table of welfare - moments
            tau_array = np.asarray(params)[:,2]
            idx_tau_before = tau_array.tolist().index(0.0) # location of tau = 0.0
            table_welfarestat_relative4 = \
            [(i/table_welfarestat4[idx_tau_before]-1.0)*100.0 for i in table_welfarestat4]

            print('\n\n\tNow plotting and saving to EPS ... Decision functions')
            policies = [lstar, bstar, qstar, xstar]
            policynames = ['$l^{*}$', '$b^{*}$', '$q^{*}$', '$x^{*}$']
            plt = klmod.DrawPolicyfun(policies, policynames, lottery_supports)
            plt.savefig(figure_directory+'policies.eps')
            plt.savefig(figure_directory+'policies.png')

            print('\n\n\tNow plotting and saving to EPS ... Distribution')
            plt = klmod.DrawDistro(simhistory['m'], lottery_supports, bins=30)
            plt.savefig(figure_directory+'distro.eps')
            plt.savefig(figure_directory+'distro.png')

            print('\n\n\tNow plotting and saving to EPS ... Simulations')
            start = 5000
            stop = 5200
            # Loop to plot Simulations
            ydata={ 'sims_m'       :   [ 'm'      , '$m$' ]  ,
                    'sims_y'       :   [ 'y'      , '$y$' ]  ,
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

            # Close all current loop's matplotlib objects to save memory!
            #plt.clf()

# End for loop

# Create tables after looping over all steady state EXPERIMENTS
if (rank == 0):
    header =  ["Steady State " + str(ex) for ex in range(len(params))]

    # Table on allocation outomes
    if (DO_TABLES):
        print("\n=====================================================")
        print("\nSteady state equilibrium: Allocations")
        print("\n-----------------------------------------------------")
    headers_alloc = ["Mean"] + header
    table_alloc = [ table_omega,
                    table_inflation,
                    table_ymean,
                    table_lmean,
                    table_xmean,
                    table_bmean,
                    table_qmean,
                    table_parti,
                    table_xavg,
                    table_qavg,
                    table_xveloCM,
                    table_qveloCM,
                    table_xveloDM,
                    table_qveloDM,
                     ]
    # Add to first column of table some labels
    table_alloc[0][0:0] = ["$\omega$"]
    table_alloc[1][0:0] = ["Inflation"]
    table_alloc[2][0:0] = ["CM part."]
    table_alloc[3][0:0] = ["$l(m)$"]
    table_alloc[4][0:0] = ["$y(m)$"]
    table_alloc[5][0:0] = ["$x(m)$"]
    table_alloc[6][0:0] = ["$b(m)$"]
    table_alloc[7][0:0] = ["$q(m)$"]
    table_alloc[8][0:0] = ["$bx$"]
    table_alloc[9][0:0] = ["$bq$"]
    table_alloc[10][0:0] = ["$bx/y$"]
    table_alloc[11][0:0] = ["$bq/y$"]
    table_alloc[12][0:0] = ["$bx/m_{DM}$"]
    table_alloc[13][0:0] = ["$bq/m_{DM}$"]
    # Print table
    if (DO_TABLES):
        print(tabulate(table_alloc, headers=headers_alloc, tablefmt='rst',\
                                                            floatfmt=".4f"))

    # Table on money Distribution
    if (DO_TABLES):
        print("\n=====================================================")
        print("\nSteady state equilibrium: Money Distribution")
        print("\n-----------------------------------------------------")
    headers_mdist = ["Statistic"] + header
    table_mdist = [ table_moneystat0,
                    table_moneystat1,
                    table_moneystat2,
                    table_moneystat3,
                    table_moneystat4,
                    table_moneystat5 ]
    # Add to first column of table some labels
    table_mdist[0][0:0] = ["Median"]
    table_mdist[1][0:0] = ["Mean"]
    table_mdist[2][0:0] = ["Std"]
    table_mdist[3][0:0] = ["Skewness"]
    table_mdist[4][0:0] = ["Gini"]
    table_mdist[5][0:0] = ["Inverse Velocity"]
    if (DO_TABLES):
        print(tabulate(table_mdist, headers=headers_mdist, tablefmt='rst',\
                                                            floatfmt=".4f"))

    # Table of welfare - moments
    tau_array = np.asarray(params)[:,2]
    idx_tau_before = tau_array.tolist().index(tau_before) # location of tau = 0.0
    idx_tau_after = tau_array.tolist().index(tau_after)
    table_welfarestat_relative4 = \
    [ -(i/table_welfarestat4[idx_tau_before]-1.0)*100.0 for i in table_welfarestat4 ]

    # 5% and 95% percentiles
    table_welfarestat_relative4_five = \
    [ -(i/table_welfarestat4_five[idx_tau_before]-1.0)*100.0 for i in table_welfarestat4_five ]

    table_welfarestat_relative4_ninefive = \
    [ -(i/table_welfarestat4_ninefive[idx_tau_before]-1.0)*100.0 for i in table_welfarestat4_ninefive ]

    if (DO_TABLES):
        print("\n=====================================================")
        print("\nSteady state equilibrium: Welfare Statistics")
        print("\n-----------------------------------------------------")
    headers_Vdist = ["Statistic"] + header
    table_Vdist = [ table_welfarestat0,
                    table_welfarestat1,
                    table_welfarestat2,
                    table_welfarestat3,
                    table_welfarestat_relative4 ]
    # Add to first column of table some labels
    table_Vdist[0][0:0] = ["Median"]
    table_Vdist[1][0:0] = ["Mean"]
    table_Vdist[2][0:0] = ["Std"]
    table_Vdist[3][0:0] = ["Skewness"]
    table_Vdist[4][0:0] = ["Mean CEV (%)"]
    if (DO_TABLES):
        print(tabulate(table_Vdist, headers=headers_Vdist, tablefmt='rst',\
                                                            floatfmt=".4f"))

# ------------- FIGURES REPRESENTATION OF TABLES ------------------------
if (rank == 0):
    ## ==============================================================
    ## SAVE FIGURES HERE!
    #figdir = rootdir + 'figures/'
    figdir = figure_directory
    ## ==============================================================

    # X-axis variable
    if PLOT_xaxis_TAU == True:
        xkey = 'tau'
        xdata={xkey : [ np.asarray(params)[:,2], 'Money Growth Rate, ' + r"$\tau$" ] }
    else:
        xkey = 'inflation'
        xdata={ xkey : [ table_inflation[1::], 'Inflation' ] }

    # Y-axis variable
    ydata={ 'omega' :   [   table_omega[1::]            ,
                            r'Nominal Wage (Normalized), $\omega$' ]  ,
            'CMpart':   [   table_parti[1::]            ,
                            r'Participation Rate (status = 2, CM)' ]  ,
            'lmean' :   [   table_lmean[1::],
                            r"Labor," +r"$l^{\star}$"                   \
                            + " (status = 2, CM)"                    ] ,
            'ymean' :   [   table_ymean[1::],
                            r"Saving (with transfer)," +r"$y^{\star}$"    \
                            + " (status = 2, CM)"                    ] ,
            'xmean' :   [   table_xmean[1::]            ,
                            'Avg. Buyer Spend, '+ r"$\bar{x}^{\star}$" \
                            + " (status= 1, DM-traded)"              ],
            'bmean' :   [   table_bmean[1::],
                            r'Avg. First-round Match, ' + r"$\bar{b}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'qmean' :   [   table_qmean[1::],
                            r'Avg. Consumption, ' + r"$\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bx' :   [   table_xavg[1::],
                            r'Avg. x, ' + r"$\bar{b}^{\star}\bar{x}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bq' :   [   table_qavg[1::],
                            r'Avg. q, ' + r"$\bar{b}^{\star}\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'xveloCM' : [   table_xveloCM[1::],
                            'Avg. bx/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloCM' : [   table_qveloCM[1::],
                            'Avg. bq/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'xveloDM' : [   table_xveloDM[1::],
                            'Avg. bx/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloDM' : [   table_qveloDM[1::],
                            'Avg. bq/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'money-median' : [    table_moneystat0[1::]   ,
                                'Median (Money Distro)' ] ,
            'money-mean' : [    table_moneystat1[1::]   ,
                                'Mean (Money Distro)' ] ,
            'money-skew' : [    table_moneystat3[1::]   ,
                                'Skewness (Money Distro)' ],
            'money-gini' : [    table_moneystat4[1::],
                                'Gini (Money Distro)'     ],
            'inv_velocity' : [    table_moneystat5[1::],
                                'Real Balance to GDP'     ],
            'cev-mean' : [  table_welfarestat_relative4[1::],
                                'CEV (Welfare) %'          ],
            }

    ydata_five={ 'omega' :   [   table_omega_five            ,
                            'Nominal Wage (Normalized), $\omega$' ]  ,
            'CMpart':   [   table_parti_five           ,
                            'Participation Rate (status = 2, CM)' ]  ,
            'lmean' :   [   table_lmean_five,
                            "Labor," +r"$l^{\star}$"                   \
                            + " (status = 2, CM)"                    ] ,
            'ymean' :   [   table_ymean_five,
                            "Saving (with transfer)," +r"$y^{\star}$"    \
                            + " (status = 2, CM)"                    ] ,
            'xmean' :   [   table_xmean_five            ,
                            'Avg. Buyer Spend, '+ r"$\bar{x}^{\star}$" \
                            + " (statu s= 1, DM-traded)"              ],
            'bmean' :   [   table_bmean_five,
                            'Avg. First-round Match, ' + r"$\bar{b}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'qmean' :   [   table_qmean_five,
                            'Avg. Consumption, ' + r"$\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bx' :   [   table_xavg_five,
                            'Avg. x, ' + r"$\bar{b}^{\star}\bar{x}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bq' :   [   table_qavg_five,
                            'Avg. q, ' + r"$\bar{b}^{\star}\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'xveloCM' : [   table_xveloCM_five,
                            'Avg. bx/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloCM' : [   table_qveloCM_five,
                            'Avg. bq/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'xveloDM' : [   table_xveloDM_five,
                            'Avg. bx/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloDM' : [   table_qveloDM_five,
                            'Avg. bq/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'money-median' : [    table_moneystat0_five   ,
                                'Median (Money Distro)' ] ,
            'money-mean' : [    table_moneystat1_five   ,
                                'Mean (Money Distro)' ] ,
            'money-skew' : [    table_moneystat3_five   ,
                                'Skewness (Money Distro)' ],
            'money-gini' : [    table_moneystat4_five,
                                'Gini (Money Distro)'     ],
            'inv_velocity' : [    table_moneystat5_five,
                                'Real Balance to GDP'     ],
            'cev-mean' : [  table_welfarestat_relative4_five,
                                'CEV (Welfare) %'          ],
            }

    ydata_ninefive={ 'omega' :   [   table_omega_ninefive           ,
                            'Nominal Wage (Normalized), $\omega$' ]  ,
            'CMpart':   [   table_parti_ninefive            ,
                            'Participation Rate (status = 2, CM)' ]  ,
            'lmean' :   [   table_lmean_ninefive,
                            "Labor," +r"$l^{\star}$"                   \
                            + " (status = 2, CM)"                    ] ,
            'ymean' :   [   table_ymean_ninefive,
                            "Saving (with transfer)," +r"$y^{\star}$"    \
                            + " (status = 2, CM)"                    ] ,
            'xmean' :   [   table_xmean_ninefive            ,
                            'Avg. Buyer Spend, '+ r"$\bar{x}^{\star}$" \
                            + " (statu s= 1, DM-traded)"              ],
            'bmean' :   [   table_bmean_ninefive,
                            'Avg. First-round Match, ' + r"$\bar{b}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'qmean' :   [   table_qmean_ninefive,
                            'Avg. Consumption, ' + r"$\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bx' :   [   table_xavg_ninefive,
                            'Avg. x, ' + r"$\bar{b}^{\star}\bar{x}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'bq' :   [   table_qavg_ninefive,
                            'Avg. q, ' + r"$\bar{b}^{\star}\bar{q}^{\star}$" \
                            + " (status = 1, DM-traded)"              ],
            'xveloCM' : [   table_xveloCM_ninefive,
                            'Avg. bx/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloCM' : [   table_qveloCM_ninefive,
                            'Avg. bq/y, '  \
                            + " (status = 1, DM-traded)"              ],
            'xveloDM' : [   table_xveloDM_ninefive,
                            'Avg. bx/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'qveloDM' : [   table_qveloDM_ninefive,
                            'Avg. bq/m, '  \
                            + " (status = 1, DM-traded)"              ],
            'money-median' : [    table_moneystat0_ninefive   ,
                                'Median (Money Distro)' ] ,
            'money-mean' : [    table_moneystat1_ninefive  ,
                                'Mean (Money Distro)' ] ,
            'money-skew' : [    table_moneystat3_ninefive  ,
                                'Skewness (Money Distro)' ],
            'money-gini' : [    table_moneystat4_ninefive,
                                'Gini (Money Distro)'     ],
            'inv_velocity' : [    table_moneystat5_ninefive,
                                'Real Balance to GDP'     ],
            'cev-mean' : [  table_welfarestat_relative4_ninefive,
                                'CEV (Welfare) %'          ],
            }

    start_eqcompare = 0
    stop_eqcompare = len(params)
    if (SAVE_RESULTS==True) and (num_cpus > 1):
        print("\nNow saving results to .NPY file ...")
        batch_out = {   'x': xdata,
                        'y': ydata,
                        'y5': ydata_five,
                        'y95': ydata_ninefive
                    }
        filename = homename + '/' + batch + '.npy'
        np.save(filename, batch_out)

    if (num_cpus==1) and (DO_PLOTS==True):

        # Load saved results on xdata and ydata
        filename = homename + '/' + batch + '.npy'
        batch_out = np.load(filename, encoding='latin1').item()
        xdata = batch_out['x']
        ydata = batch_out['y']
        ydata_five = batch_out['y5']
        ydata_ninefive = batch_out['y95']

        for idx_fig, key in enumerate(ydata.keys()):
            xtemp = xdata[xkey][0][start_eqcompare:stop_eqcompare]
            ytemp = ydata[key][0][start_eqcompare:stop_eqcompare]
            ytemp_five = ydata_five[key][0][start_eqcompare:stop_eqcompare]
            ytemp_ninefive = ydata_ninefive[key][0][start_eqcompare:stop_eqcompare]

            # Sequence of equilibria (patch: 5%-95% CI of Hsim paths distro)
            plt.fill_between(xtemp, ytemp_five, ytemp_ninefive,
                             facecolor='white', alpha=0.0, interpolate=True)
            # # TEMPORARY-FIX .......
            # plt.fill_between(xtemp[1::], ytemp_five, ytemp_ninefive,
            #                  facecolor='gray', alpha=0.238, interpolate=True)
            # # TEMPORARY-FIX .......

            # Sequence of equilibria (each: mean of Hsim paths pe equilibrium)
            plt.plot(xtemp, ytemp, 'o')

            # Before
            plt.plot(xdata[xkey][0][idx_tau_before],
                            ydata[key][0][idx_tau_before], 'dg')
            # After
            plt.plot(xdata[xkey][0][idx_tau_after],
                            ydata[key][0][idx_tau_after], 'sr')
            # Axes labels
            plt.xlabel(xdata[xkey][1])
            plt.title(ydata[key][1])
            if SAVE_FIGURES == True:
                plt.savefig(figdir + key + '.eps')
                plt.savefig(figdir + key + '.png')
            # Close all pipelines to figures, save memory
            plt.clf()
            #plt.show(block=False)
# Save each run's RESULTS to an .NPY file ---------------------------------
# if (SAVE_RESULTS==True) and (rank == 0):
#     print("\nNow saving results to .NPY file ...")
#     batch_out = { 'x': xdata, 'y': ydata }
#     filename = homename + '/' + batch + '.npy'
#     np.save(filename, batch_out)

if (rank == 0):
    print("\n\n\nView figures in: " + figdir)
    print("\nFinishing up ... GOODBYE!\t\t\t")
    if num_cpus > 1:
        # Send an automatic email to subscribers: see SEND.PY
        if SENDMAIL:
            send.message(rootdir, custom_message="COMPLETED > ", \
                                                        toaddr=email_list )
        # Kill all processes and exit Python
        MPI.COMM_WORLD.Abort()
