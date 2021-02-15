#!/usr/bin/env python

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
import scipy.stats as stats
import pandas as pd

#-- Custom Classes and Modules ----
from cssegmod import basemod as model   # Model class
from set_batch_parameters import *      # Define parameter combinations here
from setoutpaths import *               # Define output path for saving files
import send                             # for sending auto email

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

### VERSION: 29 July, 08:55 ===================
# UbarDM = 480
# chi = 0.079
# kappa = 0.025

# UbarDM = 480
# chi = 0.031 # 0.0271
# kappa = 0.025

# UbarDM = 406.0 # 405.02462 # 411.006637500
# chi = 0.03 # 0.0271
# kappa = 0.00399765 # 0.003601745 # 0.0036

CALIBRATE = 'tokyo'
# V Calibration: JS, 2018-08-07 ------------------------------------------------
# UbarDM = 406.1114 # 406.0
# chi = 0.020123432746 #0.03
# kappa = 0.005 # 0.00399765
# UbarDM = 404.25 # 406.0
# chi = 0.032
# kappa = 0.00399765 # 0.00399765
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

#Free PARAMETERS
A = 1.0
sigma = 2.0

# Perturbation about benchmark model to compute M1/GDP vs i, slope -> ARC-elasticity
epsi = 0.001

# ================= USER DEFINE EXPERIMENT SETTINGS HERE ================
# Do you want to bug your coauthors with email notifications?
SENDMAIL = False

# x-axis variable (TAU or INFLATION)
PLOT_xaxis_TAU = True

# Specify email receipients here:
email_list = ["tcy.kam@gmail.com", "junsang.skku@gmail.com"]

# Set BATCH experiment ID here:

# homename = 'resultscompare_zeroborrow'
homename = 'calibration-ds1'
rootdir = homename + '/'

# ****************************************************************************
# Borrowing Constraint Assumption: shortsell =
# True (natural limit)
# False (Zero borrowing limit)
# None (ad-hoc +ve limit)
shortsell = True
# ****************************************************************************

# For sampling from agent's Monte Carlo history
Tsim = 5000
start = 0           # Start equilibrium economy
stop = 79           # End equilibrium economy

# ================= USER DEFINE BATCH EXPERIMENT SETTINGS HERE ================

# Close any open FIGURES
plt.close("all")

# MPI rank and cores:
rank = MPI.COMM_WORLD.rank
num_cpus = MPI.COMM_WORLD.size

# Custom setting for defining grids in self.m_grid
ncpu_custom = num_cpus

# Load previous results?
LOAD_OLD_init = False
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
    ncpu_custom = 24 # NCPU-divisible, last OpenMPI run's NCPU setting
    HOMOTOPY = False
    FRESH_GUESS = False

if num_cpus > 1 and rank == 0:
    emailstart = "STARTED > "
    # Send an automatic email to subscribers: see SEND.PY
    if SENDMAIL:
        send.message( rootdir, custom_message=emailstart, toaddr=email_list )

# Approximation scheme -- see def Interp1d in src/mssmod9b.py
Interpolant = 'slinear'

# Consider 2 economies to get a linear fit of money demand relation
# tau_intercept = 0.0
tau_list = [ tau_data - epsi, tau_data, tau_data + epsi ]

# Storage for loop results
result_lmean = []
result_ivelocity = []

# loop over steady states ...
for experiment, tau in enumerate(tau_list):

    ## =======================================================================#
    # Instatiate MSS Model Class with approx. grid size 100. Defaults built in:
    klmod = model(  BETA =beta, UbarCM=1.0, UbarDM=UbarDM, KAPPA = kappa,
                CHIprop = chi, ALPHA = 0.00, A = A,
                SIGMA_CM = sigma, SIGMA_DM = 1.01, TAU = tau,
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
                Tsim=15000,         # length of monte carlo sims
                DirtyHack=True,    # check exist lotteries up TOL2 Precision
                InterpMethod=Interpolant, # basis function class, interpolant
                TESTMODE=False,     # Developer's troubleshoot setting
                ShortSell = shortsell, # Assumption on borrowing, fixed cost CHI
                ymax_adhoc = 0.00, # Relevant only to ad-hoc borrowing limit
                NCPU = ncpu_custom, RANK = rank
            )


    # Define paths ------------------------------------------------------------
    # Figures stored in FIGURES/EXPERIMENTxx/
    # rootdir = 'results/compare_alpha_HI/'
    figure_directory, outfilename, previous_outfilename, rootdir =\
                                            filefind(experiment,rootdir=rootdir)


    # Begin stopwatch --------------------------------------------------------
    tic = time.time()

    # Messages and Information for User ---------------------------------------
    if (rank == 0):
        print("\nRunning Experiment %i of %i"  %(experiment+1, len(tau_list)) )
        print("\nSHORT SELL? %s" % (str(shortsell)) )
        print("\n\tchi=%0.9f, CHI=%0.6f, A=%0.3f, UbarCM=%0.6f, UbarDM=%0.9f, TAU=%0.6f, KAPPA=%0.9f" % (chi, klmod.CHI, A, klmod.UbarCM, klmod.UbarDM, klmod.TAU, klmod.KAPPA))
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
            LOAD_OLD = LOAD_OLD_init
            HOMOTOPY = False
            SAVE_RESULTS = True
        elif experiment >= 1:
            HOMOTOPY = True
            LOAD_OLD = HOMOTOPY

        if rank == 0:
            print("\n\t*** HOMOTOPY is set to " + str(HOMOTOPY) + " ***\n")

    # Solve instance of parameterized model. Get output as dictionary RESULTS:
    results = klmod.SolveSteadyState(   outfilename, previous_outfilename,
                                        LOAD_OLD=LOAD_OLD,
                                        HOMOTOPY=HOMOTOPY,
                                        FRESH_GUESS=FRESH_GUESS,
                                        SAVE_RESULTS=SAVE_RESULTS,
                                        STORE_APPROXSEQ = STORE_APPROXSEQ
                                    )

    # Save each run's RESULTS to an .NPY file ---------------------------------
    if (rank == 0):
        lmean, ivelocity = klmod.modelstats(results)
        result_lmean.append(lmean)
        result_ivelocity.append(ivelocity)

        if (SAVE_RESULTS==True):
            print("\nNow saving results to .NPY file ...\n\n\n")
            filename = outfilename + '.npy'
            np.save(filename, results)


# Data's statistics or auxiliary model
labor_data_mean = 1.0 / 3.0
md_data_mean1 = 0.175644 # level average
#0.20431645  # Using M1 velocity data and GDP
                                # deflator Inflation
md_data_slope1 = -2.79678529#-3.32 # Using log(inflation) on x-axis
#-2.79678529

# md_data_mean2 = 0.56374462  # Using M2 velocity data and GDP
#                                  # deflator Inflation

# --------------------------------------------------------------------------
# USING THIS VERSION NOW: 21/07/2018
#
# Lucas-Nicolini, 1915-2007, 3rd order spline fit of data
# if beta_freq == 'Q':
#     md_data_mean2 = 0.147899*4.0   # Data mean: New_M1/GDP at mean Tbill rate
# elif beta_freq == 'A':
md_data_mean2 = 0.147899
md_data_slope2 = 'NA'
md_data_elast2 = -1.1279 # From auxiliary split fit model on 1915-2007 data

# Save each run's RESULTS to an .NPY file ---------------------------------
# Compute calibration's implied stats or auxiliary model
# Model's inflation rate = data's mean
idx_tau_data = tau_list.index(tau_data)
# Model's average labor supply
labor_mean = result_lmean[idx_tau_data]
# Model's M1/GDP
md_model_mean = result_ivelocity[idx_tau_data]
rise = result_ivelocity[idx_tau_data+1] - result_ivelocity[idx_tau_data-1]
step = tau_list[idx_tau_data+1] - tau_list[idx_tau_data-1]
# Slope wrt pi
md_slope = float(rise/step)  # Chi

# Slope wrt i (arc, mean)
# i_right = (1.0 + tau_list[idx_tau_data+1])/klmod.BETA - 1.0
# i_left = (1.0 + tau_list[idx_tau_data-1])/klmod.BETA - 1.0
step2 =  tau_list[idx_tau_data+1] - tau_list[idx_tau_data-1]
md_slope2 = float(rise/step2)*klmod.BETA

# Arc-Elasticity of M/GDP wrt i
i = (1.0 + tau_list[idx_tau_data])/klmod.BETA - 1.0
md_elast2 = md_slope2*i/md_model_mean
if beta_freq == 'Q':
    md_model_mean = md_model_mean/4.0 # Q --> A freq
    md_elast2 = (md_slope*16.)*beta_annual*i_data_annual/(md_model_mean)

# Let/Right derivative - FOR Trouble Shooting only (not reported)
# rise_left = result_ivelocity[idx_tau_data] - result_ivelocity[idx_tau_data-1]
# step_left = tau_list[idx_tau_data] - tau_list[idx_tau_data-1]
# md_slope_left = float(rise_left/step_left)
#
# rise_right = result_ivelocity[idx_tau_data+1] - result_ivelocity[idx_tau_data]
# step_right = tau_list[idx_tau_data+1] - tau_list[idx_tau_data]
# md_slope_right = float(rise_right/step_right)

# USING THIS VERSION NOW: 21/07/2018
# ---------------------------------------------------------------------------

rowlabels = ["Hours", "Mean M1/GDP", "Slope", "Elast. (M/GDP)-i"]
# datastat1 = [labor_data_mean, md_data_mean1, md_data_slope1]
datastat = [labor_data_mean, md_data_mean2, md_data_slope2, md_data_elast2]
modelstat =[labor_mean, md_model_mean, md_slope, md_elast2]
# modelstat2 =[labor_mean, md_model_mean, md_slope_left]
# modelstat3 =[labor_mean, md_model_mean, md_slope_right]

# Display to screen using Pandas table
# calibtab = {    "Aux. Statistic": rowlabels,
#                 "Data (M1)": datastat1,
#                 "Data (M2)": datastat2,
#                 "Model (Avg)": modelstat,
#                 "Model (Left)": modelstat2,
#                 "Model (Right)": modelstat3,
#                            }

calibtab = {    "Aux. Statistic": rowlabels,
                "Data (M1)": datastat,
                "Model (Avg)": modelstat,
                           }
# Create dataframe
df = pd.DataFrame(data=calibtab)
print(df)

# For Junsang to cut and past to Excel -->
# for statistic in range(len(modelstat)):
#     print(modelstat[statistic])
# For Junsang to cut and past to Excel <--

if (rank == 0):
    # Exit stage ...
    print("\nFinished computing equilibria ...\t\t\t")
    if num_cpus > 1:
        # Send an automatic email to subscribers: see SEND.PY
        if SENDMAIL:
            send.message(rootdir, custom_message="COMPLETED > ", \
                                                        toaddr=email_list )
        # Kill all processes and exit Python
        MPI.COMM_WORLD.Abort()
