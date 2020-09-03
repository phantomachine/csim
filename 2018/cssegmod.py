# %load_ext autoreload
# %autoreload 2

from __future__ import print_function, division, absolute_import
from textwrap import dedent
import numpy as np
from scipy.optimize import fminbound
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import pchip, Akima1DInterpolator
from scipy import interp
#from numba import jit, void, int_, double
from scipy.spatial import ConvexHull
import setops as setops
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
# from mpl_toolkits.axes_grid.axislines import Subplot
from mpl_toolkits.axisartist.axislines import Subplot
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
import itertools
import time
from mpi4py import MPI

""" CSSEGMOD.PY (v.1.1.0)
(c) 2017- T.Kam (tcy.kam@gmail.com) and J.Lee (junsanglee@skku.edu).

The GNU GPL v.3.0 terms (see LICENCE) apply everywhere to these codes and subsequent uses and modifications of them. Please cite this original source in all your work (i.e., your own paper and source codes) as:
    * Paper: T. Kam and J. Lee (2017): "TITLE OF PAPER", *Journal Details*
    * Code: github.com/phantomachine/cssegmart/

See CHANGELOG file for recent updates/bug-fix details.

"""

# Name the folder where you'd save your figures to:
figfolder = 'figures/'

class basemod(object):
    """CSSEGMART v.1.0 (with OpenMPI Parallelization via MPI4PY). Model primitives and methods """

    def __init__(self, BETA=0.9988, UbarDM = 480.0, UbarCM = 1.0,
                 ETA = 1.0, CHIprop = 0.079,
                 ALPHA = 0.0,
                 SIGMA_CM = 2.0, SIGMA_DM = 1.01,  KAPPA = 0.025,
                 A = 10.0, TAU = 0.05, DELTA = 0.0, RHO = 1.0, VARPI = 1.0,
                 m_min = 0.0,
                 m_gridsize_ltk = 20, m_gridsize_gtk = 60,
                 m_gridsize_fine=500,
                 m_gridsize_refine=10000,
                 MAXITER1 = 2000, MAXITER2 = 500, SMOOTH = 0.8, DirtyHack=False, TOL0 = 0.01, TOL1 = 0.00001,
                 TESTMODE =True,
                 TOL2 = 0.00001, Tsim = 1e5, InterpMethod='slinear',
                 ShortSell = True, ymax_adhoc = 0.025,
                 NCPU=1,RANK=0):
        self.BETA = BETA
        self.UbarDM = UbarDM        # DM utility scale UbarDM*u(q)
        self.UbarCM = UbarCM    # CM utility scale UbarCM*U(C)
        self.ETA = ETA
        self.A = A
        self.TAU = TAU
        self.DELTA = DELTA
        self.ALPHA = ALPHA
        self.SIGMA_CM = SIGMA_CM
        self.SIGMA_DM = SIGMA_DM
        self.RHO = RHO
        self.VARPI = VARPI
        self.KAPPA = KAPPA
        self.q_min = 1e-3
        self.C_min = 1e-3
        self.m_min = m_min
        # upper bound on m from preference assumption (2.3) in paper
        m_max = .99*self.invU_C_CM(A)
        self.m_max = m_max
        # Optimal C in CM derived from FoC in CM: quasilinear preferences
        Cstar = self.invU_C_CM(A)
        self.Cstar = Cstar
        # natural upper bound on l
        self.l_max = Cstar + self.invU_C_CM(A)
        # Fixed cost (labor units)
        self.CHI = CHIprop*m_max
        # Initial cardinality of action sets (rough guess)
        self.m_gridsize_fine = m_gridsize_fine
        # Cardinality of action sets (refined guess), later iterations
        self.m_gridsize_refine = m_gridsize_refine
        # Adjust grid partition {m : m < KAPPA} to have even CPU loads
        if m_gridsize_ltk >= NCPU:
            N_ltk = m_gridsize_ltk - (m_gridsize_ltk % NCPU)
        elif m_gridsize_ltk < NCPU:
            N_ltk = NCPU
            if RANK == 0:
                print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("\n\t\tWARNING: Default N_ltk to NCPU=%i grid points"
                                                                    % (N_ltk))
        self.m_gridsize_ltk = N_ltk
        if self.CHI <= KAPPA:
            mltk = self.grid1d(m_min,KAPPA,Npoints=N_ltk-1,method='linear')
            self.m_grid_ltk = np.sort(np.append(mltk, self.CHI))
        else:
            self.m_grid_ltk = self.grid1d(m_min,KAPPA,Npoints=N_ltk,method='linear')

        # Adjust grid partition {m : m > KAPPA} to have even CPU loads
        if m_gridsize_gtk >= NCPU:
            # Minus 1 to account for self.CHI point later
            N_gtk = m_gridsize_gtk - (m_gridsize_gtk % NCPU)
        elif m_gridsize_gtk < NCPU:
            # Scaling = 2
            # N_gtk = NCPU*Scaling
            N_gtk = 36
            if RANK == 0:
                print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print("\n\t\tWARNING: Default to N_gtk = %i x NCPUS = %i grids"
                                                            % (Scaling, N_gtk))
        # Gridpoints for m > KAPPA
        self.m_gridsize_gtk = N_gtk

        if self.CHI <= KAPPA:
            self.m_grid_gtk = self.grid1d(KAPPA+1e-8,m_max,Npoints=N_gtk)
        else:
            mgtk = self.grid1d(KAPPA+1e-8,m_max,Npoints=N_gtk-1)
            self.m_grid_gtk = np.sort(np.append(mgtk, self.CHI))
        # Endogenous (total grid points)
        m_grid = np.append(self.m_grid_ltk, self.m_grid_gtk)
        self.m_grid = np.sort(m_grid)
        self.m_gridsize = self.m_grid.size
        self.index_gtk = self.m_grid.tolist().index(KAPPA)+1 #loc. min(m>KAPPA)
        self.m_grid_gtk_fine = self.grid1d(KAPPA+1e-8, m_max,
                                Npoints=m_gridsize_fine)
        self.y_grid_fine = self.grid1d(self.m_min, m_max,
                                Npoints=m_gridsize_fine,method='linear')
        # Approx. settings
        self.TESTMODE = TESTMODE # Display options for TESTMODE
        self.TOL0 = TOL0     # %-precision, %-change omega convergence
        self.TOL1 = TOL1     # Convergences: Bellman, SME iteration
        self.TOL2 = TOL2     # Tolerance for how different V is from Vtilde
        self.MAXITER1 = MAXITER1
        self.MAXITER2 = MAXITER2
        self.SMOOTH = SMOOTH # Smoother in updating omega guesses
        self.DirtyHack = DirtyHack # Check for fake lotteries (approx errors)
        self.InterpMethod = InterpMethod # Basis function family
        self.T = Tsim # length of monte carlo simulation
        # Short sell constraint?
        self.shortsell = ShortSell
        self.ymax_adhoc = ymax_adhoc # Ad-hoc borrowing limit (ShortSell=None)

    def __repr__(self):
        m = "selfmod(BETA={BETA:g},UbarDM={UbarDM:g},ETA={ETA:g},KAPPA={KAPPA:g},"
        m += "TAU={TAU:g},DELTA={DELTA:g},"
        m += "A={A:g},CHI={CHI:g},ALPHA={ALPHA:g},SIGMA_CM={SIGMA_CM:g},"
        m += "SIGMA_DM={SIGMA_DM:g},RHO={RHO:g},VARPI={VARPI:g}"
        m += "m_max={m_max:g},m_min={m_min:g},m_gridsize={m_gridsize:g})"
        return m.format(BETA=self.BETA, UbarCM=self.UbarCM,
                        UbarDM=self.UbarDM, ETA=self.ETA,
                        KAPPA = self.KAPPA,A=self.A, CHI=self.CHI, ALPHA=self.ALPHA, SIGMA_CM=self.SIGMA_CM, SIGMA_DM=self.SIGMA_DM, RHO=self.RHO, VARPI = self.VARPI,
                        TAU=self.TAU, DELTA=self.TAU, m_min=self.m_min,
                        m_max=self.m_max, m_gridsize=self.m_gridsize)

    def __str__(self):
        m = """
            CSIM version of Menzio-Shi-Sun (Endogenous Tobin-Baumol Money Model):
            (c) 2016, T. Kam and J. Lee
              - BETA (discount factor)                     : {BETA:g}
              - UbarCM (CM scale payoff)                   : {UbarCM:g}
              - UbarDM (DM scale payoff)                   : {UbarDM:g}
              - ETA (convexity: labor supply disutility)   : {ETA:g}
              - KAPPA (trading post fixed cost/period)     : {KAPPA:g}
              - A (scale parameter A, leisure utility)     : {A:g}
              - CHI (fixed cost for going to CM)           : {CHI:g}
              - ALPHA (liquidity shock prob.)              : {ALPHA:g}
              - SIGMA_CM (risk aversion CM)                : {SIGMA_CM:g}
              - SIGMA_DM (risk aversion DM)                : {SIGMA_DM:g}
              - RHO (matching function concavity)          : {RHO:g}
              - VARPI (DM cost function convexity)         : {VARPI:g}
              - TAU (money supply growth rate, redefined)  : {TAU:g}
              - DELTA (proportional tax rate-deprecated)   : {DELTA:g}
              - m_min (Minimum real balance)               : {m_min:g}
              - m_max (Maximum real balance)               : {m_max:g}
              - m_gridsize (Number of grid points)         : {m_gridsize:g}
        """
        return dedent(m.format(BETA=self.BETA, UbarCM=self.UbarCM,
                                UbarDM=self.UbarDM, ETA=self.ETA,
                                A=self.A, CHI=self.CHI, ALPHA=self.ALPHA, SIGMA_CM=self.SIGMA_CM,
                                SIGMA_DM=self.SIGMA_DM, RHO=self.RHO, VARPI=self.VARPI, KAPPA = self.KAPPA,
                                TAU=self.TAU, DELTA=self.TAU, m_min=self.m_min,
                                m_max=self.m_max, m_gridsize=self.m_gridsize))

    ##-------- SYSTEM TOOLS ----------------------------------------------
    def StatusBar(self,iteration,iteration_max,stats1,stats2,flag,width=15):
        percent = float(iteration)/iteration_max
        sys.stdout.write("\r")
        progress = ""
        for i in range(width):
            if i <= int(width * percent):
                progress += "="
            else:
                progress += "-"
        sys.stdout.write(
            "[ %s ] %.2f%% %i/%i, error1 = %0.5f, error2 = %0.5f, %s    "
            % (progress,percent*100,iteration,iteration_max,stats1,stats2,flag)
            )
        sys.stdout.flush()

    def StatusBar_simple(self,iteration,iteration_max,message=None,width=15):
        percent = float(iteration)/iteration_max
        sys.stdout.write("\r")
        progress = ""
        for i in range(width):
            if i <= int(width * percent):
                progress += "*"
            else:
                progress += "."
        sys.stdout.write(
            "[ %s ] %.2f%% %i/%i | %s    "
            % (progress,percent*100,iteration,iteration_max,message)
            )
        sys.stdout.flush()

    ##-------- MAKE GRID TOOLS -------------------------------------------
    def grid1d(self, xmin, xmax, Npoints=100, method='linear'):
        """Create 1D grid points: linear, inverse ratio scale, or chebychev nodes. Default is linear: equally spaces gridpoints."""
        if method=='linear':
            y = np.linspace(xmin, xmax, Npoints)
        elif method=='inverse_ratio':
            xmin_log = np.log(xmin - xmin + 1.0)/np.log(10.0)
            xmax_log = np.log(xmax - xmin + 1.0)/np.log(10.0)
            y = np.logspace(xmin_log, xmax_log, Npoints) + (xmin - 1.0)
        elif method=='chebychev':
            m = np.arange(1,Npoints+1)
            x = np.cos( (2.0*m - 1.0)*np.pi / (2.0*Npoints) )
            constant = 0.5*(xmin + xmax)
            slope = 0.5*(xmax - xmin)
            y = constant + slope*x
        return y

    ##-------- MODEL PRIMITIVE FUNCTIONS -------------------------------

    # ---- CM preference and production functions -------
    def U_CM(self, C):
        """  Per-period Utility Function, U(C), for CM consumption C  """
        Cmin = 1e-3
        if self.SIGMA_CM == 1.0:
            u = np.log(C + Cmin)
            #U_gradient_q = (q + cmin)**(-1.0)
        elif (self.SIGMA_CM > 0.0) and (self.SIGMA_CM != 1.0):
            u = ((C+Cmin)**(1.0-self.SIGMA_CM) - Cmin**(1.0-self.SIGMA_CM))/(1.0-self.SIGMA_CM)
        return u*self.UbarCM

    def invU_CM(self, happiness):
        """  Inverse of Per-period Utility Function, U^-1(happiness) """
        Cmin = self.C_min
        if self.SIGMA_CM == 1.0:
            c = np.exp(happiness) - Cmin
            #U_gradient_q = (q + cmin)**(-1.0)
        elif (self.SIGMA_CM > 0.0) and (self.SIGMA_CM != 1.0):
            p = (1.0-self.SIGMA_CM)
            c = (p*happiness/self.UbarCM + Cmin**p)**(1/p) - Cmin
        return c

    def invU_C_CM(self, marginal_value):
        """Inverse of dU/dC function"""
        return (marginal_value/self.UbarCM)**(-1.0/self.SIGMA_CM) - self.C_min

    def h(self, labor):
        """ Cost (utility) of leisure function    """
        return self.A * (1.0-(1.0-labor)**self.ETA)

    def production_CM(self, labor):
        """ CM production function - linear in labor   """
        Y = labor
        return Y

    # ---- DM preference and technology functions -------
    def u_DM(self, q):
        """  Per-period Utility Function, u(q), for DM consumption q  """
        qmin = self.q_min
        if self.SIGMA_DM == 1.0:
            u = np.log(q + qmin)
            #U_gradient_q = (q + cmin)**(-1.0)
        elif (self.SIGMA_DM > 0.0) and (self.SIGMA_DM != 1.0):
            # u =  ((q+qmin)**(1.0-self.SIGMA_DM)-qmin**(1.0-self.SIGMA_DM))/(1.0-self.SIGMA_DM)
            u =  ((q+qmin)**(1.0-self.SIGMA_DM))/(1.0-self.SIGMA_DM) \
                                -(qmin**(1.0-self.SIGMA_DM))/(1.0-self.SIGMA_DM)
            # TEMPORARY CHECKS #########################
            # if (q+qmin).any() <= 0.0:
            #     print("q is negative!")
            # else:
            #     print(q+qmin)
            # TEMPORARY CHECKS #########################
        return u*self.UbarDM

    def invu_DM(self, happiness):
        """  Inverse of Per-period Utility Function, U^-1(happiness) """
        qmin = self.q_min
        if self.SIGMA_DM == 1.0:
            q = np.exp(happiness/self.UbarDM) - qmin
            #U_gradient_q = (q + cmin)**(-1.0)
        elif (self.SIGMA_DM > 0.0) and (self.SIGMA_DM != 1.0):
            p = (1.0-self.SIGMA_DM)
            q = (p*happiness/self.UbarDM + qmin**p)**(1/p) - qmin
        return q

    def invu_q_DM(self, marginal_value):
        """Inverse of dU/dq function"""
        return ((marginal_value/self.UbarDM)**(-1.0/self.SIGMA_DM) - self.q_min)

    def cost_DM(self, q):
        """ Cost of producing a unit of q """
        cost = q**self.VARPI
        return cost

    def invcost_DM(self, cost):
        """ Output in DM q, associated with given level of COST """
        q = cost**( 1.0/self.VARPI )
        return q

    # ---- Matching and submarket related functions -----
    def mu(self, b):
        """ Trading post's matching probability with buyer function, s, where
                            s = mu(b)
        """
        #if (b > 1.0) or (b < 0.0):
        #    sys.exit("ERROR: b = %6.3f. Must be a probability measure!" % (b))
        mub = ( 1.0 - b**self.RHO )**(1.0/self.RHO)
        return mub

    def invmu(self, kdxi):
        """ Buyer's matching probability as function of xi where,
                            xi = x - c[Q(b(x), x)]
                            kdxi = KAPPA / xi
            Exploits the equilibrium trading posts' zero-profit in DM condition for firms. Returns b(x).
        """
        bX = ( 1.0 - kdxi**self.RHO )**(1.0/self.RHO)
        return bX

    def Q(self, x, b):
        """ Equilibrium quantity q as function of trading post (x,b). Derived from equilibrium trading post zero profit condition. Assume (x,b) are NumPy arrays.
        """
        q = self.invcost_DM( x - self.KAPPA/self.mu(b) )
        q[x==0.0] = 0.0
        return q

    def Q_scalar(self, x, b):
        """ Equilibrium quantity q as function of trading post (x,b). Derived from equilibrium trading post zero profit condition. Assume (x,b) are scalar floats.
        """
        if (x==0.0) or (b==0.0):
            q = 0.0
        else:
            q = self.invcost_DM( x - self.KAPPA/self.mu(b) )
        return q

    # def kappafun(self, b):
    #     """ Composite function KAPPA/mu(b). Return: value of function and first derivative functions store in a dictionary
    #     """
    #     mu = self.mu(b)
    #     kap = self.KAPPA / mu      # kappafun(b)
    #     return kap
    #
    # def kappafun_inv(self, q):
    #     """k/mu(b) in units of quantity (labor), q"""
    #     return self.invcost_DM( 1.0 - self.KAPPA / q )

    def ymax(self, omega):
        """Natural credit limit on agents"""
        if self.TAU >= 0:
            ymax = np.minimum(self.m_max-self.TAU/omega, self.m_max)
        else:
            ymax = np.maximum(self.m_max-self.TAU/omega, self.m_max)
        return ymax

    ## --------------- METHODS -------------------------------------

    def MoneyGrowthFactor(self, omega=None):
        """Gross growth of money supply, which also equals nominal wage inflation (for steady state equilibrium only!)"""
        # Assumption in MSS
        #gamma = (1.0-self.DELTA)*(1.0 + self.TAU*omega)

        # Standard Constant money growth model
        gamma = 1.0 + self.TAU
        return gamma

    def InterpFun1d(self, xdata, ydata):
        """ Interpolate 1D functions given data points (xdata, ydata). Returns instance of class: funfit.
        NOTE: funfit.derivative() will provide the derivative functions of particular smooth approximant---depends on class of interpolating function. See SCIPY.INTERPOLATE subclasses for more detail.   """

        #xdata = np.sort(xdata)
        if self.InterpMethod=='slinear':
            funfit = spline(xdata, ydata, k=1) # B-spline 1st order
        elif self.InterpMethod=='squadratic':
            funfit = spline(xdata, ydata,  k=2) # instantiate B-spline interp
        elif self.InterpMethod=='scubic':
            funfit = spline(xdata, ydata, k=3) # instantiate B-spline interp
        elif self.InterpMethod=='squartic':
            funfit = spline(xdata, ydata, k=4) # instantiate B-spline interp
        elif self.InterpMethod=='squintic':
            funfit = spline(xdata, ydata,  k=5) # instantiate B-spline interp
        elif self.InterpMethod=='pchip':
            # Shape preserving Piecewise Cubic Hermite Interp Polynomial splines
            funfit = pchip(xdata, ydata)
        elif self.InterpMethod=='akima':
            funfit = Akima1DInterpolator(xdata, ydata)
        return funfit # instance at m point(s)

    def MoneyBalance(self, saving, omega):
        """End-of-period balance with new money (growth) transfer. This was denoted as the function $\phi()$ in the theory in the paper.
        NOTE: New assumption on Money Supply process:

            M(+1)/M = 1 + TAU,      BETA-1 < TAU < infty
        """
        gamma = self.MoneyGrowthFactor()
        mplus = (saving + self.TAU/omega) / gamma
        return mplus

    def BudgetConstraint(self, m, y, C, omega):
        """ CM Budget Constraint to get labor supply """
        # End-of-period balance with new money (growth) transfer
        mplus = self.MoneyBalance(y, omega)
        # Residual from budget constraint
        #labor = mplus - m + C                       # budget constraint
        labor = y - m + C
        return labor, mplus

    def WorkerPayoff(self, y, m, Vbar_fit, omega):
        """ Worker m lifetime utility at end-of-period real balance
            choices of (C, l, y). Worker :=: CM agent now!
            Vbar_fit: See Vbar def below, smooth and continuous approximant
        """
        # Optimal C is constant under quasilinear preferences
        #Cstar = self.invU_C_CM(self.A)                  # FoC w.r.t. C
        Cstar = self.Cstar
        labor, mplus = self.BudgetConstraint(m, y, Cstar, omega)
        # Negative for FMINBOUND rountine
        W_util = -(self.U_CM(Cstar) - self.h(labor) + self.BETA*Vbar_fit(mplus))
        return W_util

    def WorkerProblem(self, Vbar_fit, omega, SolMethod='continuous'):
        """ Worker's optimization problem given guess of ex-ante value
            function Vfit (V fitted by continuous bases).
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Know your onions!
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Based on MSSMOD.PY, new renovation with OpenMPI compute
            capability. Also, now this is CM with work, consumption and accumulation of money asset.
        """
        # ------------ Hello: ROOT 0 WORLD ... -------------------------
        COMM = MPI.COMM_WORLD         # instantiate MPI worlds
        num_cpus = COMM.size
        rank = COMM.rank
        # Shape of partition elements of root=0 domain
        Ngrid_m = self.m_grid.shape[0]
        nr = int(np.ceil(float(Ngrid_m)/num_cpus)*num_cpus)
        nc = 1
        # Domain partition element (slave's slice of job)
        block_size = nr / num_cpus
        # idxset_m_partition = np.zeros((block_size,nc),dtype='int')
        idxset_m_partition = np.arange(block_size, dtype='int')
        # CODOMAINs: Root 0's Gather destination(s) at the end
        W = np.empty(self.m_grid.shape,dtype='d')
        lstar = np.empty(self.m_grid.shape,dtype='d')
        ystar = np.empty(self.m_grid.shape,dtype='d')
        # BROADCAST: Arrays to workers
        if (rank == 0):
            BcastData = {   'self': self,
                            'Vbar_fit': Vbar_fit,
                            'omega': omega,
                            'method': SolMethod
                        }
            idxset_m = np.arange(Ngrid_m, dtype='int')
        else:
            BcastData = None
            idxset_m = np.zeros(self.m_grid.shape, dtype='int')

        # Scatter L_set to slaves each with L_partition
        COMM.Scatter([idxset_m,MPI.INT],[idxset_m_partition,MPI.INT])
        # ------------ Goodbye: ... ROOT 0 WORLD -----------------------
        #                       |
        #                       |
        # ------------ Hello: SLAVE WORLD ... --------------------------
        # Extract broadcasts from Master
        BcastData = COMM.bcast( BcastData, root=0 )
        self_local = BcastData['self']
        Vbar_fit_local = BcastData['Vbar_fit']
        omega_local = BcastData['omega']
        SolMethod = BcastData['method']
        # Local storage of local results
        W_local = np.empty(idxset_m_partition.shape,dtype='d')
        ystar_local = np.empty(idxset_m_partition.shape,dtype='d')
        lstar_local = np.empty(idxset_m_partition.shape,dtype='d')
        # Bounds on optimizer - y_ub depends on Money supply growth assumption!
        y_lb =self_local.m_min
        #y_ub =(1.0+self_local.TAU)*self_local.m_max - self_local.TAU/omega_local
        # Correction: 28-01-2017, upper bound on y.
        # if self_local.TAU > 0.0:
        #     # Case where positive inflation - legacy
        #     y_ub = np.minimum(self_local.m_max - self_local.TAU/omega_local, self_local.m_max)
        # else:
        #     # Case where negative inflation - added 10-07-2018
        #     y_ub = np.maximum(self_local.m_max - self_local.TAU/omega_local, self_local.m_max)
        y_ub = self_local.ymax(omega_local)

        # Optimal C under quasilinear preferences
        Cstar = self_local.Cstar
        # ---------------------------------------------------------------------
        # Two flavors ...
        if SolMethod == 'continuous':
            # Unidimensional optimization of continuous function
            for i, index_m in enumerate(idxset_m_partition):
                # Current state
                m = self_local.m_grid[index_m]
                                # Optimizer
                y_sol = fminbound(self_local.WorkerPayoff,
                                y_lb, y_ub, args=(m,Vbar_fit_local,omega_local))
                # Store value at optimizer
                W_local[i] = -self_local.WorkerPayoff(y_sol, m,
                                                    Vbar_fit_local, omega_local)
                ystar_local[i] = y_sol
                # Cstar = self_local.invU_C_CM(self_local.A)
                lstar_local[i] = self_local.BudgetConstraint(m,
                                                             y_sol, Cstar, omega_local)[0]
        elif SolMethod == 'discrete':
            # Table lookup problem
            for i, index_m in enumerate(idxset_m_partition):
                m = self_local.m_grid[index_m]
                # Fine grid on choices y
                y_grid_fine = self_local.grid1d(y_lb, y_ub,
                            Npoints=self_local.m_gridsize_fine,method='linear')
                # Evaluate total payoff at fine grid
                WY = self_local.WorkerPayoff(y_grid_fine, m,
                                                Vbar_fit_local, omega_local)[0]
                # Table lookup for Maximizer
                W_local[i] = -WY.min()
                y_sol = self_local.y_grid_fine[WY.argmin()]
                ystar_local[i] = y_sol
                lstar_local[i] = self_local.BudgetConstraint(m,
                                                             y_sol, Cstar, omega)
        # ---------- Goodbye: ... SLAVE WORLD --------------------------
        #                       |
        #                       |
        # ---------- Hello: ROOT 0 WORLD -------------------------------
        # Collect back to CODOMAINs in rank 0
        COMM.Gather([W_local,MPI.DOUBLE],[W,MPI.DOUBLE])
        COMM.Gather([lstar_local,MPI.DOUBLE],[lstar,MPI.DOUBLE])
        COMM.Gather([ystar_local,MPI.DOUBLE],[ystar,MPI.DOUBLE])
        # ~~~~~~~~~~~~~~~~~~~~
        W_fit = self.InterpFun1d(self.m_grid, W)

        return W, W_fit, lstar, ystar

    def BuyerProblem(self, Vbar_fit, omega, SolMethod='continuous'):
        """ Derive Buyer m's value B(m) and policy functions (x,b)(m)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Know your onions!
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Based on MSSMOD.PY, new renovation with OpenMPI compute
            capability. Buyer :=: DM agent now!
        """
        # Instantiate marginal value function ...
        Vprime = Vbar_fit.derivative()
        # Gross growth of nominal wage
        #gamma = self.MoneyGrowthFactor(omega)
        gamma = self.MoneyGrowthFactor()
        # Storage: Also for 0 < m <= KAPPA, value/policy functions
        bstar = np.zeros(self.m_gridsize,dtype='d')
        xstar = bstar.copy()
        qstar = bstar.copy()
        # Pre-fill B(.) for m < k:
        B = self.BETA*Vbar_fit( self.MoneyBalance(self.m_grid, omega) )
        #print(np.atleast_2d(B).shape)
        # ------------ Hello: ROOT 0 WORLD ... -------------------------
        COMM = MPI.COMM_WORLD         # instantiate MPI worlds
        num_cpus = COMM.size
        rank = COMM.rank
        # Shape of partition elements of root=0 domain
        Ngrid_m_gtk = self.m_grid_gtk.shape[0] # card(set{ m : m > k })
        # Number of rows and columns
        nr = int(np.ceil(float(Ngrid_m_gtk)/num_cpus)*num_cpus)
        nc = 1
        # Domain partition element (slave's slice of job)
        block_size = nr / num_cpus
        idxset_m_gtk_partition = np.arange(block_size, dtype='int')
        #np.zeros((block_size,nc),dtype='int')
        # CODOMAINs: Root 0's Gather destination(s) at the end, for m > KAPPA
        B_gtk = np.empty(self.m_grid_gtk.shape,dtype='d')
        bstar_gtk = np.empty(self.m_grid_gtk.shape,dtype='d')
        xstar_gtk = np.empty(self.m_grid_gtk.shape,dtype='d')
        qstar_gtk = np.empty(self.m_grid_gtk.shape,dtype='d')
        # BROADCAST: Arrays to workers
        if (rank == 0):
            BcastData = {   'self': self,
                            'Vbar_fit': Vbar_fit,
                            'Vprime': Vprime,
                            'omega': omega,
                            'method': SolMethod
                        }
            idxset_m_gtk = np.arange(Ngrid_m_gtk, dtype='int')
        else:
            BcastData = None
            idxset_m_gtk = np.zeros(self.m_grid_gtk.shape, dtype='int')

        # Scatter L_set to slaves each with L_partition
        COMM.Scatter([idxset_m_gtk,MPI.INT],[idxset_m_gtk_partition,MPI.INT])
        # Broadcast common data to all slaves
        BcastData = COMM.bcast( BcastData, root=0 )
        # ------------ Goodbye: ... ROOT 0 WORLD -----------------------
        #                       |
        #                       |
        # ------------ Hello: SLAVE WORLD ... --------------------------
        # Extract broadcasts from Master
        self_local = BcastData['self']
        Vfit_local = BcastData['Vbar_fit']
        Vprime_local = BcastData['Vprime']
        omega_local = BcastData['omega']
        SolMethod = BcastData['method']
        # Local storage of local results
        B_gtk_local = np.empty(idxset_m_gtk_partition.shape,dtype='d')
        bstar_gtk_local = np.empty(idxset_m_gtk_partition.shape,dtype='d')
        xstar_gtk_local = np.empty(idxset_m_gtk_partition.shape,dtype='d')
        qstar_gtk_local = np.empty(idxset_m_gtk_partition.shape,dtype='d')
        # Growth rate of money
        gamma_local =  self_local.MoneyGrowthFactor()
        # Two flavors ...
        if SolMethod == 'continuous':
            sys.exit('OPEN ISSUE. Continuous Method: Not available. This is a bilinear optimization problem!')
        elif SolMethod == 'discrete':
            # Bellman operator over states m > k (agents who go shopping in DM)
            for idx, index_m in enumerate(idxset_m_gtk_partition):
                m = self_local.m_grid_gtk[index_m]
                idx_m = idx + self_local.index_gtk
                # Set of feasible x given current m: ( k, m ]
                X = self_local.m_grid_gtk_fine[self_local.m_grid_gtk_fine <= m]
                # End-of-submarket balance after spending x = x(m), or, 0
                # phi = lambda x: (m+self_local.TAU-x)*(1.0-self_local.DELTA)/gamma_local
                phi = lambda x: (m - x + self_local.TAU/omega_local)/gamma_local
                # \beta V'(\phi_b(m, x(m))) marginal value function
                betaVpphi=lambda x: self_local.BETA*Vprime_local(phi(x))/gamma_local
                # Evaluate derivative function at all points in X
                # From FOC w.r.t. x(m), Thm 2. eqn (2.19)
                # beta*V'(phi)/gamma = u_1(x,b) (=dU/dQ x dQ/dx :=:dU/dQ x 1 )
                # and we know u_1(x,b) = u_q[Q(b(x),x)]
                # This gives the equation b(m) = b(m, x(m)) on grids X ...
                #
                # Q[b(x), x] quantity of q, back out from Thm 2. eqn (2.19)
                qX = self_local.invu_q_DM( betaVpphi(X) )
                # Use DM zero profit condition:
                # xi :=: x - c[Q(x,b(x))] = k/mu(b)
                xi = X - self_local.cost_DM(qX)
                # If xi < k / mu(b) at mu(b) = 1, i.e., b = 0,
                # then set: xi = kappa + eps, where eps > 0 to ensure b > 0
                xi[xi < self_local.KAPPA] = self_local.KAPPA
                + np.finfo(np.float64).eps
                                                    #np.finfo(np.float64).eps
                                                    # constrains b >= 0, q>=0
                # back out b(x) = mu^{-1}(k/xi), where guaranteed b > 0
                bX = self_local.invmu(self_local.KAPPA/xi)
                # bX[bX > 1.0] = 1.0
                # Flow utility at qX
                UtilX = self_local.u_DM(self_local.Q(X, bX))
                # Buyer's objective function at grids X
                phi_a = phi(np.zeros(X.size)) # no trades
                phi_b = phi(X)                # trades
                BX = self_local.BETA*Vfit_local(phi_a) + bX*(UtilX + self_local.BETA*(Vfit_local(phi_b)-Vfit_local(phi_a)))
                # Maximize over x = x(m) = x(m, b(m))
                B_gtk_local[idx] = BX.max()
                # Maximizer
                xst = X[BX.argmax()]
                xstar_gtk_local[idx] = xst
                # Side equations/functions
                bst = np.min([bX[BX.argmax()], 1.0]) #Constraint b*<= 1
                # bst = bX[BX.argmax()]
                # if bst > 1.0: # Impose constraint b*<= 1
                #     bst = 1.0 - np.finfo(np.float64).eps
                bstar_gtk_local[idx] = bst
                qstar_gtk_local[idx] = self_local.Q_scalar(xst, bst)
        # ---------- Goodbye: ... SLAVE WORLD --------------------------
        #                       |
        #                       |
        # ---------- Hello: ROOT 0 WORLD -------------------------------
        # Collect back to CODOMAINs in rank 0
        COMM.Gather([B_gtk_local,MPI.DOUBLE],[B_gtk,MPI.DOUBLE])
        COMM.Gather([bstar_gtk_local,MPI.DOUBLE],[bstar_gtk,MPI.DOUBLE])
        COMM.Gather([qstar_gtk_local,MPI.DOUBLE],[qstar_gtk,MPI.DOUBLE])
        COMM.Gather([xstar_gtk_local,MPI.DOUBLE],[xstar_gtk,MPI.DOUBLE])
        # Splice back in (B, xstar, qstar, bstar)
        #Ngrid_m_gtk = self.m_grid[self.m_grid > self.KAPPA].size
        #if (Ngrid_m_gtk % 2 == 0):
        # Original set {m : m > KAPPA} is even sized
        B[self.index_gtk ::] = B_gtk
        bstar[self.index_gtk ::] = bstar_gtk
        qstar[self.index_gtk ::] = qstar_gtk
        xstar[self.index_gtk ::] = xstar_gtk
        # Fitted version of B
        B_fit = self.InterpFun1d(self.m_grid, B)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return B, B_fit, bstar, qstar, xstar

    def Vtilde(self, Bfit, Wfit, omega):
        """ Upper envelop of B(m) and W(m - CHI) value functions.
            Note: W(m-CHI) := W(m) - A*CHI since W(m) is linear.
            CHI is new addition: real fixed cost to go to CM.
        """
        # Evaluate W
        mpoints = self.m_grid
        W_chi = Wfit(mpoints)
        # Three cases of borrowing limits:
        if self.shortsell == False:
            # Case with zero short-sell constraint
            cost = self.CHI
        if self.shortsell == None:
            # Case with ad-hoc short-sell constraint
            cost = self.CHI - self.ymax_adhoc
        elif self.shortsell == True:
            # Case with natural short-sell constraint
            cost = self.CHI - self.ymax(omega)
        # Define the jump in the W function for all m < cost (agents who cannot afford to pay COST to go to DM). This forces their valuation W(m) to be surely dominated by B(m), at all m < cost:
        W_chi[mpoints < cost] = 0.0 # -np.inf #0.0
        test = (mpoints >= cost)
        W_chi[test] = Wfit(mpoints[test] - self.CHI)
        # Return Vtilde
        return np.maximum(Bfit(mpoints), W_chi)

    def V(self, Vtilde):
        """Find convex hull of graph(Vtilde) to get V"""
        # Vtilde_graph = np.vstack((self.m_grid, Vtilde)).transpose()
        Vtilde_graph = np.column_stack((self.m_grid, Vtilde))
        # Step 0:   Find Convex hull of { m, Vtilde(m) }
        mpoint = np.array([ self.m_grid.max(), Vtilde.min() ]) # SE corner
        graph = np.vstack((Vtilde_graph, mpoint))
        #print(graph)
        chull = ConvexHull(graph)
        extreme_pts = graph[chull.vertices, :]
        # DO DIRTY hack below: --added 2016-10-28 ------------------------
        # pchip may not give (0, Vtilde(0)) as the lowest lottery prize
        # if (DirtyHack==True) and (Vtilde_graph[0,1] != extreme_pts[0,1]):
        #     extreme_pts[0,:] = np.array([0.0, Vtilde[0]])
        # DO DIRTY hack above -------------------------------------------
        # Step 1: Intersection between original function Vtilde and ext. pts.
        v_intersect_graph, ind = setops.intersect(Vtilde_graph,extreme_pts)[0:2]
        # Step 2: First difference the index locations
        idiff = np.diff(ind,n=1,axis=0)
        # Step 3: if idiff contains elements > 1 then we know exist one/more line segment (i.e., more than one set of lotteries played)
        idz = np.where(idiff > 1)[0] # Location where straddles more than 1 step
        # Step 4: Given the jumps, we have the end points defining each lottery!
        idz = np.column_stack((idz, idz+1)) #[idz, idz+1]:=:lottery(z-low,z-hi)
        # Step 5: Store lottery supports
        lottery_supports = v_intersect_graph[idz, 0]
        lottery_payoffs = v_intersect_graph[idz, 1]
        #print(lottery_supports)
        # Step 6: Interpolate to approximate V
        Vfit = self.InterpFun1d(v_intersect_graph[:,0],
                                        v_intersect_graph[:,1])
        # Step 7: Eliminate Fake Lotteries (due to numerical imprecisions)
        if self.DirtyHack==True:
            selector = []
            for idx_lot in range(lottery_supports.shape[0]):
                # Lottery prizes for at current lottery segment, idx_lot
                zlo, zhi = lottery_supports[idx_lot,:]
                # Conditions: points between current lottery segment
                find_condition = (self.m_grid > zlo) & (self.m_grid < zhi)
                # Value of Vtilde between current lottery segment
                Vtilde_z_temp = Vtilde[find_condition]
                # Value of V between current lottery segment
                V_z_temp = Vfit( self.m_grid[find_condition] )
                # Check: Is V "significantly different" from Vtilde at lottery segment?
                gap = np.absolute(V_z_temp - Vtilde_z_temp).max()
                if gap > self.TOL2:
                    selector.append(idx_lot) # Keep, if "YES"
            # Update definition of the set of lotteries
            lottery_supports_temp = lottery_supports[selector, :]
            lottery_payoffs_temp = lottery_payoffs[selector, :]
            if lottery_supports_temp.size == 0:
                lottery_supports = lottery_supports[0,:]
                lottery_payoffs = lottery_payoffs[0,:]
            else:
                lottery_supports = lottery_supports_temp
                lottery_payoffs = lottery_payoffs_temp
        #print(lottery_supports)
        # Step 8: Store V as evaluated on finite set m_grid
        V_grid = Vfit(self.m_grid)
        return V_grid, Vfit, lottery_supports, lottery_payoffs

    def Vbar(self, V_grid, W_grid):
        """ Convex combination of V and W (defined on grid of m) """
        # Discrete evals of Vbar
        Vbar_grid = (1.0 - self.ALPHA)*V_grid + self.ALPHA*W_grid
        # Interpolant OBJECT - derivative stored as Vbar_fit.derivative()
        Vbar_fit = self.InterpFun1d(self.m_grid, Vbar_grid)
        return Vbar_grid, Vbar_fit

    def ExPostActions(self, m, W_fit, B_fit, bstar_fit, xstar_fit, qstar_fit,
                                                ystar_fit, Cstar, ulist_DM_t, omega):
        """Evaluate Ex-post Decisions at a given level of money holding, m.
           See Appendix D, Algorithm 1.
        """
        if self.shortsell == False:
            # Case with zero short-sell constraint
            solvency = m - self.CHI
        if self.shortsell == None:
            # Case with zero short-sell constraint
            solvency = m - self.CHI + self.ymax_adhoc
        elif self.shortsell == True:
            # Case with natural short-sell constraint
            solvency = m - self.CHI + self.ymax(omega)
        # CMtest works for both self.shortsell == True or False
        CMtest = (solvency >= 0.0)
        ## ======== Temp Block out ============================================
        if CMtest == True and W_fit(m - self.CHI) >= B_fit(m):
            # Agent goes to CM to work and save money
            # b_m, x_m, q_m, match = 0.0, 0.0, 0.0, -1
            # ystar is constant function! We just inter/extrapolate here!
            y_m = ystar_fit(m - self.CHI)           # Note fixed cost CHI paid!
            # need to work harder if net asset is m - CHI
            l_m = self.BudgetConstraint(m - self.CHI, y_m, Cstar, omega)[0]
            # Ensure upper bound on l is respected: l < l_ub
            # l_ub = Cstar + np.minimum(self.m_max, self.m_max-self.TAU/omega)
            l_ub = Cstar + self.ymax(omega)
            l_m = np.minimum(l_m, l_ub - 1e-6)
            mprime_m = y_m
            participation = 2       # 'CM' := 2
            q_m = 0.0
            x_m = 0.0
            b_m = 0.0
            match = -1
        elif CMtest == False or W_fit(m - self.CHI) <= B_fit(m):
            # Agent goes searching for DM trading post
            l_m = 0.0
            y_m = 0.0
            # u = np.random.rand()
            u = ulist_DM_t
            if (u <= bstar_fit(m)) and (m >= self.KAPPA):
                # Matched and traded
                b_m = bstar_fit(m)
                x_m = xstar_fit(m)
                q_m = self.Q_scalar(x_m,b_m)
                # if (x_m <= 0.0) or (b_m <= 0.0) or (q_m <= 0.0):
                #     q_m = 0.0
                #
                #q_m = qstar_fit(m)
                match = 1
                mprime_m = m - x_m
                participation = 1       # DM-trade := 1
            else:
                # No match no trade
                b_m = 0.0
                x_m = 0.0
                q_m = 0.0
                match = 0
                mprime_m = m
                participation = 0       # DM-notrade := 0
        else:
            y_m = ystar_fit(m - self.CHI)           # Note fixed cost CHI paid!
            # need to work harder if net asset is m - CHI
            l_m = self.BudgetConstraint(m - self.CHI, y_m, Cstar, omega)[0]
            # Ensure upper bound on l is respected: l < l_ub
            # l_ub = Cstar + np.minimum(self.m_max, self.m_max-self.TAU/omega)
            l_ub = Cstar + self.ymax(omega)
            l_m = np.minimum(l_m, l_ub - 1e-6)
            mprime_m = y_m
            participation = 2       # 'CM' := 2
            q_m = 0.0
            x_m = 0.0
            b_m = 0.0
            match = -1

        return l_m, y_m, b_m, x_m, q_m, match, mprime_m, participation

    def LotteryTest(self, m, lottery_supports):
        """ At current m, test if there is a ex-ante lottery to be played.
        - If yes, play it and return the lottery outcome as update m.
        - If not, don't play and return same m.
        Inputs:
            m, must be a scalar, real.
            lottery_supports, is L x 2 numpy array.
        """
        # Set initial prize as None and return m <-- m
        prize = -np.inf
        # Make sure lottery supports always has two columns
        lottery_supports = np.reshape(lottery_supports, (-1,2))
        # print(lottery_supports)
        # Check if lottery exists
        if (lottery_supports.size > 0):
            # Boolean list of size len(lottery_supports)
            boolist = [ ( z[0] <= m <= z[1] ) for z in lottery_supports ]
            if any(boolist) == True:
                # Case with many lotteries -  idex True -- one containing m
                interval = lottery_supports[np.asarray(boolist)]
                # Get lottery prizes
                z0, z1 = interval.ravel()
                # print(z0, z1)
                # Construct distribution/lottery (pihat1, 1-pihat1) on {z1,z2}
                pihat1 = (z1 - m)/(z1 - z0)
                # Plays lottery...
                u = np.random.rand()
                if (u <= pihat1):
                    m = z0  # update m as prize z1
                else:
                    m = z1  # update m as prize z2
                prize = m   # update record of actual prize won
        # Test complete
        return m, prize

    def Distribution(self,lottery_supports,V_fit,
                        W_fit,B_fit,ystar,bstar,qstar,xstar,omega,
                        T=1e5,seed=45823):
        """ Compute equilibrium distribution given policy functions
            by Monte Carlo """
        # First lottery (high) prize
        #m0 = lottery_supports[0,1]
        # Instantiate fitted policy functions (CM and DM)
        xstar_fit = self.InterpFun1d(self.m_grid, xstar) # DM payment
        bstar_fit = self.InterpFun1d(self.m_grid, bstar) # DM buyer probability
        qstar_fit = self.InterpFun1d(self.m_grid, qstar) # DM buyer probability
        ystar_fit = self.InterpFun1d(self.m_grid, ystar) # CM saving (money)
        # Storage python list
        T = self.T
        M_simulate = np.zeros(T)
        y_simulate = np.zeros(T)
        l_simulate = np.zeros(T)
        q_simulate = np.zeros(T)
        b_simulate = np.zeros(T)
        x_simulate = np.zeros(T)
        match_simulate = np.zeros(T)
        lotteryprize_simulate = np.tile(-np.inf, T)
        participation_simulate = np.tile(2, T)

        # Optimal C in CM
        #Cstar = self.invU_C_CM(self.A)
        Cstar = self.Cstar

        # Generate seeded random sequence
        np.random.seed(seed)
        ulist = np.random.rand(T)
        ulist_DM = np.random.rand(T)

        # Initial agent state:
        #m = ystar_fit(0.0)
        m = 0.0
        # Monte Carlo loop
        for idx_s in range(T):
            #print("Simulation s = %i of %i" %(idx_s, T))
            # initial balance including seigniorage transfer (everyone gets it!)
            m = self.MoneyBalance(m, omega)
            # Record current state m
            # M_simulate.append(m)
            # M_simulate[idx_s] = m
            # Liquidity shock z ~ (ALPHA, 1-ALPHA)
            #u = np.random.rand()
            u = ulist[idx_s]
            if (u <= self.ALPHA):
                # No lottery for sure
                prize = -np.inf
                # Get out of Jail, Go to CM for FREE! End of CM balance:
                y_m = ystar_fit(m)
                # y_simulate.append(y_m)
                l_m = self.BudgetConstraint(m, y_m, Cstar, omega)[0]
                # l_simulate.append(l_m)
                # Not in DM so ...
                b_m = 0.0
                x_m = 0.0
                q_m = 0.0
                match = -1
                # Update to next period m'
                mprime = y_m
                # participation_simulate.append(2)  # 'CM' := 2
                participation = 2
            else:
                # Start Lottery checks
                m, prize = self.LotteryTest(m,lottery_supports)

                # Given updated m (lottery or not), evaluate ExPostActions()
                ulist_DM_t = ulist_DM[idx_s] # Uniform r.v. draw for DM matching
                l_m, y_m, b_m, x_m, q_m, match, mprime, participation = \
                self.ExPostActions(m, W_fit,
                                   B_fit, bstar_fit, xstar_fit, qstar_fit,
                                   ystar_fit, Cstar, ulist_DM_t, omega)

            lotteryprize_simulate[idx_s] = prize
            y_simulate[idx_s] = y_m
            l_simulate[idx_s] = l_m
            b_simulate[idx_s] = b_m
            x_simulate[idx_s] = x_m
            q_simulate[idx_s] = q_m
            # q_simulate[idx_s] = self.Q_scalar(x_m, b_m)
            match_simulate[idx_s] = match
            participation_simulate[idx_s] = participation

            # Record current state m
            # M_simulate.append(m)
            M_simulate[idx_s] = m
            # Beginning of next period balance (before transfers)
            m = mprime
        # End FOR Loop

        # Approximate distribution by histograms (for visuals)
        # - discard burn in period of half
        burn = int(T + T%2) // 4

        Msims = M_simulate[burn::]
        ysims = y_simulate[burn::]
        lsims = l_simulate[burn::]
        bsims = b_simulate[burn::]
        xsims = x_simulate[burn::]
        qsims = q_simulate[burn::]
        matchsims = match_simulate[burn::]
        lottpsims = lotteryprize_simulate[burn::]
        partisims = participation_simulate[burn::]

        # Flag history
        if lottpsims[lottpsims > 0.0].sum() > 0.0:
            flag = "Lottery played during history."
        else:
            flag = "Lottery NOT played during history."
        # Pack snapshots into dictionary
        history = {
                    'm': Msims,
                    'y': ysims,
                    'l': lsims,
                    'b': bsims,
                    'x': xsims,
                    'q': qsims,
                    'match': matchsims,
                    'lottp': lottpsims,
                    'parti': partisims,
                }
        # Elvis has left the building ...
        return history,flag

    def MarketClearing(self, simhistory):
        """Money supplied = Money demanded condition determining omega (nominal wage rate)
        """
        omega = 1.0/simhistory['m'].mean()
        return omega

    ##-------- STEADY STATE EQUILIBRIUM SOLVER -------------------
    def SolveSteadyState(self, outfilename, previous_outfilename=None,
                    LOAD_OLD=True, HOMOTOPY=None, FRESH_GUESS=None,     SAVE_RESULTS=False, STORE_APPROXSEQ=False, randseed=123264):
        """Iterating on OMEGA to solve for steady state equilibrium"""

        rank = MPI.COMM_WORLD.rank
        tic = time.time()

        # if (LOAD_OLD == True) or (LOAD_OLD == True and FRESH_GUESS == False):
        if (LOAD_OLD == True):
            if HOMOTOPY == True:
                fileloaded = previous_outfilename+'.npy' # Load last experiment
            else:
                fileloaded = outfilename+'.npy'
            if rank == 0:
                print("\n\t\tLoading RESULTS from ... "+ fileloaded)
            # Load dictionary containing RESULTS (old, saved as .NPY)
            results = np.load(fileloaded, encoding='latin1').item()

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
            V_tilde = results['valuegrid']['Vtilde']
            B = results['valuegrid']['B']
            W = results['valuegrid']['W']
            # Fitted value functions
            Vbarfit = results['valuefit']['Vbar']
            Vfit = results['valuefit']['V']
            Bfit = results['valuefit']['B']
            Wfit = results['valuefit']['W']
            # policy functions
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
            # For initial guess
            Vbar_old = Vbar.copy()
            #omega = self.MarketClearing(simhistory)
            omega_old = omega.copy()
        elif (LOAD_OLD == False):
            # Initial guess of ex-ante value function
            Vbar_init = self.u_DM(self.m_grid)
            # Step 0: Initial guess of omega and Vbarfit (fitted function)
            omega_old = 1.0
            Vbar_old = Vbar_init.copy()
            Vbarfit = self.InterpFun1d(self.m_grid, Vbar_old)

        if (LOAD_OLD == False) or (LOAD_OLD == True and FRESH_GUESS == True):

            # Storage-intermediate saves
            Vbar_seq = []
            V_seq = []
            V_tilde_seq = []
            lottery_seq = []
            errorOmega_seq = []
            errorV_seq = []

            MAXITER2 = self.MAXITER2

            for iteration in range(self.MAXITER1):
                #print('\niteration = %i' % (iteration))
                Vbar_omega_old = Vbar_old
                error_bellman = 1.0
                for iteration_bellman in range(MAXITER2):
                    # if rank==0:
                    #     print("\niter_bell %i \t %6.4g" % (iteration_bellman, error_bellman))
                    # Ex-post problems
                    # Step 1: Solve Worker problem
                    W,Wfit,lstar,ystar = self.WorkerProblem(Vbarfit,
                                               omega_old,SolMethod='continuous')
                    # Step 2: Solve Buyer problem
                    B,Bfit,bstar,qstar,xstar = self.BuyerProblem(Vbarfit,
                                                 omega_old,SolMethod='discrete')
                    # Step 3: Ex-ante problem in mixed strategies
                    if (rank == 0):
                        # V_tilde = self.Vtilde(B, W)
                        # Modded: 2017-04-17 to use fitted functions
                        V_tilde = self.Vtilde(Bfit, Wfit, omega_old)
                        V,Vfit,lottery_supports,lottery_payoffs=self.V(V_tilde)
                        # Define Vbar
                        Vbar, Vbarfit = self.Vbar(V,W)
                        # Distance between value functions
                        error_bellman = (np.absolute(Vbar - Vbar_old)).max()
                        # Update value function guess
                        Vbar_old = Vbar
                        # Stopping rule for Bellman operator iteration
                        if (error_bellman < self.TOL1) and (iteration > 0):
                            break
                        else:
                            continue

                if (rank == 0):
                    if (iteration_bellman == MAXITER2):
                        if (error_bellman >= self.TOL1):
                            flag_bellman=\
                            " VFI (fix omega): Max.iteration-Not Converged! | "
                        else:
                            flag_bellman=\
                            " VFI (fix omega): Max.iteration-Converged. | "
                    else:
                        if (error_bellman >= self.TOL1):
                            flag_bellman=\
                            " VFI (fix omega): Not Converged! | "
                        else:
                            flag_bellman =\
                            " VFI (fix omega): Converged. | "


                # Store current V(.; omega) given omega guess:
                    if (STORE_APPROXSEQ == True):
                        V_seq.append(V)
                        Vbar_seq.append(Vbar)
                        V_tilde_seq.append(V_tilde)
                        lottery_seq.append(lottery_payoffs)
                # For troubleshooting: Development stage only
                    if self.TESTMODE == True:
                        print("\nLotteries: prizes")
                        print(lottery_supports)

                # Step 4: Get implied distribution
                    simhistory, flag = self.Distribution(lottery_supports,
                                                    Vfit,Wfit,Bfit,
                                                    ystar,bstar,qstar,xstar,
                                                    omega_old,
                                                    T=self.T, seed=randseed )
                    # # This is a manual hack (2018-09-14) Python dict seems to
                    # # mess up 'q' in simhistory! Note self.Q() also updated
                    # qs = self.Q(simhistory['x'], simhistory['b'])
                    # simhistory['q'] = qs

                    # Step 5: Compute implied nominal wage
                    omega = self.MarketClearing(simhistory)
                    # Step 6: Check distance
                    #error1 = np.absolute(omega - omega_old)
                    error1 = np.absolute((omega - omega_old)/omega_old)
                    Vbar_omega = Vbar
                    error2 = (np.absolute(Vbar_omega - Vbar_omega_old)).max()
                    if (STORE_APPROXSEQ == True):
                        errorOmega_seq.append(error1)
                        errorV_seq.append(error2)
                    # Step 7: Update omega, Vbar_init, repeat and rinse
                    omega_old = self.SMOOTH*omega_old + (1-self.SMOOTH)*omega
                    Vbar_old = Vbar

                    # Concatenate Lottery flag message with flag_bellman
                    flag = flag_bellman + flag

                    ##---------------------------------------------------------
                    ## Adaptive fine tuning hack for search speedup

                    # Reset MAXITER2 to be longer (Value iterations)
                    # if error1 < 1e-2 and error2 < 1e-2:
                    if error1 < 5.0*self.TOL0 and error_bellman < 5.0*self.TOL1:
                        # Ensure proper value fn convergence when closer
                        MAXITER2 = 2*self.MAXITER2
                        # Refine action set when closer to convergence:
                        m_gridsize_fine_new = self.m_gridsize_refine
                        # DM action set
                        self.m_grid_gtk_fine = self.grid1d(self.KAPPA+1e-8,self.m_max,\
                                                Npoints=m_gridsize_fine_new)
                        # CM action set
                        self.y_grid_fine = self.grid1d(self.m_min, self.m_max, \
                                    Npoints=m_gridsize_fine_new,method='linear')
                    ##---------------------------------------------------------

                    # Step 8: Display convergence stats
                    # if (error1 < self.TOL0) and (error2 < self.TOL1):
                    if (error1 < self.TOL0) and (error_bellman < self.TOL1):
                        # Convergence satistied
                        if self.TESTMODE == True:
                            # print("\n%i/%i, err1 = %0.5f, err2 = %0.5f : %s"
                            #     % (iteration+1,iteration+1,error1,error2,flag) )
                            print("\n%i/%i, err1 = %0.7f, err2 = %0.7f : %s" % (iteration+1,iteration+1,error1,error_bellman,flag) )
                            # print("\nLotteries: prizes")
                            # print(lottery_supports)
                        else:
                            # self.StatusBar(iteration+1,iteration+1,
                            #                         error1,error2,flag)
                            self.StatusBar(iteration+1,iteration+1,
                                                    error1,error_bellman,flag)
                        break
                    else:
                        # Convergence not satistied
                        if self.TESTMODE == True:
                            print("%i/%i, err1 (omega) = %0.7f, err2 (V_omega) = %0.7f : %s" % (iteration,self.MAXITER1,error1,error_bellman,flag) )
                            # % (iteration,self.MAXITER1,error1,error2,flag) )
                        else:
                            self.StatusBar(iteration+1,self.MAXITER1,
                                            error1,error_bellman,flag)
                        continue

        # Store results in Python dictionaries with these KEYS:
        if rank == 0:
            diagnostics = { 'Vbar_seq'      : Vbar_seq,
                            'V_seq'         : V_seq,
                            'Vtilde_seq'    : V_tilde_seq,
                            'lottery_seq'   : lottery_seq,
                            'errorOmega_seq': errorOmega_seq,
                            'errorV_seq'    : errorV_seq
                            }
            valuegrid = {   'Vbar'     : Vbar,
                            'V'        : V,
                            'Vtilde'   : V_tilde,
                            'B'        : B,
                            'W'        : W
                            }
            valuefit = {    'Vbar'  : Vbarfit,
                            'V'     : Vfit,
                            'B'     : Bfit,
                            'W'     : Wfit
                            }
            policygrid = {  'ystar' : ystar,
                            'lstar' : lstar,
                            'bstar' : bstar,
                            'xstar' : xstar,
                            'qstar' : qstar,
                            }
            lottery = { 'supports' : lottery_supports,
                        'payoffs'  : lottery_payoffs
                        }
            distro = {  'simhistory' : simhistory,
                        'omega'      : omega
                        }
            # pack above dictionaries into another dictionary RESULTS
            results = { 'diagnostics': diagnostics,
                        'valuegrid'  : valuegrid,
                        'valuefit'   : valuefit,
                        'policygrid' : policygrid,
                        'lottery'    : lottery,
                        'distro'     : distro
                        }
            # Stopwatch
            toc = time.time() - tic
            print("\n\nElapsed time:", toc, "seconds")
            # Return RESULTS to ROOT 0
            return results

    ##-------- GRAPHICS DEPARTMENT -------------------------------
    def DrawValuefun(self, V, B, W, Vtilde, Vfit,
                        lottery_supports, lottery_payoffs, PlotSeparate=False):
        """Plot equilibrium value functions"""
        fig1 = plt.figure(facecolor="white")
        ymin = min(W.min(),B.min(),V.min())
        ymax = max(W.max(),B.max(),V.max())
        plt.ylim((ymin, ymax))
        #plt.tight_layout()
        # Area under graph of Vtilde (draw patch)
        plt.fill_between(self.m_grid,Vtilde, ymin, facecolor='gray', alpha=0.05, interpolate=True)
        # Graph of original non-convex function
        plt.plot(self.m_grid, Vtilde, 'g-', lw=3, clip_on=False)
        # Support of lotteries and linear segments (lotteries): Lottery i
        Vtildefit = self.InterpFun1d(self.m_grid,Vtilde)
        Bfit = self.InterpFun1d(self.m_grid, B)
        Wfit = self.InterpFun1d(self.m_grid, W)
        for i in range(lottery_supports.shape[0]):
            # Draw (x,y) coordinates of lottery supports
            if lottery_supports.ndim == 2:
                for j in range(lottery_supports.shape[1]):
                    plt.plot( np.tile(lottery_supports[i,j],5),
                              np.linspace(ymin, lottery_payoffs[i,j], 5), '-.k')
                                                        # Dash-dot lines (y-axis)
                    plt.plot( np.linspace(self.m_grid.min(), lottery_supports[i,j], 5), np.tile(lottery_payoffs[i,j],5), '-.k' )
                                                    # Dash-dot lines (x-axis)
            # Draw marker points of lottery supports
            plt.plot(lottery_supports[i], lottery_payoffs[i],
                                    'mo', markersize=5, clip_on=False)
            # Insert tick labels
            xticklabels = lottery_supports.ravel().tolist()
            #xticklabels.append(mhat)
            yticklabels = lottery_payoffs.ravel().tolist()
            xticks = [format(t, "0.2f") for t in xticklabels]
            yticks = [format(t, "0.2f") for t in yticklabels]
            plt.xticks(xticklabels, xticks, rotation='vertical')
            plt.yticks(yticklabels, yticks, rotation='horizontal')
            # Shade the areas covexified over!
            if lottery_supports.ndim == 2:
                x = np.linspace(lottery_supports[i,0],lottery_supports[i,1], 200)
            else:
                x = np.linspace(lottery_supports[0],lottery_supports[1], 200)
            # slope = np.diff(lottery_payoffs,axis=1)[i]/np.diff(lottery_supports,axis=1)[i]
            y1 = Vfit(x)#lottery_payoffs[i,0] + slope*(x - x[0])
            y2 = Vtildefit(x)        # interpolate Vtilde over sub-domain x
            plt.fill_between(x, y2, y1, where=y2 <= y1,
                                facecolor='gray', alpha = 0.25, interpolate=True)

        # Plot V:
        # V = spline(v_intersect_graph[:,0],v_intersect_graph[:,1], k=1)
        # plt.plot(m, V(m), 'yd', markersize=2, clip_on=False)
        Wf,=plt.plot(self.m_grid, W, 'b.-',
                        markersize=2,clip_on=False,label='$W$')
        Bf,=plt.plot(self.m_grid, B, 'g--',
                        markersize=2,clip_on=False,label='$B$')
        Vf,=plt.plot(self.m_grid, V, 'md-',
                        alpha=0.5,lw=3,markersize=2,clip_on=False,label='$V$')
        # Tick points for m0 and mhat:
        #m0 = lottery_supports[0,1]
        #plt.plot(m0, Vfit(m0), 'sb', mhat, Vfit(mhat), 'dr', clip_on=False)
        # Legends on top
        plt.legend(handles=[Vf,Bf,Wf],
                                bbox_to_anchor=(0.5, 1.05),
                                    ncol=3, fancybox=True, shadow=True)

        plt.xlabel('$m$')
        plt.tight_layout()
        # plt.savefig(figfolder+'values.eps')
        # plt.savefig(figfolder+'values.png')

        # Separate plotting option
        if PlotSeparate == True:
            # Graph of B
            fig2 = plt.figure(facecolor="white")
            Bs,=plt.plot(self.m_grid, B, 'g.-', lw=3, clip_on=False, label='$B$')
            plt.ylabel('$B(m)$')
            plt.xlabel('$m$')
            # Graph of W
            fig3 = plt.figure()
            Ws,=plt.plot(self.m_grid, W, 'r.-', lw=3, clip_on=False, label='$W$')
            plt.xlabel('$m$')
            plt.ylabel('$W(m)$')

        return plt

    def DrawPolicyfun(self, policies, policynames, lottery_supports=None):
        """Plot equilibrium decision functions"""
        nvars = len(policies)
        ncol = 2
        nrow = (nvars + np.mod(nvars,ncol))/ncol
        fig = plt.figure(facecolor="white", tight_layout=True)
        #fig, ax = plt.subplots(facecolor="white", tight_layout=True)
        for i, policy in enumerate(policies):
            plt.subplot(nrow, ncol, i+1)
            #plt.subplots_adjust(hspace = .001)
            plt.plot(self.m_grid, policy, 'k.--')
            ax = fig.gca()
            ax.set_ylabel(policynames[i])
            ax.set_xlabel('$m$')

            # Draw patches for regions where lottery exists
            plist = []
            if lottery_supports is not None:
                # Shade lottery zone:
                y, ymax = ax.get_ylim()
                height = ymax - y
                colorscheme="#f97c07"
                for i in range(lottery_supports.shape[0]):
                    if lottery_supports.ndim == 2:
                        xi = lottery_supports[i,0]
                        xi_plus = lottery_supports[i,1]
                    else:
                        xi = lottery_supports[0]
                        xi_plus = lottery_supports[1]
                    width = xi_plus - xi
                    # lottery zone
                    #ax.add_patch(
                    plist.append(
                        patches.Rectangle(                                              (xi, y),        # (x,y) coordinate
                            width,          # width from x
                            height,          # height from y
                            edgecolor=colorscheme,
                            facecolor=colorscheme,
                            alpha=0.5
                        )
                    )

            # Shade no-shopping zone
            colorscheme="#f442b0"
            x = self.m_grid.min()
            width = self.KAPPA - x
            #ax.add_patch(
            plist.append(
                patches.Rectangle(                                                  (x, y),        # (x,y) coordinate
                    width,          # width from x
                    height,          # height from y
                    edgecolor=colorscheme,
                    facecolor=colorscheme,
                    alpha=0.2
                )
            )
            # Render patches ...
            for p in plist:
                ax.add_patch(p)
        return plt

    # def DrawDistro(self, support_G=support_G, g=g, xdata=None, bins=100):
    def DrawDistro(self, data, lottery_supports=None, bins=100):
        """Plot histograms:
            Admits two options:
            (1) a know discrete distribution with known support_G and density function g, or
            (2) given random sample stored as xdata to be plotted as histogram with 100 bins.
           By default we plot the histograms as relative frequencies (probability).
        """
        #plt.figure()
        num_bins = bins
        # the histogram of the data
        fig, ax = plt.subplots(facecolor="white", tight_layout=True)
        plt.hist(data, num_bins, density=1,
                        facecolor='green', alpha=0.5)
        plt.xlabel('$m$')
        plt.ylabel('Relative frequency')
        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)

        # Hide the left, right and top spines
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_visible(True)
        # Tick points for m0 and mhat:
        #plt.plot(m0, 0.0, 'sb', mhat, 0.0, 'dr', clip_on=False)

        # Shade lottery zone:
        colorscheme="#f97c07"
        if lottery_supports is not None:
            for i in range(lottery_supports.shape[0]):
                if lottery_supports.ndim == 2:
                    x = lottery_supports[i,0]
                    xp = lottery_supports[i,1]
                else:
                    x = lottery_supports[0]
                    xp = lottery_supports[1]
                y = -0.1
                width = xp - x
                ax.add_patch(
                    patches.Rectangle(                                             (x, y),        # (x,y) coordinate
                        width,          # width from x
                        0.001,          # height from y
                        edgecolor=colorscheme,
                        facecolor=colorscheme,
                        alpha=0.5
                    )
                )

        return plt

    def DrawTimeSeries(self, y, ylabelstring, x=None):
        """Plot Time Series DATA"""
        if x == None:
            x = np.arange(y.size)
        plt.figure(facecolor="white")
        Ws,=plt.plot(x, y, 'o-r', lw=1, clip_on=False)
        plt.plot(x, y, 'o-r', lw=1, clip_on=False)
        plt.xlabel('$t$')
        plt.ylabel(ylabelstring)
        plt.tight_layout()
        return plt

    def AnimateFunction(self, V_seq, lottery_seq):
        """
            Usage: V_seq is a List, where V_seq[i] is a Numpy array
                   AnimateOptions is a Dictionary
        """
        print("\n We're now making a movie ... One moment please! \n")
        fig = plt.figure(facecolor='white')
        ngrid = V_seq[0].size
        X = np.arange(ngrid)
        V, l1 = setops.flatten(V_seq)
        ax = plt.axes(xlim=(X.min(), X.max()), ylim=(min(V), max(V)))
        line, = ax.plot([], [])
        plt.xlabel('$m$' + ' $(\\times$ ' + str(ngrid-1) + '$^{-1})$')
        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,
        # animation function.  This is called sequentially
        def animate(i):
            x = X
            y = V_seq[i]
            line.set_data(x, y)
            ax.set_ylim(V_seq[i].min(), V_seq[i].max())
            return line,

        # blit=True means only re-draw the parts that have changed.
        anime = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(V_seq), interval=200, blit=True)
        # anime.save('fitted-vfun-iter_anime.mp4', fps=1)
        return anime

    def DrawValuefunSequence(self, V_tilde_seq, V_seq, lottery_seq,graphV_seq, Selector=[0]):
        ngrid = len(V_seq[0])
        for i, j in enumerate(Selector):
            plt.figure()
            # Change xticks
            plt.plot(V_seq[j],'.--',label='V('+str(j)+')')
            x = np.arange(V_seq[j].size)
            index_bool = np.in1d(V_seq[j], lottery_seq[j].flatten(),
                                                            invert=False)
            plt.plot(x[index_bool], V_seq[j][index_bool],'o-m',
                                                            label='Lottery')
            plt.plot(V_tilde_seq[j],'.g',label='$\~V$'+'('+str(j)+')')
            plt.plot(graphV_seq[j][:,1], 'xr')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                                        ncol=4, fancybox=True, shadow=True)
            plt.xlabel('$m$' + ' $(\\times$ ' + str(ngrid-1) + '$^{-1})$')
            plt.savefig(figfolder+'Fitted-V-snapshot_'+str(j)+'.png')
        return plt

    #---------- STATISTICS --------------------------------------------------
    def moments_discrete(self, X, p=None):
        """Moments (1st - 4th) of discrete distributions:
        p := p(X) is relative frequencies, X is outcomes/support of p"""
        if self.montecarlo == False:
            mean = np.dot( p, X )
            variance = np.dot( p, (X-mean)**2.0 )
            std = np.sqrt(variance)
            skewness = np.dot( p, ((X-mean)/std)**3.0 )
            kurtosis = np.dot( p, ((X-mean)/std)**4.0 ) / (variance**2.0)
        else:
            mean = X.mean()
            variance = ((X-mean)**2.0).mean()
            std = np.sqrt(variance)
            skewness = ((X-mean)**3.0).mean() / (std**2.0)
            kurtosis = ((X-mean)**4.0).mean() / (variance**2.0)

        moms = {    'mean':mean,
                    'variance':variance,
                    'std':std,
                    'skewness':skewness,
                    'kurtosis':kurtosis
                }
        return moms

    def gini_discrete(self, p, X):
        """Gini coefficient for discrete distributions:
        p := p(X) is relative frequencies, X is outcomes support of p.
        NOTE: Coded based on Wikipedia formulae in https://en.wikipedia.org/wiki/Gini_coefficient#Discrete_probability_distribution
        """
        graph_pdf = np.column_stack((X, p)) # graph of discrete p.d.f.
        XP = graph_pdf[graph_pdf[:,0].argsort()] # Sort array (X, p) by X ascend
        # The array [S(0), S(1), ..., S(n)],
        # where S(n) = S(n-1) + p(n).dot(X(n)) and S(0) = 0 :
        S = np.append(0.0, (XP[:,0]*XP[:,1])).cumsum()
        # Two-adjacent-element sum of S elements (increasing order on S)
        TwoStepSum = S + np.roll(S,-1)
        # Now construct Gini coefficient ...
        numerator = np.dot(XP[:,1], TwoStepSum[0:-1])
        denominator = S[-1] # Equivalent: np.dot(XP[:,0], XP[:,1])
        return 1.0 - numerator/denominator

    def gini_rawdata(self, X):
        """Calculating Gini coefficient from raw data. Modified from: http://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/
        """
        if isinstance(X, (list)):
            sorted_list = sorted(X)
        elif isinstance(X, (np.ndarray)):
            sorted_list = sorted(X.tolist())
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
        fair_area = height * len(X) / 2.
        return (fair_area - area) / fair_area

    def modelstats(self, results):
        """Define model simulated statistics for purposes of calibration of estimation by Method of Simulated Moments

        results:    output from SolveSteadyState()
        """
        # Extract omega and structure simhistory
        omega = results['distro']['omega']
        simhistory = results['distro']['simhistory']
        participation = simhistory['parti']
        # STAT 1: Conditional mean allocations of l
        partrateCM = participation[participation==2].size/participation.size
        partrateDM = 1.0 - partrateCM
        lmean = partrateCM*(simhistory['l'][participation==2]).mean()/self.l_max
        # STAT 2: Inverse of Money Velocity
        m_mean = 1.0/omega
        x_mean = simhistory['x'][participation==1].mean()
        C_mean = self.Cstar
        # Model's steady-state 1/velocity = (Nominal M)/(Nominal GDP)
        ivelocity = m_mean / (partrateDM*x_mean + partrateCM*C_mean)
        return lmean, ivelocity
