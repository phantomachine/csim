% Codes for the paper: "On the Stability of Money Demand" by
% Robert E. Lucas Jr. (University of Chicago) and
% Juan Pablo Nicolini (FRB-Minneapolis and Universidad Di Tella)
% Written by Manuel Macera (UMN/Fed-Minneapolis)
% Modified by Joao Ayres (UMN/Fed-Minneapolis)
% Date: February 2015

% Description: this file is used to generate figures 7a and 7b, which
% plot the curves (demand/(demand+deposits)) and M1/GDP vs. Interest Rate
% for different parameter values.

close all; clear all; clc

%% INPUTS

% Number of different calibrations (we have 10 parameters in each)
Nc = 4;
inputs = zeros(4,10);

% INPUTS: Calibration 1 (choose values)
eta = 1.5621;                           % parameter of the distribution
theta_c = 1.01;                         % loss rate
theta_a = .01;                          % reserve requirement for MMDA
theta_d = .1;                           % reserve requirement for demand deposits
r_star = 0.000;                         % interet rate (Regulation Q)
k_d = 0.0300;                           % cost for deposits
k_a = 0.0494;                           % cost for mmda's
k_dh = 0.0074;                          % for decentralization (household cost)
phi = 0.0057;                           % time cost
k_db = k_d-k_dh;                        % for decentralization (bank cost)
A_star = 0.4138;                        % free parameter A (money balances is 25% of GDP when r = 6%)

% No changes here
inputs(1,1) = eta;
inputs(1,2) = theta_c;
inputs(1,3) = theta_a;
inputs(1,4) = theta_d;
inputs(1,5) = r_star;
inputs(1,6) = k_d;
inputs(1,7) = k_a;
inputs(1,8) = k_dh;
inputs(1,9) = phi;
inputs(1,10) = k_db;
inputs(1,11) = A_star;

% INPUTS: Calibration 2 (choose values)
eta = 1.5621;                           % parameter of the distribution
theta_c = 1.01;                         % loss rate
theta_a = .01;                          % reserve requirement for MMDA
theta_d = .1;                           % reserve requirement for demand deposits
r_star = 0.000;                         % interet rate (Regulation Q)
k_d = 0.0300;                           % cost for deposits
k_a = 0.0466;                           % cost for mmda's
k_dh = 0.0074;                          % for decentralization (household cost)
phi = 0.0057;                           % time cost
k_db = k_d-k_dh;                        % for decentralization (bank cost)
A_star = 0.4138;                        % free parameter A (money balances is 25% of GDP when r = 6%)

% No changes here
inputs(2,1) = eta;
inputs(2,2) = theta_c;
inputs(2,3) = theta_a;
inputs(2,4) = theta_d;
inputs(2,5) = r_star;
inputs(2,6) = k_d;
inputs(2,7) = k_a;
inputs(2,8) = k_dh;
inputs(2,9) = phi;
inputs(2,10) = k_db;
inputs(2,11) = A_star;

% INPUTS: Calibration 3 (choose values)
eta = 1.5621;                           % parameter of the distribution
theta_c = 1.01;                         % loss rate
theta_a = .01;                          % reserve requirement for MMDA
theta_d = .1;                           % reserve requirement for demand deposits
r_star = 0.000;                         % interet rate (Regulation Q)
k_d = 0.0300;                           % cost for deposits
k_a = 0.0433;                           % cost for mmda's
k_dh = 0.0074;                          % for decentralization (household cost)
phi = 0.0057;                           % time cost
k_db = k_d-k_dh;                        % for decentralization (bank cost)
A_star = 0.4138;                        % free parameter A (money balances is 25% of GDP when r = 6%)

% No changes here
inputs(3,1) = eta;
inputs(3,2) = theta_c;
inputs(3,3) = theta_a;
inputs(3,4) = theta_d;
inputs(3,5) = r_star;
inputs(3,6) = k_d;
inputs(3,7) = k_a;
inputs(3,8) = k_dh;
inputs(3,9) = phi;
inputs(3,10) = k_db;
inputs(3,11) = A_star;

% INPUTS: Calibration 4 (choose values)
eta = 1.5621;                           % parameter of the distribution
theta_c = 1.01;                         % loss rate
theta_a = .01;                          % reserve requirement for MMDA
theta_d = .1;                           % reserve requirement for demand deposits
r_star = 0.000;                         % interet rate (Regulation Q)
k_d = 0.0300;                           % cost for deposits
k_a = 0.0400;                           % cost for mmda's
k_dh = 0.0074;                          % for decentralization (household cost)
phi = 0.0057;                           % time cost
k_db = k_d-k_dh;                        % for decentralization (bank cost)
A_star = 0.4138;                        % free parameter A (money balances is 25% of GDP when r = 6%)

% No changes here
inputs(4,1) = eta;
inputs(4,2) = theta_c;
inputs(4,3) = theta_a;
inputs(4,4) = theta_d;
inputs(4,5) = r_star;
inputs(4,6) = k_d;
inputs(4,7) = k_a;
inputs(4,8) = k_dh;
inputs(4,9) = phi;
inputs(4,10) = k_db;
inputs(4,11) = A_star;

%% DATA

% Importing Data
load data_consolidated

% HP filter smoothing parameter
hpsmooth = 100;

% Trend components of monetary aggregates (for the period 1984 - 2012)
m1j_trend       = hp_trend(m1j(years>=1984),hpsmooth);              % New M1 (% of GDP) 
currsl_trend    = hp_trend(currsl(years>=1984),hpsmooth);           % Currency (% of GDP)
deprs_trend     = hp_trend(deprs(years>=1984),hpsmooth);            % Deposits (% of GDP)
mmdars_trend    = hp_trend(mmdars(years>=1984),hpsmooth);           % MMDA (% of GDP)
tbills_trend    = hp_trend(tbills3m(years>=1984)./100,hpsmooth);    % Interest Rate

% Only r>=0
for i = 1:length(tbills_trend)
    if tbills_trend(i) < 0
        tbills_trend(i) = 0.0001;
    end 
end

%% Interst Rate and Grids

% Interest Rate
r       = tbills_trend(1);              % nominal interest rate (in 1984) hp filter

% Parameter for optimization
limit   = 10e8;                         % for optimizations

% GRIDS
% Grid 1 (g1): for a grid of interest rate
r_g1 = linspace(0.0001,.15,100)';
gamma_g1 = zeros(1,length(r_g1))';
delta_g1 = zeros(1,length(r_g1))';
n_g1 = zeros(1,length(r_g1))';
% Grid 2 (g2): for the series of interest rate (1915 - 2012) - actual data
r_g2 = tbills3m./100;
gamma_g2 = zeros(1,length(r_g2))';
delta_g2 = zeros(1,length(r_g2))';
n_g2 = zeros(1,length(r_g2))';
% Grid 3 (g3): for the series of interest rate (1984 - 2012) - hp filter
r_g3 = tbills_trend;
gamma_g3 = zeros(1,length(r_g3))';
delta_g3 = zeros(1,length(r_g3))';
n_g3 = zeros(1,length(r_g3))';
% Grid 1 (g1): for a grid of interest rate (decentralized economy)
r_g1d = linspace(.0001,.15,200)';
gamma_g1d = zeros(1,length(r_g1d))';
n_g1d = zeros(1,length(r_g1d))';

% Grid with results (for different calibrations)
calib_g1s1 = zeros(4,length(r_g1))';
calib_g3s2 = zeros(4,length(r_g3))';
calib_g2s2 = zeros(4,length(r_g2))';
calib_g3s3 = zeros(4,length(r_g3))';
calib_g2s3 = zeros(4,length(r_g2))';
calib_g1s1d = zeros(4,length(r_g1d))';
calib_g1s2d = zeros(4,length(r_g1d))';

%% LOOP

for ic = 1:Nc

% Parameter values
eta = inputs(ic,1);
theta_c = inputs(ic,2);
theta_a = inputs(ic,3);
theta_d = inputs(ic,4);
r_star = inputs(ic,5);
k_d = inputs(ic,6);
k_a = inputs(ic,7);
k_dh = inputs(ic,8);
phi = inputs(ic,9);
k_db = inputs(ic,10);
A_star = inputs(ic,11);

% ENDOGENOUS VARIABLES AND INTEREST RATE
mu = 1/(eta-1);
Fz = @(x) 1-(1+x)^(-eta);
Omegaz = @(x) (1-((1+x*(eta))/((1+x)^(eta))));

% PART I - CENTRALIZED ECONOMY: SIMULATION

% Kappa function
kappa = @(x) ((((theta_c-1)/x)+theta_c-theta_d)/(theta_d-theta_a))*((k_a-k_d)/(k_d));

% For each r in the grid, we solve for gamma, and then we compute n. We first define the
% following functions (left hand side and right hand side of...):
faux = @(x) (1/mu)*(theta_c-1+x*(theta_c-theta_d))/(k_d);
flhs = @(x,y) (((x*faux(y))^2)*phi)/(1-phi*x*faux(y));
frhsu = @(x,y) (theta_c-1)*Omegaz(x)+y*(theta_c*Omegaz(x)+theta_d*(Omegaz(kappa(y)*x)-Omegaz(x))+theta_a*(1-Omegaz(kappa(y)*x)));
frhsd = @(x,y) 1+k_d*(Fz(kappa(y)*x)-Fz(x))+k_a*(1-Fz(kappa(y)*x));

% COMPUTATION
epsilon = 1e-5;
% Grid1
for i = 1:length(r_g1)
    limit = (mu/phi)*(k_d)/(theta_c-1+r_g1(i)*(theta_c-theta_d));
    gamma_g1(i) = fzero(@(x) flhs(x,r_g1(i))-(frhsu(x,r_g1(i))/frhsd(x,r_g1(i))),[0,limit-epsilon]);
    n_g1(i) = gamma_g1(i)*(1/mu)*(theta_c-1+r_g1(i)*(theta_c-theta_d))/(k_d);
    delta_g1(i) = kappa(r_g1(i))*gamma_g1(i);
end
% Grid2
for i = 1:length(r_g2)
    limit = (mu/phi)*(k_d)/(theta_c-1+r_g2(i)*(theta_c-theta_d));
    gamma_g2(i) = fzero(@(x) flhs(x,r_g2(i))-(frhsu(x,r_g2(i))/frhsd(x,r_g2(i))),[0,limit-epsilon]);
    n_g2(i) = gamma_g2(i)*(1/mu)*(theta_c-1+r_g2(i)*(theta_c-theta_d))/(k_d);
    delta_g2(i) = kappa(r_g2(i))*gamma_g2(i);
end
% Grid3
for i = 1:length(r_g3)
    limit = (mu/phi)*(k_d)/(theta_c-1+r_g3(i)*(theta_c-theta_d));
    gamma_g3(i) = fzero(@(x) flhs(x,r_g3(i))-(frhsu(x,r_g3(i))/frhsd(x,r_g3(i))),[0,limit-epsilon]);
    n_g3(i) = gamma_g3(i)*(1/mu)*(theta_c-1+r_g3(i)*(theta_c-theta_d))/(k_d);
    delta_g3(i) = kappa(r_g3(i))*gamma_g3(i);
end

% Series: g1 - grid1, g2 - grid2 and g3 - grid3
% s1: money balances as a % of GDP
% s2: ratio currency to deposits
% s3: ratio dep/(dep+mmdas)
% Grid1
g1s1 = zeros(1,length(r_g1))';
g1s2 = zeros(1,length(r_g1))';
g1s3 = zeros(1,length(r_g1))';
for i = 1:length(r_g1)
    g1s1(i) = A_star*((theta_c-1)*Omegaz(gamma_g1(i))+1)/n_g1(i);
    g1s2(i) = (theta_c*Omegaz(gamma_g1(i)))/(Omegaz(delta_g1(i))-Omegaz(gamma_g1(i)));
    g1s3(i) = (Omegaz(delta_g1(i))-Omegaz(gamma_g1(i)))/(1-Omegaz(gamma_g1(i)));
end
% Grid2
g2s1 = zeros(1,length(r_g2))';
g2s2 = zeros(1,length(r_g2))';
g2s3 = zeros(1,length(r_g2))';
for i = 1:length(r_g2)
    g2s1(i) = A_star*((theta_c-1)*Omegaz(gamma_g2(i))+1)/n_g2(i);
    g2s2(i) = (theta_c*Omegaz(gamma_g2(i)))/(Omegaz(delta_g2(i))-Omegaz(gamma_g2(i)));
    g2s3(i) = (Omegaz(delta_g2(i))-Omegaz(gamma_g2(i)))/(1-Omegaz(gamma_g2(i)));
end
% Grid3
g3s1 = zeros(1,length(r_g3))';
g3s2 = zeros(1,length(r_g3))';
g3s3 = zeros(1,length(r_g3))';
for i = 1:length(r_g3)
    g3s1(i) = A_star*((theta_c-1)*Omegaz(gamma_g3(i))+1)/n_g3(i);
    g3s2(i) = (theta_c*Omegaz(gamma_g3(i)))/(Omegaz(delta_g3(i))-Omegaz(gamma_g3(i)));
    g3s3(i) = (Omegaz(delta_g3(i))-Omegaz(gamma_g3(i)))/(1-Omegaz(gamma_g3(i)));
end

% PART 2 - DECENTRALIZED ECONOMY: SIMULATION

% Case 1
case1_faux = @(x) ((theta_c-1)+x*(theta_c-theta_d))/(mu*k_d);
case1_lhs = @(x,y) (((x*case1_faux(y))^2)/(1-phi*x*case1_faux(y)))*phi;
case1_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));
% Case 2
case2_faux = @(x) ((1+x)*(theta_c-1)+r_star)/(mu*k_dh);
case2_lhs = @(x,y) (((x*case2_faux(y))^2)/(1-phi*x*case2_faux(y)))*phi;
case2_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));
% GRIDS
% Series
% s1: money balances as a % of GDP
% s2: ratio currency to M1
g1s1d = zeros(1,length(r_g1d))';
g1s2d = zeros(1,length(r_g1d))';

% COMPUTATION
for i = 1:length(r_g1d)
    % Checking if Regulation Q is binding
    if r_g1d(i)*(1-theta_d) > r_star
        % Case 1
        limit = ((mu*k_d)/phi)*(1/(theta_c-1+r_g1d(i)*(theta_c-theta_d)));
        gamma_g1d(i) = fzero(@(x) case1_lhs(x,r_g1d(i))-case1_rhs(x,r_g1d(i)),[0,limit-epsilon]);
        n_g1d(i) = gamma_g1d(i)*case1_faux(r_g1d(i));
        % Case 2
        if k_d - k_dh < (r_g1d(i)*(1-theta_d)-r_star)*(gamma_g1d(i)/mu)*(1/n_g1d(i))
            limit = ((mu*k_dh)/phi)*(1/((1+r_g1d(i))*(theta_c-1)+r_star));
            gamma_g1d(i) = fzero(@(x) case2_lhs(x,r_g1d(i))-case2_rhs(x,r_g1d(i)),[0,limit-epsilon]);
            n_g1d(i) = gamma_g1d(i)*case2_faux(r_g1d(i));
        end
    else
        limit = (mu/phi)*(k_d)/(theta_c-1+r_g1d(i)*(theta_c-theta_d));
        gamma_g1d(i) = fzero(@(x) flhs(x,r_g1d(i))-(frhsu(x,r_g1d(i))/frhsd(x,r_g1d(i))),[0,limit-epsilon]);
        n_g1d(i) = gamma_g1d(i)*(1/mu)*(theta_c-1+r_g1d(i)*(theta_c-theta_d))/(k_d);
    end
    g1s1d(i) = A_star*((theta_c-1)*Omegaz(gamma_g1d(i))+1)/n_g1d(i);
    g1s2d(i) = (theta_c*Omegaz(gamma_g1d(i)))/((theta_c-1)*Omegaz(gamma_g1d(i))+1);
end

calib_g1s1(:,ic) = g1s1;
calib_g3s2(:,ic) = g3s2;
calib_g2s2(:,ic) = g2s2;
calib_g3s3(:,ic) = g3s3;
calib_g2s3(:,ic) = g2s3;
calib_g1s1d(:,ic) = g1s1d;
calib_g1s2d(:,ic) = g1s2d;

end

%% CENTRALIZED ECONOMY: FIGURES

% Figure 7a: Ratio Dep/(Dep+MMDAs) (trend component), 1984 - 2012
figure
p1 = plot(years(years>=1984),deprs_trend./(deprs_trend+mmdars_trend),'k','LineWidth',3); hold on
p2 = plot(years(years>=1984),calib_g3s3(:,1),'x-k');
p3 = plot(years(years>=1984),calib_g3s3(:,2),'^-k');
p4 = plot(years(years>=1984),calib_g3s3(:,3),'o-k');
p5 = plot(years(years>=1984),calib_g3s3(:,4),'+-k'); hold off, grid on
legend([p1 p2 p3 p4 p5],{'data','calib1','calib2','calib3','calib4'},'Location','NorthWest')
%title(upper('ratio Dep / (Dep + mmdas) (trend component), 1984 - 2012'),'Fontsize',14)
set(gca,'YLim',[0,1])
ylabel(upper('ratio'))
orient landscape

% Figure 7b: Money balances as a % of GDP vs. Interest Rate 3MTbill, 1983 - 2012
figure
p1 = plot(r_g1,calib_g1s1(:,1),'x-k'); hold on
p2 = plot(r_g1,calib_g1s1(:,2),'^-k');
p3 = plot(r_g1,calib_g1s1(:,3),'o-k');
p4 = plot(r_g1,calib_g1s1(:,4),'+-k');
scatter(tbills3m(years>=1983)./100,m1j(years>=1983)./100,'filled','k');
scatter(tbills3m(years<=1935)./100,m1j(years<=1935)./100,'filled','k');
legend([p1 p2 p3 p4],{'calib1','calib2','calib3','calib4'},'Location','NorthEast')
%title(upper('money balances as a % of gdp vs. interest rate 3MTbill, 1915 - 1935 & 1983 - 2012'),'Fontsize',14), grid on
set(gca,'YLim',[0.10,0.55],'XLim',[0,0.15])
xlabel(upper('Interest Rate')); ylabel(upper('M1 / GDP'))
orient landscape