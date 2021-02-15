% Codes for the paper: "On the Stability of Money Demand" by
% Robert E. Lucas Jr. (University of Chicago) and
% Juan Pablo Nicolini (FRB-Minneapolis and Universidad Di Tella)
% Written by Manuel Macera (UMN/Fed-Minneapolis)
% Modified by Joao Ayres (UMN/Fed-Minneapolis)
% Date: February 2015

% Description: this file is used for the calibration of the model (for both 
% centralized and decentralized cases) and to generate the following 
% figures: 4a,4b,5a,5b,5c,5d,6,8a,8b,8c


%% DATA

close all; clc

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

%% CENTRALIZED ECONOMY: CALIBRATION

% Exogenous Parameters
theta_c = 1.01;                         % loss rate
theta_a = .01;                          % reserve requirement for MMDA
theta_d = .1;                           % reserve requirement for demand deposits
r = tbills_trend(1);                    % nominal interest rate (in 1984) HP filter
r_star = 0.000;                         % interet rate (Regulation Q)
param.theta_c = theta_c;
param.theta_a = theta_a;
param.theta_d = theta_d;
param.r = r;

% Targets
T(1) =  currsl_trend(1)/m1j_trend(1);             % ratio of cash to money holdings (HP filtered)
T(2) = deprs_trend(1)/mmdars_trend(1);            % ratio of deposits to mmdas (HP filtered)
T(3) = .01;                                       % resource cost as a fraction of GDP
T(4) = .01;                                       % time cost

% For optimizations:
limit   = 10e8;

% We solve for eta
% Initial guess
X0 = 1.8;    % [eta]
% Optimization
options = optimset('Display','iter','MaxFunEvals',5000,'MaxIter',5000);
[X1 RES] = fsolve(@(X) calibration(X,param,T),X0,options);
% Solution
eta = X1;
mu = 1/(eta-1);
Fz = @(x) 1-(1+x)^(-eta);
Omegaz = @(x) (1-((1+x*(eta))/((1+x)^(eta))));
gamma = fzero(@(x) T(1)*(Omegaz(x)*(theta_c-1)+1)-theta_c*Omegaz(x),[0,limit]);
delta = fzero(@(x) T(2)*(1-Omegaz(x))-(Omegaz(x)-Omegaz(gamma)),[0,limit]);
C1 = ((gamma/delta)*(((theta_c-1)/r)+theta_c-theta_d))/(theta_d-theta_a);
C2 = ((C1/(C1+1))*(Fz(delta)-Fz(gamma))) + 1 - Fz(delta);
k_a = T(3)/C2;
k_d = (C1*k_a)/(1+C1);
n = (gamma/mu)*(theta_c-1+r*(theta_c-theta_d))/(k_d);
phi = (1/n)*(T(4)/(1+T(3)+T(4)));

%% CENTRALIZED ECONOMY: SIMULATION

% Kappa function
kappa = @(x) ((((theta_c-1)/x)+theta_c-theta_d)/(theta_d-theta_a))*((k_a-k_d)/(k_d));

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

% Free Parameter A (money balances is 25% of GDP when r = 6%)
r_A = 0.06;
m_gdp = 0.25;
limit = (mu/phi)*(k_d)/(theta_c-1+r_A*(theta_c-theta_d));
gamma_A = fzero(@(x) flhs(x,r_A)-(frhsu(x,r_A)/frhsd(x,r_A)),[0,limit-epsilon]);
n_A = gamma_A*(1/mu)*(theta_c-1+r_A*(theta_c-theta_d))/(k_d);
A_star = (m_gdp*n_A)/((theta_c-1)*Omegaz(gamma_A)+1);

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

%% CENTRALIZED ECONOMY: PLOTS

% Figure 5a: Ratio currency to deposits, 1984 - 2012
figure
plot(years(years>=1984),currsl(years>=1984)./deprs(years>=1984),'k','LineWidth',3), hold on
plot(years(years>=1984),g2s2(years>=1984),'x-k'), hold off, grid on
legend(upper({'data','model'}),'Location','Northwest')
set(gca,'YLim',[0,1.2])
ylabel(upper('ratio'))
orient landscape

% Figure 5b: Ratio currency to deposits (trend component), 1984 - 2012
figure
plot(years(years>=1984),currsl_trend./deprs_trend,'k','LineWidth',3), hold on
plot(years(years>=1984),g3s2,'x-k'), hold off, grid on
legend(upper({'data','model'}),'Location','Northwest')
set(gca,'YLim',[0,1.2])
ylabel(upper('ratio'))
orient landscape

% Figure 5c: Ratio Dep/(Dep+MMDAs), 1984 - 2012
figure
plot(years(years>=1984),deprs(years>=1984)./(deprs(years>=1984)+mmdars(years>=1984)),'k','LineWidth',3), hold on
plot(years(years>=1984),g2s3(years>=1984),'x-k'), hold off, grid on
legend(upper({'data','model'}),'Location','Northwest')
set(gca,'YLim',[0,1])
ylabel(upper('ratio'))
orient landscape

% Figure 5d: Ratio Dep/(Dep+MMDAs) (trend component), 1984 - 2012
figure
plot(years(years>=1984),deprs_trend./(deprs_trend+mmdars_trend),'k','LineWidth',3), hold on
plot(years(years>=1984),g3s3,'x-k'), hold off, grid on
legend(upper({'data','model'}),'Location','Northwest')
set(gca,'YLim',[0,1])
ylabel(upper('ratio'))
orient landscape

% Figure 6: Money balances as a % of GDP vs. Interest Rate 3MTbill, 1915 - 1983 & 1983 - 2012
figure
plot(r_g1,g1s1,'x-k'), hold on
scatter(tbills3m(years>=1983)./100,m1j(years>=1983)./100,'filled','k');
scatter(tbills3m(years<=1935)./100,m1j(years<=1935)./100,'filled','k');
legend(upper({'model','data'}),'Location','Northeast')
set(gca,'YLim',[0.0,0.55],'XLim',[0,0.15])
xlabel(upper('interest rate')); ylabel(upper('M1 / GDP'))
orient landscape

% Figure 8a: Money balances as a % of GDP vs. interest rate 3MTbill, 1935 - 1982
figure
plot(r_g1,g1s1,'x-k'), hold on
scatter(tbills3m(years>= 1935 & years<=1982)./100,m1j(years>= 1935 & years<=1982)./100,'filled','k');
legend(upper({'model','data'}),'Location','Northeast')
set(gca,'YLim',[0.0,0.55],'XLim',[0,0.15])
xlabel(upper('interest rate')); ylabel(upper('M1 / GDP'))
orient landscape

%% CENTRALIZED ECONOMY: PLOTTING FUNCTIONS

% Grids and parameters
gamma_grid = linspace(.01,4,1000)';
n1_low = zeros(1,length(gamma_grid))';
n2_low = zeros(1,length(gamma_grid))';
n1_med = zeros(1,length(gamma_grid))';
n2_med = zeros(1,length(gamma_grid))';
n1_high = zeros(1,length(gamma_grid))';
n2_high = zeros(1,length(gamma_grid))';
r_low = .02;
r_med = .05;
r_high = .09;

% Function
for i = 1:length(gamma_grid)
    n1_low(i) = (gamma_grid(i)/mu)*(1/(k_d))*(theta_c-1+r_low*(theta_c-theta_d));
    A = (frhsu(gamma_grid(i),r_low))/((frhsd(gamma_grid(i),r_low)));
    n2_low(i) = (-A+sqrt((A^2)+4*(1/phi)*A))/2;
    n1_med(i) = (gamma_grid(i)/mu)*(1/(k_d))*(theta_c-1+r_med*(theta_c-theta_d));
    A = frhsu(gamma_grid(i),r_med)/frhsd(gamma_grid(i),r_med);
    n2_med(i) = (-A+sqrt((A^2)+4*(1/phi)*A))/2;
    n1_high(i) = (gamma_grid(i)/mu)*(1/(k_d))*(theta_c-1+r_high*(theta_c-theta_d));
    A = frhsu(gamma_grid(i),r_high)/frhsd(gamma_grid(i),r_high);
    n2_high(i) = (-A+sqrt((A^2)+4*(1/phi)*A))/2;
end

% Figure 4a: functions determining gamma and n for given r (low r):
figure
plot(gamma_grid,n1_low,'k','LineWidth',3), hold on
plot(gamma_grid,n2_low,'-k','LineWidth',3), hold off, 
xlabel('\gamma','Fontsize',17,'Fontweight','bold'), ylabel('n','Fontsize',17,'Fontweight','bold')
set(gca,'XLim',[min(gamma_grid),max(gamma_grid)]) % 'YLim',[0,0.5]
text(2.0,1.5,lower('low r'),'Fontweight','bold','Fontname','arial narrow','Fontsize',16,'Color','k')
orient landscape

% Figure 4b: functions determining gamma and n for given r (low r, medium r, high r):
figure
plot(gamma_grid,n1_low,'-k','LineWidth',3), hold on
plot(gamma_grid,n1_med,'--k','LineWidth',3), 
plot(gamma_grid,n2_low,'-k','LineWidth',3), 
plot(gamma_grid,n2_med,'--k','LineWidth',3)
plot(ones(1,20)*2.55,linspace(0,1.9,20),'--k','LineWidth',1)
plot(gamma_grid,n2_med,'--k','LineWidth',3), hold off
xlabel('\gamma','Fontsize',17,'Fontweight','bold'), ylabel('n','Fontsize',17,'Fontweight','bold')
set(gca,'XLim',[min(gamma_grid),max(gamma_grid)],'YLim',[0,4])
text(3.1,1.25,lower('low r'),'Fontweight','bold','Fontname','arial narrow','Fontsize',16,'Color','k')
text(1.6,2.35,lower('high r'),'Fontweight','bold','Fontname','arial narrow','Fontsize',16,'Color','k')
text(2.47,2.13,'A','Fontweight','bold','Fontname','arial narrow','Fontsize',18,'Color','k')
text(1.41,1.83,'B','Fontweight','bold','Fontname','arial narrow','Fontsize',18,'Color','k')
orient landscape

%% DECENTRALIZED ECONOMY: CALIBRATION

% New parameters: k_d = k_dh + k_db
% We set k_dh such that the minimum of the ratio Currency to M1 reaches
% its minimum at r_target.

r_target = 0.035;   % r target
k_dh1 = 0;
k_dh2 = k_d;
% grid for r (bounds + distance between points)
r1 = 0.01;         % lower bound  
r2 = 0.05;         % upper bound
n_grid = 1000;     % size of the grid
% Grid
r_g1d = linspace(r1,r2,n_grid)';
gamma_g1d = zeros(1,length(r_g1d))';
n_g1d = zeros(1,length(r_g1d))';
% Series
% s2: ratio currency to M1
g1s2d = zeros(1,length(r_g1d))';
r_min = 0;

% Case 1 (in this case, the solution does not depend on k_dh)
case1_faux = @(x) ((theta_c-1)+x*(theta_c-theta_d))/(mu*k_d);
case1_lhs = @(x,y) (((x*case1_faux(y))^2)/(1-phi*x*case1_faux(y)))*phi;
case1_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));

while abs(r_min - r_target) >= 1e-4
    k_dh = (k_dh1 + k_dh2)/2;    
    % Case 2
    case2_faux = @(x) ((1+x)*(theta_c-1)+r_star)/(mu*k_dh);
    case2_lhs = @(x,y) (((x*case2_faux(y))^2)/(1-phi*x*case2_faux(y)))*phi;
    case2_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));
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
            end
       else
       limit = (mu/phi)*(k_d)/(theta_c-1+r_g1d(i)*(theta_c-theta_d));
       gamma_g1d(i) = fzero(@(x) flhs(x,r_g1d(i))-(frhsu(x,r_g1d(i))/frhsd(x,r_g1d(i))),[0,limit-epsilon]);
       end
       % Ratio currency to M1
       g1s2d(i) = (theta_c*Omegaz(gamma_g1d(i)))/((theta_c-1)*Omegaz(gamma_g1d(i))+1);
    end    
    [a loc] = min(g1s2d);
    r_min = r_g1d(loc);
    if r_min > r_target
        k_dh1 = k_dh;
    else
        k_dh2 = k_dh;
    end
end

%% DECENTRALIZED ECONOMY: SIMULATION

% Case 1
case1_faux = @(x) ((theta_c-1)+x*(theta_c-theta_d))/(mu*k_d);
case1_lhs = @(x,y) (((x*case1_faux(y))^2)/(1-phi*x*case1_faux(y)))*phi;
case1_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));
% Case 2
case2_faux = @(x) ((1+x)*(theta_c-1)+r_star)/(mu*k_dh);
case2_lhs = @(x,y) (((x*case2_faux(y))^2)/(1-phi*x*case2_faux(y)))*phi;
case2_rhs = @(x,y) ((theta_c-1)*Omegaz(x) + y*(1+(theta_c-1)*Omegaz(x))-r_star*(1-Omegaz(x)))/(1+k_d*(1-Fz(x)));
% GRIDS
% Grid 1 (g1): for a grid of interest rate
r_g1d = linspace(.0001,.15,200)';
gamma_g1d = zeros(1,length(r_g1d))';
n_g1d = zeros(1,length(r_g1d))';
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

%% DECENTRALIZED ECONOMY: PLOTS

% Figure 8b: Money balances as a % of GDP vs. Interest Rate 3MTbill, 1935 - 1982
figure
plot(r_g1d,g1s1d,'x-k'), hold on
scatter(tbills3m(years>= 1935 & years<=1982)./100,m1j(years>= 1935 & years<=1982)./100,'k','filled');
set(gca,'YLim',[0.0,0.55],'XLim',[0,0.15])
legend(upper({'model','data'}),'Location','Northeast')
xlabel(upper('interest rate')); ylabel(upper('M1 / GDP'))
orient landscape

% Figure 8c: Curr/M1 vs. Interest Rate 3MTbill, 1935 - 1981
figure
plot(r_g1d,g1s2d,'x-k'), hold on
scatter(tbills3m(years>= 1935 & years<=1982)./100,currsl(years>= 1935 & years<=1982)./m1j(years>= 1935 & years<=1982),'k','filled');
set(gca,'YLim',[0,1],'XLim',[0,0.15])
legend(upper({'model','data'}),'Location','Northwest')
xlabel(upper('interest rate')); ylabel(upper('Currency / M1'))
orient landscape


%% Parameters

param.theta_c = theta_c;
param.theta_d = theta_d;
param.theta_a = theta_a;
param.eta = eta;
param.phi = phi;
param.k_a = k_a;
param.k_d = k_d;
param.k_dh = k_dh;
param.k_db = k_d - k_dh;
param.A = A_star;
param.gamma = gamma;
param.delta = delta;
param.n = n;
param.rstar = r_star;
disp(param)
disp(T)