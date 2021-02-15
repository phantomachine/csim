% Codes for the paper: "On the Stability of Money Demand" by
% Robert E. Lucas Jr. (University of Chicago) and
% Juan Pablo Nicolini (FRB-Minneapolis and Universidad Di Tella)
% Written by Manuel Macera (UMN/Fed-Minneapolis)
% Modified by Joao Ayres (UMN/Fed-Minneapolis)
% Date: February 2015

% Description: this file generetes the figures (1a,1b,1c,2a,2b,2c,3) with 
% time series data for the US

%% Data

close all; clear all; clc

% Importing Data
load data_consolidated

%% Figures

% Figure 1a: (M1/GDP) vs. Interest Rate (3-Month T-Bill), 1915 - 1980
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,m1j(years> 1910 & years<=1980)./100,'k','filled');
xlabel(upper('interest rate')); ylabel(upper('M1 / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.1,.5],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 1b: Currency as a % of GDP vs. interest rate 3MTBILL, 1915 - 1980
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,currsl(years> 1910 & years<=1980)./100,'k','filled');
xlabel(upper('interest rate')); ylabel(upper('Currency / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.03,.12],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 1c: Demand deposits as a % of GDP vs. interest rate 3MTBILL, 1915 - 1980
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,dep(years> 1910 & years<=1980)./100,'k','filled');
xlabel(upper('interest rate')); ylabel(upper('Demand Deposits / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.05,.4],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 2a: M1 as a % of GDP vs. interest rate 3MTBILL, 1915 - 2012
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,m1sl(years> 1910 & years<=1980)./100,'kd','LineWidth',1.2); hold on
scatter(tbills3m(years> 1980 & years<=2012)./100,m1sl(years> 1980 & years<=2012)./100,'k','filled'); hold off
legend(upper({'1915-1980';'1981-2012'}))
xlabel(upper('interest rate')); ylabel(upper('M1 / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.1,.5],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 2b: Currency as a % of GDP vs. interest rate 3MTBILL, 1915 - 2012
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,currsl(years> 1910 & years<=1980)./100,'kd','LineWidth',1.2); hold on 
scatter(tbills3m(years> 1980 & years<=2012)./100,currsl(years> 1980 & years<=2012)./100,'k','filled'); hold off
legend(upper({'1915-1980';'1981-2012'}))
xlabel(upper('interest rate')); ylabel(upper('Currency / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.03,.12],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 2c: Demand deposits as a % of GDP vs. interest rate 3MTBILL, 1915 - 2012
figure
scatter(tbills3m(years> 1910 & years<=1980)./100,dep(years> 1910 & years<=1980)./100,'kd','LineWidth',1.2); hold on
scatter(tbills3m(years> 1980 & years<=2012)./100,dep(years> 1980 & years<=2012)./100,'k','filled'); hold off
legend(upper({'1915-1980';'1981-2012'}))
xlabel(upper('interest rate')); ylabel(upper('Demand Deposits / GDP'))
set(gca,'XLim',[0,.16],'YLim',[.05,.4],'Ygrid','on','Fontsize',11)
orient landscape

% Figure 3: New-M1 vs. opportunity cost, 1915 - 2012
figure
% We use linear extrapolation for the years where the return is not specified
blin_a = regress(log(ir_mmda(:)),[ones(length(tbills3m),1) log(tbills3m(:))]);
blin_d = regress(log(ir_taccts(:)),[ones(length(tbills3m),1) log(tbills3m(:))]);
rd = exp(blin_d(1)+blin_d(2)*log(tbills3m));
ra = exp(blin_a(1)+blin_a(2)*log(tbills3m));
% For the last two years, we do not have the series of DEPRS and MMDARS
% defined. We assume the ratios are constant in the final periods:
dep_m1 = deprs./m1j;
mmda_m1 = mmdars./m1j;
oppcost = tbills3m - dep_m1.*rd - mmda_m1.*ra;
scatter(tbills3m(years> 1910 & years<=1980)./100,m1j(years> 1910 & years<=1980)./100,'kd','LineWidth',1.2); hold on
scatter(oppcost(years> 1980 & years<=2012)./100,m1j(years> 1980 & years<=2012)./100,'k','filled'); hold off
legend(upper({'1915-1980';'1981-2012'}))
xlabel(upper('opportunity cost (rate)')); ylabel(upper('New M1 / GDP'))
%title(upper('New-M1 vs. opportunity cost, 1915 - 2012'),'Fontsize',11)
set(gca,'XLim',[0,.16],'YLim',[.1,.5],'Ygrid','on','Fontsize',11)
orient landscape