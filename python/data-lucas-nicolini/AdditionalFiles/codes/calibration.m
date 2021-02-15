function dist = calibration(X,param,T)

% we are solving for the following variables
eta = X(1);

% Restriction 1
if eta < 1
    dist = 1e15;
    return
end

% Parameters
r       = param.r;
theta_a = param.theta_a;
theta_c = param.theta_c;
theta_d = param.theta_d;
limit   = 10e7;

% Distribution
mu = 1/(eta-1);
Fz = @(x) 1-(1+x)^(-eta);
Omegaz = @(x) (1-((1+x*(eta))/((1+x)^(eta))));

% solving for gamma
gamma = fzero(@(x) T(1)*(Omegaz(x)*(theta_c-1)+1)-theta_c*Omegaz(x),[0,limit]);

% solving for delta
delta = fzero(@(x) T(2)*(1-Omegaz(x))-(Omegaz(x)-Omegaz(gamma)),[0,limit]);

% solving for k_a
C1 = ((gamma/delta)*(((theta_c-1)/r)+theta_c-theta_d))/(theta_d-theta_a);
C2 = ((C1/(C1+1))*(Fz(delta)-Fz(gamma))) + 1 - Fz(delta);
k_a = T(3)/C2;

% solving for k_d
k_d = (C1*k_a)/(1+C1);

% solving for n
n = (gamma/mu)*(theta_c-1+r*(theta_c-theta_d))/(k_d);

% solving for phi
phi = (1/n)*(T(4)/(1+T(3)+T(4)));

% Distance
C3 = (theta_c-1)*Omegaz(gamma)+r*(theta_c*Omegaz(gamma)+theta_d*(Omegaz(delta)-Omegaz(gamma))+theta_a*(1-Omegaz(delta)));
LHS = (phi*(n^2))/(1-phi*n);
RHS = C3/(1+T(3));
dist = limit*(LHS-RHS);