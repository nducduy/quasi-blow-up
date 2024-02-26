import numpy as np
import math
from matplotlib import pyplot as plt
from rbergomi import rbergomi
from rSteinStein import rSteinStein
from rSteinStein3 import rSteinStein3



#rStein-Stein model

from utils import utils
#bsinv

#from rbergomi import utils
vec_bsinv = np.vectorize(utils.bsinv)
#%matplotlib inline
#!pip install py_vollib
import py_vollib
from py_vollib.black_scholes  import black_scholes as bs
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv


# stocks are all martingales.

# stocks are all martingales.

#Quasi-blow up
#initial values for sigma
sigma1_0 = 0.2;
sigma2_0= 0.6;
r = 0;
H1 = 0.1;
H2 = 0.7;

#blow up

sigma1_0 = 0.2;
sigma2_0= 0.6;
r = 0;
H1 = 0.1;
H2 = 0.7;



rho1=-0;
rho2=-0;
# time increment. To see quasi blow up, set dt = 0.1*1/365
dt = 0.1*1/365;

maturity = np.arange(1,3000,10);#% 1:10:2000;
maturity_len = len(maturity);
maturity_max = maturity[-1]; #Get the last element

#derivative of implied vol at ATM
derivative = np.zeros(maturity_len)-10;

#simulate many paths
num_paths = 5000;

# small number
eps = 0.001;

stock1_paths = np.zeros((num_paths,maturity_max));
sigma1_paths = np.zeros((num_paths, maturity_max));
log_stock1_paths = np.zeros((num_paths, maturity_max));
stock2_paths = np.zeros((num_paths, maturity_max));
sigma2_paths = np.zeros((num_paths, maturity_max));
log_stock2_paths = np.zeros((num_paths, maturity_max));


index = np.zeros((num_paths, maturity_max));

# initial values
S0stock1=10;
S0stock2=9.8;
stock1_paths[:,0] = S0stock1;
stock2_paths[:,0] = S0stock2;

#initial sigma1,sigma2

sigma1_paths[:,0]=sigma1_0;
sigma2_paths[:,0]=sigma2_0;

#weights for index
w1 = 1;
w2 = 0;
index[:,0] = w1*stock1_paths[0,0]+ w2*stock2_paths[0,0];



MaturityT=dt*maturity_max; # %time to maturity
ndt=math.floor(1/dt); #round 1/dt to an integer
#TimeT=(maturity_max-1)*dt;

# generate the (S1,V1) sample paths

rSS1 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H1,X0=sigma1_0,S0=S0stock1,rho=rho1);
V1,S1 = rSS1.simul_paths()

sigma1_paths = np.transpose(V1);
stock1_paths = np.transpose(np.exp(S1)); #because simul_paths returns log price




# generate the (S21,V21) sample paths
rSS2 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H2,X0=sigma2_0,S0=S0stock2,rho=rho2);
V2,S2 = rSS2.simul_paths()

sigma2_paths = np.transpose(V2);
stock2_paths = np.transpose(np.exp(S2)); #because simul_paths returns log price




# initiate stocks array
stocks = np.zeros((num_paths, 2));

for j in range(1, maturity_max - 1):  # range must be 1---> to end here
    # Compute index
    # stocks=np.array([stock1_paths[p,j],stock2_paths[p,j]])
    stocks[:, 0] = stock1_paths[:, j]
    stocks[:, 1] = stock2_paths[:, j]
    # descending sort
    stocks = -stocks;
    stocks.sort();
    stocks = -stocks;
    # index
    index[:, j] = w1 * stocks[:, 0] + w2 * stocks[:, 1];

# derivative of implied vol at ATM
derivative = np.zeros((1, maturity_len));
ATM_implied_vol = np.zeros((1, maturity_len));

# compute index_hat = E[index_{maturity}|F_0],
# which is the price of index future at 0
index_hat = np.zeros((1, maturity_len));

for m in range(0, maturity_len - 1):
    index_hat[0, m] = np.mean(index[:, maturity[m]]);

for m in range(0, maturity_len - 1):
    # compute option price
    # %ATM strike is the price of index future with matrutity maturity(1,m)
    I = index_hat[0, m];

    # I = stock1_paths[0,0];
    # % consider two strikes only
    strike = np.zeros((1, 2))
    strike[0, 0] = I - eps;
    strike[0, 1] = I;  # [I-eps I];
    log_moneyness = np.log(strike / I);
    strike_len = np.max(strike.shape);
    implied_vol = np.zeros((1, strike_len));

    # % time to maturity
    TT = maturity[m] * dt;

    for s in range(0, strike_len):
        # % index option price using Monte Carlo
        payoff_index = index[:, maturity[m]] - strike[0, s];
        payoff_index = payoff_index[payoff_index > 0];
        index_call_price = np.sum(payoff_index) / num_paths;
        # %Volatility = blsimpv(Price,Strike,Rate,Time,Value)
        # % compute implied vol, price is now index_hat
        implied_vol[0, s] = iv(index_call_price, I, strike[0, s], maturity[m] * dt, 0,
                               'c');  # imp_v = iv(price, S, K, t, r, flag='c','p')
        # call_price_list[0,s] = index_call_price ;
        # % implied vol at ATM
        if strike[0, s] == I:
            ATM_implied_vol[0, m] = iv(index_call_price, I, strike[0, s], TT, 0, 'c');
        # discret time derivative of implied vol
        derivative[0, m] = (implied_vol[0, 0] - implied_vol[0, 1]) / (log_moneyness[0, 0] - log_moneyness[0, 1]);

#Fit the Power Regression Model
# this gives quasi blow up
p = np.polyfit(np.log(maturity[2:(maturity_len-1)]*dt),np.log(np.abs(derivative[0,2:(maturity_len-1)])),1);
m_fit = round(p[0],3);
b_fit = np.exp(p[1]);

plt.plot(maturity[2:(maturity_len-1)]*dt,np.abs(derivative[0,2:(maturity_len-1)]),'*',label='simulated skew');
plt.plot(maturity[2:(maturity_len-1)]*dt,b_fit*(maturity[2:(maturity_len-1)]*dt)**(m_fit),label=r"fitted curve $T^\alpha,\alpha=$"+ str(m_fit));
plt.legend()
plt.xlabel(r"$T$")
plt.ylabel("|Skew|")
plt.savefig('rBergomi_2stocks.eps', format='eps')
plt.show()