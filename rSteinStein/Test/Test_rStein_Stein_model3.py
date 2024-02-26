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

#initial values for variance process
sigma1_0 = 0.2; #X0
sigma2_0= 0.25; #X0
r = 0;
#blow up
#H1 = 0.7; sigma1_0=0.2, sigma2_0=0.6
#H2 = 0.6;

#quasi-blow up
H1=0.6
H2=0.75

#rho1=-0.5;
#rho2=-0.5;

rho1=-0.5;
rho2=-0.5;

# time increment. To see quasi blow up, set dt = 0.1*1/365
dt = 0.1*1/365;
dt = 0.1/356;

maturity = np.arange(1,2000,10);#% 1:10:2000;
maturity_len = len(maturity);
maturity_max = maturity[-1]; #Get the last element

#derivative of implied vol at ATM
derivative = np.zeros(maturity_len)-10;

#derivative of call price at ATM
derivative_call_k = np.zeros(maturity_len);

#simulate many paths
num_paths = 5000;

# small number
eps = 0.001;

stock1_paths = np.zeros((num_paths,maturity_max));
sigma1_paths = np.zeros((num_paths, maturity_max));
log_stock1_paths = np.zeros((num_paths, maturity_max));

stock21_paths = np.zeros((num_paths, maturity_max));
sigma21_paths = np.zeros((num_paths, maturity_max));
log_stock21_paths = np.zeros((num_paths, maturity_max));

stock22_paths = np.zeros((num_paths, maturity_max));
sigma22_paths = np.zeros((num_paths, maturity_max));
log_stock22_paths = np.zeros((num_paths, maturity_max));

stock23_paths = np.zeros((num_paths, maturity_max));
sigma23_paths = np.zeros((num_paths, maturity_max));
log_stock23_paths = np.zeros((num_paths, maturity_max));

stock24_paths = np.zeros((num_paths, maturity_max));
sigma24_paths = np.zeros((num_paths, maturity_max));
log_stock24_paths = np.zeros((num_paths, maturity_max));


index1 = np.zeros((num_paths, maturity_max));
index2 = np.zeros((num_paths, maturity_max));
index3 = np.zeros((num_paths, maturity_max));
index4 = np.zeros((num_paths, maturity_max));

# initial values
S0stock1=10;
S0stock21=10;
S0stock22=9.980;
S0stock23=9.960;
S0stock24=9.940;

#
# initial values
#S0stock1=1000;
#S0stock21=1000;
#S0stock22=990;
#S0stock23=985;
#S0stock24=980;

stock1_paths[:,0] = S0stock1;
stock21_paths[:,0] = S0stock21;
stock22_paths[:,0] = S0stock22;
stock23_paths[:,0] = S0stock23;
stock24_paths[:,0] = S0stock24;



#initial sigma1,sigma2

sigma1_paths[:,0]=sigma1_0;
sigma21_paths[:,0]=sigma2_0;
sigma22_paths[:,0]=sigma2_0;
sigma23_paths[:,0]=sigma2_0;
sigma24_paths[:,0]=sigma2_0;

#weights for index
w1 = 1;
w2 = 0;
index1[:,0] = w1*stock1_paths[0,0]+ w2*stock21_paths[0,0];
index2[:,0] = w1*stock1_paths[0,0]+ w2*stock22_paths[0,0];
index3[:,0] = w1*stock1_paths[0,0]+ w2*stock23_paths[0,0];
index4[:,0] = w1*stock1_paths[0,0]+ w2*stock24_paths[0,0];



MaturityT=dt*maturity_max; # %time to maturity
ndt=math.floor(1/dt); #round 1/dt to an integer
#TimeT=(maturity_max-1)*dt;


# generate the (S1,V1) sample paths
                               #(n_steps = 100, N = 1000, T = 1.00,H=0.5,X0=1,S0=1,r=0,rho=0):
rSS1 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H1,X0=sigma1_0,S0=S0stock1,rho=rho1);
V1,S1 = rSS1.simul_paths()

sigma1_paths = np.transpose(V1);
stock1_paths = np.transpose(np.exp(S1)); #because simul_paths returns log price

# generate the (S21,V21) sample paths
rSS21 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H2,X0=sigma2_0,S0=S0stock21,rho=rho2);
V21,S21 = rSS21.simul_paths()
sigma21_paths = np.transpose(V21);
stock21_paths = np.transpose(np.exp(S21));



# generate the (S22,V22) sample paths
rSS22 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H2,X0=sigma2_0,S0=S0stock22,rho=rho2);
V22,S22 = rSS22.simul_paths()

sigma22_paths = np.transpose(V22);
stock22_paths = np.transpose(np.exp(S22));

# generate the (S23,V23) sample paths
rSS23 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H2,X0=sigma2_0,S0=S0stock23,rho=rho2);
V23,S23 = rSS23.simul_paths()

sigma23_paths = np.transpose(V23);
stock23_paths = np.transpose(np.exp(S23));


# generate the (S24,V24) sample paths
rSS24 = rSteinStein3.rsteinstein3(n_steps=ndt, N=num_paths, T=MaturityT, H=H2,X0=sigma2_0,S0=S0stock24,rho=rho2);
V24,S24 = rSS24.simul_paths()

sigma24_paths = np.transpose(V24);
stock24_paths = np.transpose(np.exp(S24));



# initiate stocks array
stocks = np.zeros((num_paths, 2));

for j in range(1, maturity_max - 1):  # range must be 1---> to end here
    # Compute index
    # stocks=np.array([stock1_paths[p,j],stock2_paths[p,j]])
    stocks[:, 0] = stock1_paths[:, j]
    stocks[:, 1] = stock21_paths[:, j]
    # descending sort
    stocks = -stocks;
    stocks.sort();
    stocks = -stocks;
    # index
    index1[:, j] = w1 * stocks[:, 0] + w2 * stocks[:, 1];

    stocks[:, 0] = stock1_paths[:, j]
    stocks[:, 1] = stock22_paths[:, j]
    # descending sort
    stocks = -stocks;
    stocks.sort();
    stocks = -stocks;
    # index
    index2[:, j] = w1 * stocks[:, 0] + w2 * stocks[:, 1];

    stocks[:, 0] = stock1_paths[:, j]
    stocks[:, 1] = stock23_paths[:, j]
    # descending sort
    stocks = -stocks;
    stocks.sort();
    stocks = -stocks;
    # index
    index3[:, j] = w1 * stocks[:, 0] + w2 * stocks[:, 1];

    stocks[:, 0] = stock1_paths[:, j]
    stocks[:, 1] = stock24_paths[:, j]
    # descending sort
    stocks = -stocks;
    stocks.sort();
    stocks = -stocks;
    # index
    index4[:, j] = w1 * stocks[:, 0] + w2 * stocks[:, 1];

# derivative of implied vol at ATM
derivative1 = np.zeros((1, maturity_len));
derivative2 = np.zeros((1, maturity_len));
derivative3 = np.zeros((1, maturity_len));
derivative4 = np.zeros((1, maturity_len));

ATM_implied_vol1 = np.zeros((1, maturity_len));
ATM_implied_vol2 = np.zeros((1, maturity_len));
ATM_implied_vol3 = np.zeros((1, maturity_len));
ATM_implied_vol4 = np.zeros((1, maturity_len));

# compute index_hat = E[index_{maturity}|F_0],
# which is the price of index future at 0
index_hat1 = np.zeros((1, maturity_len));
index_hat2 = np.zeros((1, maturity_len));
index_hat3 = np.zeros((1, maturity_len));
index_hat4 = np.zeros((1, maturity_len));

for m in range(0, maturity_len - 1):
    index_hat1[0, m] = np.mean(index1[:, maturity[m]]);
    index_hat2[0, m] = np.mean(index2[:, maturity[m]]);
    index_hat3[0, m] = np.mean(index3[:, maturity[m]]);
    index_hat4[0, m] = np.mean(index4[:, maturity[m]]);

for m in range(0, maturity_len - 1):
    # compute option price
    # %ATM strike is the price of index future with matrutity maturity(1,m)
    I1 = index_hat1[0, m];
    I2 = index_hat2[0, m];
    I3 = index_hat3[0, m];
    I4 = index_hat4[0, m];

    # I = stock1_paths[0,0];
    # % consider two strikes only
    strike1 = np.zeros((1, 2))
    strike1[0, 0] = I1 - eps;
    strike1[0, 1] = I1;  # [I-eps I];
    log_moneyness1 = np.log(strike1 / I1);
    strike_len1 = np.max(strike1.shape);
    implied_vol1 = np.zeros((1, strike_len1));
    call_price_list1 = np.zeros((1, strike_len1));

    strike2 = np.zeros((1, 2))
    strike2[0, 0] = I2 - eps;
    strike2[0, 1] = I2;  # [I-eps I];
    log_moneyness2 = np.log(strike2 / I2);
    strike_len2 = np.max(strike2.shape);
    implied_vol2 = np.zeros((1, strike_len2));
    call_price_list2 = np.zeros((1, strike_len2));

    strike3 = np.zeros((1, 2))
    strike3[0, 0] = I3 - eps;
    strike3[0, 1] = I3;  # [I-eps I];
    log_moneyness3 = np.log(strike3 / I3);
    strike_len3 = np.max(strike3.shape);
    implied_vol3 = np.zeros((1, strike_len3));
    call_price_list3 = np.zeros((1, strike_len3));

    strike4 = np.zeros((1, 2))
    strike4[0, 0] = I4 - eps;
    strike4[0, 1] = I4;  # [I-eps I];
    log_moneyness4 = np.log(strike4 / I4);
    strike_len4 = np.max(strike4.shape);
    implied_vol4 = np.zeros((1, strike_len4));
    call_price_list4 = np.zeros((1, strike_len4));

    # % time to maturity
    TT = maturity[m] * dt;

    for s in range(0, strike_len1):
        # % index option price using Monte Carlo
        payoff_index1 = index1[:, maturity[m]] - strike1[0, s];
        payoff_index1 = payoff_index1[payoff_index1 > 0];
        index_call_price1 = np.sum(payoff_index1) / num_paths;
        # %Volatility = blsimpv(Price,Strike,Rate,Time,Value)
        # % compute implied vol, price is now index_hat
        implied_vol1[0, s] = iv(index_call_price1, I1, strike1[0, s], maturity[m] * dt, 0,
                                'c');  # imp_v = iv(price, S, K, t, r, flag='c','p')
        # call_price_list[0,s] = index_call_price ;
        # % implied vol at ATM
        if strike1[0, s] == I1:
            ATM_implied_vol1[0, m] = iv(index_call_price1, I1, strike1[0, s], TT, 0, 'c');
        # discret time derivative of implied vol
        derivative1[0, m] = (implied_vol1[0, 0] - implied_vol1[0, 1]) / (log_moneyness1[0, 0] - log_moneyness1[0, 1]);

        payoff_index2 = index2[:, maturity[m]] - strike2[0, s];
        payoff_index2 = payoff_index2[payoff_index2 > 0];
        index_call_price2 = np.sum(payoff_index2) / num_paths;
        # %Volatility = blsimpv(Price,Strike,Rate,Time,Value)
        # % compute implied vol, price is now index_hat
        implied_vol2[0, s] = iv(index_call_price2, I2, strike2[0, s], maturity[m] * dt, 0,
                                'c');  # imp_v = iv(price, S, K, t, r, flag='c','p')
        # call_price_list[0,s] = index_call_price ;
        # % implied vol at ATM
        if strike2[0, s] == I2:
            ATM_implied_vol2[0, m] = iv(index_call_price2, I2, strike2[0, s], TT, 0, 'c');
        # discret time derivative of implied vol
        derivative2[0, m] = (implied_vol2[0, 0] - implied_vol2[0, 1]) / (log_moneyness2[0, 0] - log_moneyness2[0, 1]);

        payoff_index3 = index3[:, maturity[m]] - strike3[0, s];
        payoff_index3 = payoff_index3[payoff_index3 > 0];
        index_call_price3 = np.sum(payoff_index3) / num_paths;
        # %Volatility = blsimpv(Price,Strike,Rate,Time,Value)
        # % compute implied vol, price is now index_hat
        implied_vol3[0, s] = iv(index_call_price3, I3, strike3[0, s], maturity[m] * dt, 0,
                                'c');  # imp_v = iv(price, S, K, t, r, flag='c','p')
        # call_price_list[0,s] = index_call_price ;
        # % implied vol at ATM
        if strike3[0, s] == I3:
            ATM_implied_vol3[0, m] = iv(index_call_price3, I3, strike3[0, s], TT, 0, 'c');
        # discret time derivative of implied vol
        derivative3[0, m] = (implied_vol3[0, 0] - implied_vol3[0, 1]) / (log_moneyness3[0, 0] - log_moneyness3[0, 1]);

        payoff_index4 = index4[:, maturity[m]] - strike4[0, s];
        payoff_index4 = payoff_index4[payoff_index4 > 0];
        index_call_price4 = np.sum(payoff_index4) / num_paths;
        # %Volatility = blsimpv(Price,Strike,Rate,Time,Value)
        # % compute implied vol, price is now index_hat
        implied_vol4[0, s] = iv(index_call_price4, I4, strike4[0, s], maturity[m] * dt, 0,
                                'c');  # imp_v = iv(price, S, K, t, r, flag='c','p')
        # call_price_list[0,s] = index_call_price ;
        # % implied vol at ATM
        if strike4[0, s] == I4:
            ATM_implied_vol4[0, m] = iv(index_call_price4, I4, strike4[0, s], TT, 0, 'c');
        # discret time derivative of implied vol
        derivative4[0, m] = (implied_vol4[0, 0] - implied_vol4[0, 1]) / (log_moneyness4[0, 0] - log_moneyness4[0, 1]);

# Fit the Power Regression Model
plt.plot(maturity[3:(maturity_len - 1)] * dt, np.abs(derivative1[0, 3:(maturity_len - 1)]), 'ro-',
         label=r"$S^2_0 =$" + str(S0stock21));
plt.plot(maturity[3:(maturity_len - 1)] * dt, np.abs(derivative2[0, 3:(maturity_len - 1)]), 'bo-',
         label=r"$S^2_0 =$" + str(S0stock22));
plt.plot(maturity[3:(maturity_len - 1)] * dt, np.abs(derivative3[0, 3:(maturity_len - 1)]), 'ko-',
         label=r"$S^2_0 =$" + str(S0stock23));
plt.plot(maturity[3:(maturity_len - 1)] * dt, np.abs(derivative4[0, 3:(maturity_len - 1)]), 'mo-',
         label=r"$S^2_0 =$" + str(S0stock24));
plt.legend()
plt.xlabel(r"$T$")
plt.ylabel("|Skew|")
#plt.show()
#plt.savefig('rBergomi_quasi.pdf', bbox_inches='tight')
plt.savefig('SteinSteindestination_path.eps', format='eps')

#plt.savefig('destination_path.pdf', format='pdf')
plt.show()



