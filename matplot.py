import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


df = pd.read_csv("data.csv")
kms = list(df["km"])
prices = list(df["price"])

kms_mean = np.mean(kms)
prices_mean = np.mean(prices)

# plt.plot(kms, prices, 'o')
plt.scatter(kms, prices)

plt.title("Price of a vehicule depending on its mileage")
plt.xlabel("kms")
plt.ylabel("price")

def find_param_1(kms, prices, kms_mean, prices_mean):
    dev = 0
    
    for i in range(len(kms)):
        dev += (kms[i] - kms_mean) * (prices[i] - prices_mean)
    print(dev)
    return dev

def find_param2(kms, kms_mean):
    dev = 0
    for i in range(len(kms)):
        dev += (kms[i] - kms_mean)**2
    print(dev)
    return dev
try:
    slope, intercept, r, p, std_err = stats.linregress(kms, prices)
    def myfunc(x):
        return slope * x + intercept
    
    print("slope :", slope)
    print("intercept :", intercept)
    
    mymodel = list(map(myfunc, kms))
    plt.plot(kms, mymodel)
    
    param1 = find_param_1(kms, prices,  kms_mean, prices_mean)
    param2 = find_param2(kms, kms_mean)
    
    my_slope = param2/param1
    print(my_slope)
    # plt.show()
except KeyboardInterrupt:
    exit()