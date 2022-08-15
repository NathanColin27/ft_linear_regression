import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


df = pd.read_csv("data.csv")
kms = list(df["km"])
prices = list(df["price"])

# plt.plot(kms, prices, 'o')
plt.scatter(kms, prices)

plt.title("Price of a vehicule depending on its mileage")
plt.xlabel("kms")
plt.ylabel("price")

slope, intercept, r, p, std_err = stats.linregress(kms, prices)

def myfunc(x):
      return slope * x + intercept

mymodel = list(map(myfunc, kms))
plt.plot(kms, mymodel)

plt.show()