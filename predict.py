from data import Data
import numpy as np
import matplotlib.pyplot as plt
import json

try:
    with open("thetas.json", "r") as infile:
        thetas = json.load(infile)
except:
    thetas = {"theta0":0,"theta1":0,}
    # print("Run train.py to generate theta values. Current values are set to 0")
    
kms = int(input("Provide a mileage : \n"))
data = Data()

def predict_nrm(km_nrm, w, b) -> float :
    return (b + w * km_nrm)

def predict(km, data : Data = Data()) -> float :
    return data.denormalize_price(predict_nrm(data.normalize_km(km), thetas["theta1"], thetas["theta0"]))

if thetas["theta1"] != 0 and thetas["theta0"] != 0:
    prediction = int(predict(kms, data))
    print(f"Price of a {kms}kms'car : {prediction}€")
else:
    prediction = 0
    print(f"This prediction is not accurate since theta values were not generated. \nPrice of a {kms}kms'car : {prediction}€")
    exit()

plt.scatter(data.km, data.price, color = 'r', label="Sample Data")
x = np.array([0, 250000])
y = predict(x)
plt.plot(x,y, 'b-', label="Prediction")
plt.title(f"Prediction of the price of a {kms}km's car")

plt.plot(kms, prediction, 'g', marker='X', markersize=10, label=f"Estimated price") 
plt.vlines(x = kms, ymin = 0, ymax = prediction, colors = 'green')
plt.hlines(xmin = 0, y = prediction, xmax = kms, colors = 'green')
plt.legend()
plt.xlabel("Mileage")
plt.ylabel("Price")

plt.show()