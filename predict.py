from data import Data
import json

kms = int(input("Provide a mileage : \n"))
data = Data()
with open("thetas.json", "r") as infile:
    thetas = json.load(infile)

def predict_nrm(km_nrm, w, b) -> float :
    return (b + w * km_nrm)

def predict(km, data : Data = Data()) -> float :
    return data.denormalize_price(predict_nrm(data.normalize_km(km), thetas["theta1"], thetas["theta0"]))

print(predict(kms, data))