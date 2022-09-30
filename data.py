import numpy as np
import pandas as pd
import math

class Data:
    def __init__(self):
        self.df = pd.read_csv("data.csv")
        self.km =  np.array(self.df["km"])
        self.km_avg = np.average(self.km)
        self.km_std_dev = math.sqrt( np.sum( (self.km - self.km_avg)**2 ) / self.km.size)
        self.norm_km = np.array(self.normalize(self.km))
        
        self.price =  np.array(self.df["price"])
        self.price_avg = np.average(self.price)
        self.price_std_dev = math.sqrt( np.sum( (self.price - self.price_avg)**2 ) / self.price.size)
        self.norm_price = np.array(self.normalize(self.price))

    def normalize(self,data):
        max = np.max(data)
        min = np.min(data)
        mean = np.mean(data)
        return np.divide(np.subtract(data, mean), (max - min))

    def denormalize(self, data, norm_data):
        max = np.max(data)
        min = np.min(data)
        mean = np.mean(data)
        return np.add(np.multiply(norm_data, (max - min)), mean)
    
    def normalize_km(self, km) :
        return (km - self.km_avg) / self.km_std_dev

    def denormalize_price(self, normalized_price) :
        return (normalized_price * self.price_std_dev) + self.price_avg