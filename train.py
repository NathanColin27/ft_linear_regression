from data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

data = Data()

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (array): Data, m examples 
      y (array): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.size
    total_cost = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i]) ** 2
        total_cost = total_cost + cost
    total_cost = (1 / (2 * m)) * total_cost
    
    return total_cost

def compute_gradient(x, y , w, b):
    m = x.size
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
    
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      alpha (float):     Learning rate
      iters (int):   number of iterations to run gradient descent
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
    """
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

alpha = 0.01
iters = 10000

w_final, b_final = gradient_descent(data.norm_km, data.norm_price, 0, 0, alpha, iters)

def predict_nrm(km_nrm, w, b) -> float :
    return (b + w * km_nrm)

def predict(km, data : Data = Data()) -> float :
    return data.denormalize_price(predict_nrm(data.normalize_km(km), w_final, b_final))

def plot(data : Data = Data()):
    plt.scatter(data.km, data.price, color = 'r', marker='x')
    x = np.array([0, 250000])
    y = predict(x)
    plt.plot(x,y, 'b-')
    plt.show()

thetas = {
    "theta0": b_final,
    "theta1": w_final,
}
 
with open("thetas.json", "w") as outfile:
    json.dump(thetas, outfile)


    
plot(data)
print(predict(150000, data))






