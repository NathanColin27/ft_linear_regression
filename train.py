from data import Data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

data = Data()

def compute_gradient(x, y , w, b):
    """
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : Target values
      w (float)         : Current w value
      b (float)         : Current b value
    
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
    """
    
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

def gradient_descent(x, y, w_init, b_init, alpha, iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : Target values
      w_init (float)    : Parameter initial value
      b_init (float)    : Parameter initial value
      alpha (float)     : Learning rate
      iters (int)       : Number of iterations to run gradient descent
    
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
    """

    w = w_init
    b = w_init

    for i in range(iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

alpha = 0.01
iters = 10000

w_final, b_final = gradient_descent(data.norm_km, data.norm_price, 0, 0, alpha, iters)

thetas = {
    "theta0": b_final,
    "theta1": w_final,
}
 
with open("thetas.json", "w") as outfile:
    json.dump(thetas, outfile)
    
print("θ0 and θ1 saved in thetas.json")
