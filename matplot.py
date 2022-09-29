import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math, copy
from scipy import stats

stats.linregress
def normalize(data):
    max = np.max(data)
    min = np.min(data)
    return np.divide(np.subtract(data, min), (max - min))

# def denormalize(data):
#     max = np.max(data)
#     min = np.min(data)
#     return np.divide(np.subtract(data, min), (max - min))

df = pd.read_csv("data.csv")
kms = np.array(df["km"])
prices = np.array(df["price"])


# plt.plot(kms, prices, 'o')
# plt.scatter(kms, prices)

# plt.title("Price of a vehicule depending on its mileage")
# plt.xlabel("kms")
# plt.ylabel("price")

norm_kms = normalize(kms)
norm_prices = normalize(prices)

def gradient_descent(m_now, b_now, kms, prices, L):
    m_gradient = 0 
    b_gradient = 0
    
    n = len(kms)
    
    for i in range(n):
        x = kms[i]
        y = prices[i]
        
        m_gradient += -(2/n) * x * ( y - (m_now * x + b_now))
        b_gradient += -(2/n) *  ( y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m,b

m = 0
b = 0
L = 0.0001
epochs = 1000000

for i in range(epochs):
    m, b = gradient_descent(m,b,norm_kms, norm_prices, L)

print(m,b)
plt.scatter(kms, prices)
# plt.plot(list(np.arange(min(norm_kms), max(norm_kms))), [m * x + b for x in range(min(kms), max(kms))])
# plt.show()
plt.scatter(kms, prices)






















# exit()

# def compute_cost(x, y, w, b): 
#     """
#     Computes the cost function for linear regression.
    
#     Args:
#       x (ndarray (m,)): Data, m examples 
#       y (ndarray (m,)): target values
#       w,b (scalar)    : model parameters  
    
#     Returns
#         total_cost (float): The cost of using w,b as the parameters for linear regression
#                to fit the data points in x and y
#     """
#     # number of training examples
#     m = len(x)
    
#     cost = 0 
#     for i in range(m): 
#         f_wb = w * x[i] + b   
#         cost = cost + (f_wb - y[i]) ** 2  
 
#     total_cost = (1 / (2 * m)) * cost 

#     return total_cost

# def compute_gradient(x, y, w, b): 
#     """
#     Computes the gradient for linear regression 
#     Args:
#       x (ndarray (m,)): Data, m examples 
#       y (ndarray (m,)): target values
#       w,b (scalar)    : model parameters  
#     Returns
#       dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
#       dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
#      """
    
#     # Number of training examples
#     m = x.shape[0]    
#     dj_dw = 0
#     dj_db = 0
    
#     for i in range(m):  
#         f_wb = w * x[i] + b 
#         dj_dw_i = (f_wb - y[i]) * x[i] 
#         dj_db_i = f_wb - y[i] 
#         dj_db += dj_db_i
#         dj_dw += dj_dw_i 
#     dj_dw = dj_dw / m 
#     dj_db = dj_db / m 
        
#     return dj_dw, dj_db

# def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
#     """
#     Performs gradient descent to fit w,b. Updates w,b by taking 
#     num_iters gradient steps with learning rate alpha
    
#     Args:
#       x (ndarray (m,))  : Data, m examples 
#       y (ndarray (m,))  : target values
#       w_in,b_in (scalar): initial values of model parameters  
#       alpha (float):     Learning rate
#       num_iters (int):   number of iterations to run gradient descent
#       cost_function:     function to call to produce cost
#       gradient_function: function to call to produce gradient
      
#     Returns:
#       w (scalar): Updated value of parameter after running gradient descent
#       b (scalar): Updated value of parameter after running gradient descent
#       J_history (List): History of cost values
#       p_history (list): History of parameters [w,b] 
#       """
    
#     w = copy.deepcopy(w_in) # avoid modifying global w_in
#     # An array to store cost J and w's at each iteration primarily for graphing later
#     J_history = []
#     p_history = []
#     b = b_in
#     w = w_in
    
#     for i in range(num_iters):
#         # Calculate the gradient and update the parameters using gradient_function
#         dj_dw, dj_db = gradient_function(x, y, w , b)     

#         # Update Parameters using equation (3) above
#         b = b - alpha * dj_db                            
#         w = w - alpha * dj_dw                            

#         # Save cost J at each iteration
#         if i<100000:      # prevent resource exhaustion 
#             J_history.append( cost_function(x, y, w , b))
#             p_history.append([w,b])
#         # Print cost every at intervals 10 times or as many iterations if < 10
#         if i% math.ceil(num_iters/10) == 0:
#             print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
#                   f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
#                   f"w: {w: 0.3e}, b:{b: 0.5e}")
 
#     return w, b, J_history, p_history #return w and J,w history for graphing

# # initialize parameters
# w_init = 0
# b_init = 0
# # some gradient descent settings
# iterations = 10000
# tmp_alpha = 1.0e-2
# # run gradient descent
# w_final, b_final, J_hist, p_hist = gradient_descent(norm_kms ,norm_prices, w_init, b_init, tmp_alpha, 
#                                                     iterations, compute_cost, compute_gradient)
# print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# # plot cost versus iteration  
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
# ax1.plot(J_hist[:100])
# ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
# ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
# ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
# ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
# plt.show()

# print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
# print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
# print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

# # initialize parameters
# w_init = 0
# b_init = 0
# # set alpha to a large value
# iterations = 10
# tmp_alpha = 8.0e-1
# # run gradient descent
# w_final, b_final, J_hist, p_hist = gradient_descent(norm_kms ,norm_prices, w_init, b_init, tmp_alpha, 
#                                                     iterations, compute_cost, compute_gradient)