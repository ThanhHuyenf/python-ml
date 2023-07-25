import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
  """ plots logistic data with two axis """
  # Find Indices of Positive and Negative Examples
  pos = y == 1
  neg = y == 0
  pos = pos.reshape(-1,)  #work with 1D or 1D y vectors
  neg = neg.reshape(-1,)

  # Plot examples
  ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
  ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors="blue", lw=3)
  ax.legend(loc=loc)

  ax.figure.canvas.toolbar_visible = False
  ax.figure.canvas.header_visible = False
  ax.figure.canvas.footer_visible = False


fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()

def sigmoid(z):
  z = np.clip( z, -500, 500 )           # protect against overflow
  g = 1.0/(1.0+np.exp(-z))
  return g


def compute_cost_logistic(X, y, w, b):
  m = X.shape[0]
  cost = 0
  for i in range(m):
    z_i = np.dot(w, X[i]) + b
    f_wb_i = sigmoid(z_i)
    print(f"z_{i} = {z_i}, f_wb_{i} = {f_wb_i}")
    cost += -y[i]*np.log(f_wb_i)- (1-y[i])*np.log(1-f_wb_i)
  cost = cost/m
  return cost

compute_cost_logistic(X_train, y_train, np.array([1,1]), -3)