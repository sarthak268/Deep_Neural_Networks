import numpy as np

step_size = 0.01
T = 10
delta_t = 1 / T

def sigmoid(x):
	return (exp(x) / (1 + exp(x)))

def g(y):#forward
	out = sigmoid(np.matmul(W, y) + b)
	return out

def find_gradient_energy():

def geodesic_path(z0, zt):
