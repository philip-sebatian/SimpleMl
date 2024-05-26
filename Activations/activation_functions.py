from Activations.activation import Activation
import numpy as np

class Relu(Activation):
    def __init__(self):
        self.func = lambda x: np.maximum(x, 0)
        self.func_der = lambda x: np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    def __init__(self):
        self.func = lambda x: 1/(1+np.exp(-x))
        self.func_der = lambda x: np.exp(-x)/np.square((1+np.exp(-x)))



class Softmax(Activation):
    def __init__(self):
        self.func = lambda x: self.softmax(x)
        self.func_der = lambda x: self.softmax_derivative(x)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for numerical stability
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def softmax_derivative(self, x):
        s = self.softmax(x).reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    
