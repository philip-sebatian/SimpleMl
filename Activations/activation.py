import numpy as np

from Layers.layer import layer

class Activation(layer):
    def __init__(self):
        self.func=None
        self.func_der=None
    
    def forward(self,input):
        self.input=input
        return self.func(input)
    
    def backwards(self,out_grads,lr):
        return np.multiply(out_grads,self.func_der(self.input))
        