import numpy as np

from Layers.layer import layer

class Activation(layer):
    def __init__(self,activation,activation_der):
        self.func=activation
        self.func_der=activation_der
    
    def forward(self,input):
        self.input=input
        return self.func(input)
    
    def backwards(self,out_grads,lr):
       # print(out_grads)
        #print(self.func_der(self.input))
        return np.multiply(out_grads,self.func_der(self.input))
        