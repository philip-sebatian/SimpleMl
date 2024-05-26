import numpy as np
from layer import layer

class Dense(layer):
    def __init__(self,num_input_features,number_nuerons):
        self.weigths=np.random.rand(number_nuerons,num_input_features)
        self.bias=np.random.rand(number_nuerons,1)
    
    def forward(self,input):
        self.input=input
        self.output=np.dot(self.weigths,input)+self.bias
        return self.output

    def backward(self,out_grad,lr):
        grad_w= np.dot(out_grad,self.input.T)
        grad_b= out_grad
        grad_x= np.dot(self.weigths.T,out_grad) #derivative w.r.t to the inputs to the layer so that we can propagate this backwards
        self.weigths-=grad_w*lr
        self.bias-grad_b*lr
        return grad_x

class Conv(layer):
    pass


class Rnn(layer):
    pass 