import numpy as np


class layer:
    def __init__(self,feature_size,neurons):
        self.input=None 
        self.output=None
        self.weigths=None
        self.bias=None
    def forward(self,input):
        pass
    def backwards(self,output_grad,lr):
        pass