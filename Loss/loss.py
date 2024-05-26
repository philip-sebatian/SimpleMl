import numpy as np

class loss:
    def __init__(self):
        self.loss_func=None
        self.loss_func_derivative=None
    def get_loss(self,actual,pred):
        return self.loss_func(actual,pred)

    def get_derivative(self,input):
        return self.loss_func_derivative(input)