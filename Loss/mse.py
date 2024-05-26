from Loss.loss import loss
import numpy as np

class MSE(loss):
    def __init__(self):
        super().__init__()

        # Define the loss function and its derivative
        self.loss_func = lambda y_true, y_pred: self.mse(y_true, y_pred)
        self.loss_func_derivative = lambda y_true, y_pred: self.mse_derivative(y_true, y_pred)

    def mse(self, true, pred):
        """
        Computes the mean squared error loss between true and predicted values.
        """
        return np.mean(np.square(true - pred))

    def mse_derivative(self, true, pred):
        """
        Computes the derivative of the mean squared error loss function with respect to the predicted values.
        """
        # The derivative of MSE w.r.t. predicted values is (2 / N) * (pred - true)
        # where N is the total number of elements in the true or predicted array
        return 2 * (pred - true) / np.size(true)
    
    