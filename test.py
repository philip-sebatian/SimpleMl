from Activations.activation_functions import Softmax
import numpy as np


ss=Softmax()
x=np.random.rand(10)
y=ss.forward(x)
print(y)
print(np.sum(y))
