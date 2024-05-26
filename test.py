from Activations.activation_functions import *
from Layers.dense import *
from Loss.mse import *


network=[
    Dense(3,2),
    Relu(),
    Dense(1,3),
    Sigmoid(),
]
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
epochs=1000
los = MSE()
lr=0.1
for i in range(epochs):
    error=0
    for x,y in zip(X,Y):
        output=x 
        for layers in network:
            output=layers.forward(output)
        error+=los.get_loss(y,output)
        grad=los.loss_func_derivative(y,output)
        for ll in reversed(network):
            grad=ll.backwards(grad,lr)
            
        
    print(i,"/",error)

for i in X:
    x=i
    for ll in network:
        x=ll.forward(x)
    print(x)

