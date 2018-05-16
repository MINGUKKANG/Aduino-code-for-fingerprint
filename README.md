# ENAS-Tensoflow
Efficient Neural Architecture search via parameter sharing(ENAS) micro search Tensorflow code for windows user

Now I work very hard ㅠㅠ

```python
import numpy as np
import math

W1 = [[1.0,1.2], [-1.0,-1.1]]
W2 = [[1.0,-1.0],[0.5,1.0]]
W3 = [[-0.8,1.0],[0.3,0.4]]
W4 = [[0.1,-0.2],[1.3,-0.4]]

b1 = [[-0.3],[1.6]]
b2 = [[1.0],[0.7]]
b3 = [[0.5],[-0.1]]
b4 = [[1.0],[-0.2]]

X = [[1.0],[0.0]]

def mat(a,b,bias):
    return np.matmul(a,b) + bias

def relu(x):
    holder = []
    length = len(x)
    for j in range(length):
        if x[j]>=0:
            holder.append(x[j])
        else:
            holder.append(0)
    holder = np.reshape(holder,[-1,1])
    return holder

def sigmoid(x):
    holder = []
    length = len(x)
    for i in range(length):
        z = math.exp(-x[i])
        z = 1/(1+z)
        holder.append(z)
    holder = np.reshape(holder,[-1,1])
    return holder

def activation(inputs, activation_fn = "sigmoid"):
    if activation_fn is "sigmoid":
        inputs = sigmoid(inputs)

    else:
        inputs = relu(inputs)
    return inputs

net1 = activation(mat(W1,X,b1), activation_fn ="sigmoid")
net2 = activation(mat(W2,net1,b2),activation_fn ="sigmoid")
net3 = activation(mat(W3,net2,b3),activation_fn ="sigmoid")
net4 = activation(mat(W4,net3,b4),activation_fn ="sigmoid")

print(net4)
```
