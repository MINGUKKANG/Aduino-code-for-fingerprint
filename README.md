# ENAS-Tensoflow
Efficient Neural Architecture search via parameter sharing(ENAS) micro search Tensorflow code for windows user

Now I work very hard ㅠㅠ

```python
import tensorflow as tf
import numpy as np

R = np.array([[1,1,1],[2,1,3],[0,1,0]], dtype = np.float32)
G = np.array([[2,2,2],[1,0,1],[0,0,1]], dtype = np.float32)
B = np.array([[0,3,0],[1,0,1],[1,0,0]], dtype = np.float32)

inputs = [R,G,B] # CHW (3,3,3)
inputs = np.transpose(inputs, axes = (1,2,0))
inputs = np.reshape(inputs,[1,3,3,3])

W_R = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype = np.float32)
W_G = np.array([[0,2,0],[0,2,0],[0,2,0]], dtype = np.float32)
W_B = np.array([[1,0,0],[0,2,0],[0,0,1]], dtype = np.float32)
w = [W_R,W_G,W_B]
w = np.transpose(w,axes = (1,2,0))
w = np.reshape(w,[3,3,3,1])

net = tf.nn.conv2d(inputs,w, strides = [1,1,1,1], padding ="SAME")

net = tf.reshape(net,[3,3])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

result = sess.run(net)
print(result)
```
