# ENAS-Tensoflow
Efficient Neural Architecture search via parameter sharing(ENAS) micro search Tensorflow code for windows user

Now I work very hard ㅠㅠ

```python
import numpy as np

def dis(a,b):
    dim = len(a)
    holder = []
    for i in range(dim):
        square = (a[i] -b[i])**2
        holder.append(square)
    return np.sum(holder)

def center(data,idx): # you should enter list!
    k_mean_center = []
    for j in (idx):
        k_mean_center.append(data[j])
    return k_mean_center

def make_distance_matrix(data, centers):
    result = []
    temp = []
    for a in centers: # 3
        for b in data: # 7
            temp.append(dis(a,b))
        result.append(temp)
        temp = []
    return result

def make_A(distance_matrix):
    compare = []
    A = np.zeros_like(distance_matrix, dtype = np.float32)
    for c in range(len(distance_matrix[1])):
        for d in range(len(distance_matrix)):
            compare.append(distance_matrix[d][c])
        min_val = min(compare)
        idx = compare.index(min_val)
        A[idx][c] = 1.0
        compare = []
    return A

def make_mean(A,data):
    A.astype(int)
    num_cluster = len(A)
    holder = []
    mean = []
    for e in range(num_cluster):
        for f in range(len(data)):
            if A[e][f] == 1:
                holder += [data[f]]
        result = np.mean(holder,axis=0)
        holder = []
        mean.append(result)
    mean = np.reshape(mean,[-1,len(data[0])])
    return mean

def make_J(data,A,means):
    num_cluster = len(means)
    num_data = len(data)
    distance = []
    for g in range(num_cluster):
        for h in range(num_data):
            if A[g][h] == 1.0:
                tanos = np.ones_like(data[h])*data[h]
                distance.append(dis(means[g],tanos))
    J = np.sum(distance)
    return J

def k_mean_cluster(data,num_cluster, idx):
    Centers = center(data,idx)
    distance_matrix = make_distance_matrix(data,Centers)
    A = make_A(distance_matrix)
    means = make_mean(A,data)
    distance_matrix = make_distance_matrix(data,means)
    A = make_A(distance_matrix)
    means = make_mean(A,data)
    J = make_J(data,A,means)
    print("~~~~~~~~~~idx = ",np.add(idx,1),"~~~~~~~~~~~~~")
    print("_______________A_MATRIX_______________")
    print(A)
    print("_______________J_VALUE________________")
    print(J)
    print("\n\n")

if __name__ =="__main__":
    data = [[18.0, 5.0], [20.0, 9.0], [20.0, 14.0], [20.0, 17.0], [5.0, 15.0], [9.0, 15.0], [6., 20.0]]
    k = 3
    k_mean_cluster(data,k,[0,1,2])
    k_mean_cluster(data,k,[1,2,6])
```
