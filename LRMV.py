import numpy as np
import matplotlib.pyplot as plt

x = np.arange(4).reshape(4,1)
y = x + 3
m = x.shape[0]
X = np.append(np.ones((m,1)),x,axis=1)
X = np.append(X,x**2,1)
X = np.append(X,x**3,1)
n = X.shape[1];
alpha = .0082
theta = np.zeros((n,1))
#print("X",X.shape)
#print("y",y.shape)
#print("theta",theta.shape)

hypothesis = np.matmul(X,theta)
#print("hypothesis",hypothesis.shape)

j = np.sum( np.square(hypothesis - y)) / (2 * m)
visual_j = [j]

print("cost",j)
for i in range(10000):
    hypothesis = np.matmul(X,theta)
    theta = theta - (alpha * (np.matmul(X.T, hypothesis - y)))/m
    #delta =  np.linalg.pinv(np.matmul(np.linalg.pinv(hypothesis - y),X) / m)   #works ten times faster with alpha of .09
    #theta = theta - alpha * delta
    visual_j.append(np.sum( np.square(hypothesis - y)) / (2 * m))
j = np.sum( np.square(hypothesis - y)) / (2 * m)
print("cost",j)
#print("theta",theta)

def predict(tmp):
    tmp = np.array([1, tmp, tmp**2, tmp**3]).reshape(1,n)
    Prediction = np.matmul(tmp,theta)
    print("Prediction",Prediction)
predict(1)

def visual_cost():
    plt.plot(visual_j)
    plt.xlabel('iteration')
    plt.ylabel('Cost')
    plt.show()
visual_cost()
