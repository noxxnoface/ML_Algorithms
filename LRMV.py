import numpy as np
import matplotlib.pyplot as plt

x = np.arange(4).reshape(4,1)
y = x + 3 + 5 * x ** 2
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

#print("cost",j)
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
    print("Prediction",Prediction.squeeze())
predict(20)

#Second Method
"""
def visual_cost():
    plt.plot(visual_j)
    plt.xlabel('iteration')
    plt.ylabel('Cost')
    plt.show()
visual_cost()

x = np.arange(9).reshape(9,1)
y = 2 * x**2 - 5 * (x ** 3) + 5
X = np.append(np.ones((9,1)),x,axis=1)
X = np.append(X,x**2,1)
X = np.append(X,x**3,1)
#print(X.shape)
#print(y.shape)
theta = np.array([0.0,0.0,0.0,0.0]).reshape(4,1)
#print(theta.shape)
hypothesis = np.matmul(X,theta)
#print(hypothesis.shape)
alpha = 0.09
j = np.sum( np.square(hypothesis - y)) / (4 * 9)
#print(j)
for i in range(10000):
    hypothesis = np.matmul(X,theta)
    delta =  np.linalg.pinv(np.matmul(np.linalg.pinv(hypothesis - y),X) / 9)
    theta = theta - alpha * delta
j = np.sum( np.square(hypothesis - y)) / (4 * 9)
print("cost", j)
#print(theta)
#print(hypothesis.astype(int))
#print(y)
def predict(n):                                  # Prediction
    n = np.array([1, n, n**2, n**3]).reshape(1,4)
    #print('\n')
    #print(n)
    #print(theta)
    #print('\n')
    hypothesis = np.matmul(n,theta)
    print(hypothesis)
print("PREDICTION:")
predict(8)
"""
