import numpy as np

x = np.arange(9).reshape(9,1)
y = 2 * x**2 - 5 * (x ** 3) + 5
X = np.append(np.ones((9,1)),x,axis=1)
X = np.append(X,x**2,1)
X = np.append(X,x**3,1)
print(X.shape)
print(y.shape)
theta = np.array([0.0,0.0,0.0,0.0]).reshape(4,1)
print(theta.shape)
hypothesis = np.matmul(X,theta)
print(hypothesis.shape)
alpha = 0.09
j = np.sum( np.square(hypothesis - y)) / (4 * 9)
print(j)
for i in range(10000):
    hypothesis = np.matmul(X,theta)
    delta =  np.linalg.pinv(np.matmul(np.linalg.pinv(hypothesis - y),X) / 9)
    theta = theta - alpha * delta
j = np.sum( np.square(hypothesis - y)) / (4 * 9)
print(j)
print(theta)
print(hypothesis.astype(int))
print(y)
def predict(n):                                  # Prediction
    n = np.array([1, n, n**2, n**3]).reshape(1,4)
    print('\n')
    print(n)
    print(theta)
    print('\n')
    hypothesis = np.matmul(n,theta)
    print(hypothesis)
predict(8)
