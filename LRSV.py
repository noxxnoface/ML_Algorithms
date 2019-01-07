import numpy as np

x = np.arange(9).reshape(9,1)
y =  7 * x + 9
print(x.shape)
print(y.shape)
X = np.append(np.ones((9,1)),x,axis=1)
print(X.shape)
theta = np.array([0.0,0.0]).reshape(2,1)
print(theta.shape)
hypothesis = np.matmul(X,theta)
print(hypothesis.shape)
alpha = 0.09                                     # Optimal Learning Rate
j = np.sum( np.square(hypothesis - y)) / (2 * 9) # Cost Function
print(j)
for i in range(9000):                            # Gradient Descent
    hypothesis = np.matmul(X,theta)
    theta[0] = theta[0] - alpha * np.sum(hypothesis - y) / 9
    theta[1] = theta[1] - alpha * np.sum((hypothesis - y) * x) / 9
j = np.sum( np.square(hypothesis - y)) / (2 * 9)
print(j)
print(theta)
print(hypothesis.astype(int))m
print(y)
def predict(n):                                  # Prediction
    n = np.array([1, n]).reshape(1,2)
    print('\n')
    print(n)
    print(theta)
    print('\n')
    hypothesis = np.matmul(n,theta)
    print(hypothesis)
predict(6)
