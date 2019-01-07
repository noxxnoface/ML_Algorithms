import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 20, 90, 50, 32, 22, 9, 89, 99, 78]).reshape(10, 1)
y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1]).reshape(
    10, 1
)  # be aware of shapes of inputs
m = x.shape[0]
X = np.append(np.ones((m, 1)), x, axis=1)
n = X.shape[1]
alpha = 0.1
theta = np.zeros((n, 1))

hypothesis = 1 / (1 + np.exp(-(np.matmul(X, theta))))

j = -1 * (1 / m) * (y.T.dot(np.log(hypothesis)) + (1 - y).T.dot(np.log(1 - hypothesis)))
visual_j = [j]

print("cost", j)
for i in range(2000):
    hypothesis = 1 / (1 + np.exp(-(np.matmul(X, theta))))
    theta = theta - (alpha / m) * X.T.dot(hypothesis - y)
j = -1 * (1 / m) * (np.log(hypothesis).T.dot(y) + np.log(1 - hypothesis).T.dot(1 - y))
print("cost", j)


def predict(no):
    no = np.array([1, no]).reshape(1, n)
    Prediction = 1 / (1 + np.exp(-(np.matmul(no, theta))))
    print("Prediction", Prediction)


predict(20)

plt.scatter(
    x, y, c=y[:] == 0
)  # color takes array which is fucking useful for visualizing classifications
plt.show()
