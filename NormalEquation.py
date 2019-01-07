import numpy as np

x = np.arange(4).reshape(4,1)
y = x + 3 + 2 * (x ** 2)
m = x.shape[0]
n = 4
X = np.append(np.ones((m,1)),x,axis=1)
X = np.append(X,x**2,1)
X = np.append(X,x**3,1)
#print("X",X.shape)
#print("y",y.shape)

theta = np.matmul(np.linalg.pinv(np.matmul(X.T, X)), np.matmul(X.T, y))
#print(theta)

def predict(tmp):
    tmp = np.array([1, tmp, tmp**2, tmp**3]).reshape(1,n)
    Prediction = np.matmul(tmp,theta)
    print("Prediction",Prediction)
predict(2)
