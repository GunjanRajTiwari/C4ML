import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above 
    '''
    m = y.size
    h = sigmoid(np.dot(X,theta))

    J = np.sum((-1/m)*((y.T)*np.log(h)+((1-y).T)*np.log(1-h)))

    if np.isnan(J):
        return(np.inf)
    return(J)

def gradient(theta, X, y):
    m = y.size
    theta = theta.reshape(-1,1)
    h = sigmoid(np.dot(X,theta))

    grad = (1/m)*np.dot(X.T,h-y)

    return grad.flatten()

