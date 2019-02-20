#this is implementation of Logistic Regression for Binary Classification using 2-layer Neural Network
import numpy as np
from scipy.stats import logistic
from sklearn import preprocessing

def sigm(z):
    return logistic.cdf(z)

def get_Xy():
    X = np.array([[66, 11, 65, 50, 80],
                  [22, 22, 99, 33, 20],
                  [83, 70, 70, 12, 40],
                  [70, 35, 33, 50, 30]])  # 4X5

    X_scaled = preprocessing.scale(X)

    y = np.array([[0, 0, 0, 1, 1]])       # 1X5

    return X_scaled,y

def test_data():
    X = np.array([[60, 13, 75, 55, 80],
                  [21, 26, 90, 43, 30],
                  [70, 75, 110, 22, 30],
                  [70, 34, 38, 55, 40]])

    X_scaled = preprocessing.scale(X)
    y = np.array([[0, 0, 1, 1, 1]])

    return X_scaled,y

def init_para(no_units, no_features):       #1st layer has 6 units, 2nd layer/output layer has only 1 unit

    W1=np.random.randn(no_units, no_features) * 0.01
    b1=np.zeros(shape=(no_units,1))
    W2=np.random.randn(1,6) * 0.01        #W2=1x6
    b2=np.zeros(shape=(1,1))              #b2=scalar

    return W1,b1,W2,b2



def train(iterations,confidence):
    no_units=6
    no_features=4
    W1,b1,W2,b2=init_para(no_units,no_features)
    X,y=get_Xy()                                    # X--->train_X ,  y--->test_y
    m=5              #training examples
    for i in range(iterations):
        Z1 = np.dot(W1, X) + b1  # foreprop begins...
        A1 = logistic.cdf(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = logistic.cdf(Z2)  # foreprop ends...

        logprobs = np.multiply(np.log(A2), y) + np.multiply((1 - y), np.log(1 - A2))
        cost = -np.sum(logprobs) / float(m)  # cost function...

        # dA2 = -(y/A2)+((1-y)/(1-A2))         #back prop begins...
        dZ2 = A2 - y
        dW2 = (np.dot(dZ2, A1.T)) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        # dA1 = dZ2*(W2)
        dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
        dW1 = (np.dot(dZ1, X.T)) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True)  # backprop ends...
        print cost
        W1 = W1 - confidence * dW1  # update begins...
        b1 = b1 - confidence * db1
        W2 = W2 - confidence * dW2
        b2 = b2 - confidence * db2  # update ends...

    return W1,b1,W2,b2


def predict():
    iterations=20000
    confidence=0.001
    W1,b1,W2,b2=train(iterations,confidence)
    X,y=test_data()                                 # X--->test_X  ,  y--->test_y
    Z1=np.dot(W1,X)+b1
    A1 = sigm(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigm(Z2)


    A2=np.squeeze(A2)
    y=np.squeeze(y)
    corr=0

    for j in range(len(y)):
        A2[j]=1 if A2[j]>0.5 else 0
        y[j]=1  if y[j]>0.5 else 0


    for i in range(len(y)):
        if (A2[i]-y[i]==0):
            corr += 1


    accuracy=(corr*100.0)/len(y)
    return accuracy


accu=predict()
print accu













