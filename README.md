# Logistic-Regression-using-ANN
#Implementation of Logistic Regression using Neural Network (from scratch)
---------------------------------------------------------------------------------------------------------------------------
Details:

The logistic regression is a classification model used for binary classification. Here, it is done using 2-layer ANN.
To do so, 1st hidden layer has been filled with 6-neurons and 2nd layer(output layer) has only 1-neuron. As focus is to
do binary classification, logistic (sigmoid) function has been given special place as an activation function. Moreover we 
have to classify as {0,1} also acts as a satisfactory reason for using sigmoid function.
In example, we have assumed 5 training examples and 4 input features.

----------------------------------------------------------------------------------------------------------------------------
Limitations to be taken care of:

Never set the values of weights as zero or very large as doing so would make the slope of linear function(Z). As a result 
output of Activation function would become even more extreme(either 0 or 1) i.e. slope at the discussed points over sigmoid
would tend to become zero. Hence, the updates in the gradient descent update rule will be very small and the optimization 
will be very slow.

It is said that it is more better to initialize W as the points lying over a Gaussian Distribution.

------------------------------------------------------------------------------------------------------------------------------
Observations:

1.In the given code, feel free to increase the learning rate, it would turn out to make accuracy worse, which signifies that
the steps taken are very aggressive which can even diverge (sometimes) the cost function.

2.Value of b can be set as zero as it is just a bias term (an intercept) which would not change the predictions.

3.As we decrease the no. of iterations, the accuracy would tend to become smaller which is a declaration that the assumed 
no. of iterations are not sufficient enough to converge it. In other words, there is a scope of even more convergence.

------------------------------------------------------------------------------------------------------------------------------
