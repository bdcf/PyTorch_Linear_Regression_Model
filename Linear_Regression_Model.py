import torch
from torch import nn # contains neural network building blocks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Data can be anything as long as it can be turned into a numerical representation. We will use 
# linear regression to create some data on a straight line firstly we create known paratemeters
weight = 0.7
bias = 0.3
start = 0
end  = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias # y = mx + b
# We must split the data into training and test sets
split = int(0.8 * len(x))
xTrain, yTrain, xTest, yTest = x[:split], y[:split], x[split:], y[split:]
# Now we have to build a function to better visualize the data
def prediction_algo(train_data = xTrain, train_labels = yTrain, test_data = xTest, test_labels = yTest, predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Prediction Data")
    plt.legend(prop={"size":14})
    plt.show()
# almost everything in pytorch comes from nn.module so it is needed for the regression model
class regressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # define computation model where tense is the input data as forward is always used for 
        # the calculations we must always override it with our own. this is only when using 
        # nn.Module.
    def forward(self, tense: torch.Tensor) -> torch.Tensor:
        # Do the Linear regression
        return self.weights * tense + self.bias
# This starts with random values then it looks at the training data we made with torch.arange 
# to better represent the ideal values for weight and bias it does this through gradient descent
# or backpropogation. Using manual seed allows you to get the same random numbers again and again 
# as you set a seed instead of it selecting one at runtime
torch.manual_seed(42)
# Create an instance of the module
modelZero = regressionModel()
parm = list(modelZero.parameters())
# inference_mode allows the code to work much faster as it doesnt keep track of anything other than
# the raw data(so it excludes things like gradient calulcations in return for speed).
#with torch.inference_mode():
#    predictionY = modelZero(xTest)
#prediction_algo(predictions=predictionY)
# to measure how right or wrong we are we need a loss function also known as a cost function
lossFunc = nn.L1Loss()
# we also need an optimizer function to adjust the weights and bias' we will use random gradient 
# descent
optfunc = torch.optim.SGD(params=modelZero.parameters(), lr = 0.01)
# Now with this we can set up a training loop for the AI using epochs which are loops through.
epoch = 10000
for epoch in range(epoch):
    # train mode sets all params that require gradients to require gradients.
    modelZero.train()
    predictionY = modelZero(xTrain)
    loss = lossFunc(predictionY, yTrain)
    print(f"Loss: {loss}")
    optfunc.zero_grad()
    # do backpropogation on the loss with respectt model parameters
    loss.backward()
    # step to perform gradient descent
    optfunc.step()
    # eval turns off gradient tracking
    modelZero.eval()
    # Now we set up the testing loop
    with torch.inference_mode():
        test_pred = modelZero(xTest)
        test_loss = lossFunc(test_pred, yTest)
    if epoch % 10 == 0:
        print(f"epoch: {epoch} | Loss: {loss} | Test Loss {test_loss}")
prediction_algo(predictions=test_pred)


