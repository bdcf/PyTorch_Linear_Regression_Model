import torch
from torch import nn # contains neural network building blocks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Data can be anything as long as it can be turned into a numerical representation. We will use 
# linear regression to create some data on a straight line.
weight = 0.7
bias = 0.3
start = 0
end  = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias # y = mx + b
# We must split the data into training and test sets.
split = int(0.8 * len(x))
xTrain, yTrain, xTest, yTest = x[:split], y[:split], x[split:], y[split:]
# Now we have to build a function to better visualize the data.
def prediction_algo(train_data = xTrain, train_labels = yTrain, test_data = xTest, test_labels = yTest, predictions = None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Prediction Data")
    plt.legend(prop={"size":14})
    plt.show()
# Almost everything in pytorch comes from nn.module so it is needed for the regression model.
class regressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        # Define computation model where tense is the input data as forward is always used for 
        # the calculations in nn.Module we must always override it with our own.
    def forward(self, tense: torch.Tensor) -> torch.Tensor:
        # Do the Linear regression.
        return (self.weights + 0.0101) * tense + self.bias + 0.0205
# Created a new linear regression model using the subclass of nn.Module.
#class regressionModelTwo(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.linearLayer = nn.Linear(in_features=1, out_features=1)
#    def forward(self, tense: torch.Tensor) -> torch.Tensor:
#        return self.linearLayer(tense)
# This starts with random values, then it looks at the training data made with torch.arange 
# to better represent the ideal values for weight and bias. It does this through gradient descent. 
# Using a manual seed allows you to get the same random numbers again and again as you set a seed 
# instead of it selecting one randomly during runtime.
torch.manual_seed(42)
# Create an instance of the module.
modelZero = regressionModel()
parm = list(modelZero.parameters())
# inference_mode allows the code to work much faster as it doesnt keep track of anything other than
# the raw data(so it excludes things like gradient calulcations in return for speed).
#with torch.inference_mode():
#    predictionY = modelZero(xTest)
#prediction_algo(predictions=predictionY)
# To measure how right or wrong we are we need a loss function also known as a cost function
lossFunc = nn.L1Loss()
# We also need an optimizer function to adjust the weights and bias' we will use random gradient 
# descent
optfunc = torch.optim.SGD(params=modelZero.parameters(), lr = 0.01)
# Now with this we can set up a training loop for the AI using epochs which are loops through.
epoch = 10000
# Add arrays to track epoch count, loss values and test loss values
epochCount = []
lossValues = []
testLossValues = []
for epoch in range(epoch):
    # Set model to training mode which enables gradient computation for parameters.
    modelZero.train()
    predictionY = modelZero(xTrain)
    loss = lossFunc(predictionY, yTrain)
    print(f"Loss: {loss}")
    optfunc.zero_grad()
    # Do backpropogation on the loss with respectt model parameters.
    loss.backward()
    # Step to perform gradient descent.
    optfunc.step()
    # Eval turns off gradient tracking.
    modelZero.eval()
    # Now we set up the testing loop.
    with torch.inference_mode():
        test_pred = modelZero(xTest)
        test_loss = lossFunc(test_pred, yTest)
    if epoch % 10 == 0:
        epochCount.append(epoch)
        lossValues.append(loss)
        testLossValues.append(test_loss)
        print(f"epoch: {epoch} | Loss: {loss} | Test Loss {test_loss}")
prediction_algo(predictions=test_pred)
# Create a model directory and save path to save the AI model
modelPath = Path("models")
modelPath.mkdir(parents=True, exist_ok=True)
modelName = "regression_model_one.pth"
modelSavePath = modelPath / modelName
torch.save(obj=modelZero.state_dict(), f=modelSavePath)
# Now to load the model and make sure both are the same
loadModel = regressionModel()
loadModel.load_state_dict(torch.load(f=modelSavePath))
loadModel.eval()
with torch.inference_mode():
    predictionLoad = loadModel(xTest)
# For checking the loaded models accuracy.
#eTensor = torch.eq(test_pred, predictionLoad)
#print(f"{eTensor}")

