import matplotlib
import torch
from sklearn.model_selection import train_test_split

import torch.nn as nn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)

        return out


def train_model(model, input_train, output_train, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_corrects = 0.0

    output_model = model(input_train)  # model prediction for the given inputs
    loss = criterion(output_model, output_train)  # the difference between the model predictions and the actual answers
    optimizer.zero_grad()  # clears the gradient for all parameters: x.grad = 0
    loss.backward()  # calculates the gradient dy/dx (actually: x.grad+=dy/dx, as x is a parameter & grad is a property)
    optimizer.step()  # updates the parameter with the calculated gradient: x += -lr * x.grad

    max_scores, pred = torch.max(output_model, dim=1)  # max_scores: the max between the 2 outputs, pred is the class

    running_loss += loss.item() * input_train.size(0)
    running_corrects += torch.sum(pred == output_train.data)

    epoch_loss = running_loss / len(input_train)
    epoch_acc = running_corrects / len(input_train)

    return epoch_acc, epoch_loss


def validate_model(model, input_test, output_test, criterion):
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    output_model = model(input_test)
    loss = criterion(output_model, output_test)

    _, pred = torch.max(output_model, dim=1)

    running_loss += loss.item() * input_test.size(0)
    running_corrects += torch.sum(pred == output_test.data)

    epoch_loss = running_loss / len(input_test)
    epoch_acc = running_corrects / len(input_test)

    return epoch_acc, epoch_loss


data = pd.read_csv("datasets/diabetes.csv")
x = data.iloc[:, 0: -1].values
y = data.iloc[:, -1].values
y = np.array(y, dtype='float64')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

net = Model()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

number_of_epochs = 1023

final_losses = []
final_accuracies = []
pred = []

test_losses = []
test_accuracies = []

for i in range(number_of_epochs):
    i += 1

    epoch_acc, epoch_loss = train_model(net, x_train, y_train, loss_function, optimizer)

    final_losses.append(epoch_loss)
    final_accuracies.append(epoch_acc)

    test_acc, test_loss = validate_model(net, x_test, y_test, loss_function)

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    if i % 10 == 1:
        print("epoch number {}: loss is {loss:.5f}% & accuracy is {acc:.5f}%"
              .format(i, loss=epoch_loss * 100, acc=epoch_acc * 100))

plt.plot(range(number_of_epochs), final_losses)
plt.plot(range(number_of_epochs), test_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Train", "Test"])
plt.title("Trains vs Test Loss")
plt.show()
