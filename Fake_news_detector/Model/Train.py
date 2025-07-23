import pandas as pd
import numpy as np
import joblib as jb
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def sigmoid(x):
    '''
    parameters:
        x: output of the linear function
    '''
    return torch.sigmoid(x)

def linear_forward(X, w, b):
    '''
    function:
        Perform linear forward propagation.
    parameters:
        X: input data, shape (m, n)
        w: weights, shape (n, 1)
        b: bias, scalar
    '''
    z = torch.matmul(X, w) + b
    return z

def compute_loss(y, y_hat):
    '''
    function:
        calculate the difference between the predicted value and the true value
    parameters:
        y: true labels, shape (m, 1)
        y_hat: predicted labels, shape (m, 1)
    '''
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))
    return loss

def compute_gradients(X, y, y_hat):
    '''
    function:
        calculate the gradients of the loss function with respect to the weights and bias
    parameters:
        X: input data, shape (m, n)
        y: true labels, shape (m, 1)
        y_hat: predicted labels, shape (m, 1)    
    '''
    dw = torch.matmul(X.T, (y_hat - y)) / y.shape[0]
    db = torch.mean(y_hat - y)
    return dw, db

def Loss_plot(loss, epoch):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch, loss)
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('D:\\Fake_News\\Model\\loss_plot.png')
    plt.show()
    
def save_model(w, b, model_path):
    '''
    functiom:
        Save the model parameters to a file.
    paraemeters:
        w: weights, shape (n, 1)
        b: bias, scalar
        model_path: path to save the model    
    '''
    torch.save({'w': w.cpu(), 'b': b.cpu()}, model_path)
    print(f'Model parameters have been saved to {model_path}')

def predict(X, w, b):
    '''
    function:
        calculate the predicted labels
    parameters:
        X: input data, shape (m, n)
        w: weights, shape (n, 1)
        b: bias, scalar    
    '''
    z = linear_forward(X, w, b)
    y_hat = sigmoid(z)
    return (y_hat > 0.5).float()

def evaluate(X, y, w, b):
    '''
    function:
        calculate the training accuracy rate
    parameters:
        X: input data, shape (m, n)
        y: true labels, shape (m, 1)
        w: weights, shape (n, 1)
        b: bias, scalar
    '''
    y_pred = predict(X, w, b)
    accuracy = torch.mean((y_pred == y).float()).item()
    print(f'Training accuracy: {accuracy * 100:.2f}%')

def train(X, y, model_path, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    '''
    function:
        Train the model using gradient descent.
    parameters:
        X: input data, shape (m, n)
        y: true labels, shape (m, 1)
    '''
    alpha = 0.03  # learning rate
    epochs = 35000  # number of iterations

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    # initialize weights and bias
    n_features = X.shape[1]
    w = torch.zeros((n_features, 1), dtype=torch.float32, requires_grad=True, device=device)
    b = torch.tensor(0.0, requires_grad=True, device=device)

    Loss_list = []

    # gradient descent update
    for epoch in range(epochs):
        Z = linear_forward(X, w, b)
        y_hat = sigmoid(Z)
        L = compute_loss(y, y_hat)
        Loss_list.append(L.item())
        dw, db = compute_gradients(X, y, y_hat)
        w = w - alpha * dw
        b = b - alpha * db

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {L:.4f}')

    Loss_plot(Loss_list, list(range(epochs)))
    save_model(w, b, model_path)
    evaluate(X, y, w, b)

def main():
    # import the processed trainingdata
    x_train_pth = 'Fake_news_detector/News_dataset/X_train.pkl'
    y_train_pth = 'Fake_news_detector/News_dataset/Y_train.pkl'

    # import the model save path
    model_path = 'Fake_news_detector/Model/model.pkl'

    X_train = jb.load(x_train_pth).toarray()
    Y_train = jb.load(y_train_pth)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    train(X_train, Y_train, model_path)

if __name__ == "__main__":
    main()