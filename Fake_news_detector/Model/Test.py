import torch
import joblib as jb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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

def predict(X, weights, bias):
    '''
    function:
        use the trained model(weights and bias) to predict the labels of the input data
    parameters:
        X: input data, shape (m, n)
        weights: trained weights, shape (n, 1)
        bias: trained bias, scalar    
    '''
    return sigmoid(linear_forward(X, weights, bias))

def test(X_test, y_test, weights,bias):
    '''
    function:
        calculate the predicted labels of the test dataset
        calculate the accuracy of the predicted labels
    parameters:
        X_test: test input data, shape (m, n)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # turn data into tensor
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    weights = weights.to(device)
    bias = bias.to(device)

    # compute the predicted labels
    pre_y = predict(X_test, weights, bias)

    # binary classification(pre_y > 0.5 -> pre_labels = 1, else pre_labels = 0)
    pre_labels = (pre_y > 0.5).float()

    # calculate the accuracy
    pre_labels = pre_labels.view(-1)
    y_test = y_test.view(-1)
    correct = (pre_labels == y_test).sum().item()
    accuracy = (correct / y_test.shape[0]) * 100
    print(f'Test accuracy: {accuracy:.2f}%')

    # other indicator
    y_true = y_test.cpu().numpy()
    y_pred = pre_labels.cpu().numpy()

    # Calculate accuracy: the ratio of correctly predicted samples to the total samples.
    # It gives an overall measure of how often the classifier is correct.
    acc = accuracy_score(y_true, y_pred)

    # Calculate precision: the ratio of correctly predicted positive samples to all predicted positives.
    # It measures how many of the predicted positive cases are actually positive.
    # Useful when the cost of false positives is high.
    prec = precision_score(y_true, y_pred)

    # Calculate recall (also called sensitivity or true positive rate): 
    # the ratio of correctly predicted positive samples to all actual positives.
    # It shows how well the model detects positive cases.
    # Important when missing positive cases is costly.
    rec = recall_score(y_true, y_pred)

    # Calculate F1 score: the harmonic mean of precision and recall.
    # Provides a single score that balances both false positives and false negatives.
    # Useful when you need a balance between precision and recall.
    f1 = f1_score(y_true, y_pred)

    # Print all metrics
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def main():
    # import the test dataset
    X_test_pth = 'Fake_news_detector/News_dataset/X_test.pkl'
    y_test_pth = 'Fake_news_detector/News_dataset/Y_test.pkl'

    # import the trained model
    model_path = 'Fake_news_detector/Model/model.pkl'

    X_test = jb.load(X_test_pth).toarray()
    y_test = jb.load(y_test_pth)
    model = torch.load(model_path)
    w, b = model['w'], model['b']

    test(X_test, y_test, w, b)

if __name__ == '__main__':
    main()