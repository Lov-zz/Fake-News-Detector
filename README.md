# Fake News Detection with Logistic Regression

## Project Overview
This project implements a logistic regression model to detect fake news. The dataset used contains political and world news articles, categorized as real or fake.

## Features
- Data preprocessing using TF-IDF vectorization
- Logistic regression model implemented with PyTorch
- Training and testing scripts
- Evaluation metrics including accuracy, precision, recall, and F1 score

## Results
The model's loss decreases steadily over epochs, indicating good convergence.

![Loss of each 50 epochs](images/loss_plot.png)

### 🔢 Evaluation Metrics

| Metric       | Value   |
|--------------|---------|
| Accuracy     | 96.29%  |
| Precision    | 95.24%  |
| Recall       | 97.12%  |
| F1 Score     | 96.17%  |

## Requirements
- Python 3.9+
- PyTorch
- scikit-learn
- pandas
- numpy
- joblib
- matplotlib
- Flask (optional, for API)

## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Download the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data) and place it in the News_dataset/ directory.

# Usage
## Training
Run the training script to train the logistic regression model:
```
python train.py
```
## Testing
Run the testing script to evaluate the model on test data:
```
python test.py
```

# License
This project is licensed under the MIT License. See the [LICENSE](Fake_news_detector\LICENSE) file for details.
