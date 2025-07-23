import joblib
import torch
import re

# text_clean
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# predict function
def predict(text, vectorizer, weights, bias, device):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    X = torch.tensor(vector.toarray(), dtype=torch.float32).to(device)
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    b = torch.tensor(bias, dtype=torch.float32).to(device)

    z = torch.matmul(X, w) + b
    y_prob = torch.sigmoid(z).item()
    return y_prob

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # import the vectorizer and model
    vectorizer = joblib.load('Fake_news_detector/News_dataset/tfidf_vectorizer.pkl')
    model = torch.load('Fake_news_detector/Model/model.pkl')
    w, b = model['w'], model['b']

    while True:
        text = input("please input the news text（input 'exit' to quit）：")
        if text.lower() == 'exit':
            break
        prob = predict(text, vectorizer, w, b, device)
        print(f"Model predicts the probability of real news: {prob:.4f}")
        if prob < 0.5:
            print("result：fake news🚩")
        else:
            print("result：real news✅")

if __name__ == '__main__':
    main()
