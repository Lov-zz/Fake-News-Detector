import pandas as pd
import joblib as jb
from sklearn.feature_extraction.text import TfidfVectorizer

# import the train and test csv datasets
Train_file_pth = 'Fake_news_detector/News_dataset/Train.csv'
Test_file_pth = 'Fake_news_detector/News_dataset/Test.csv'

# import the save path of vectorizered data
X_train_pth = 'Fake_news_detector/News_dataset/X_train.pkl'
X_test_pth = 'Fake_news_detector/News_dataset/X_test.pkl'

# import the save path of label data
Y_train_pth = 'Fake_news_detector/News_dataset/Y_train.pkl'
Y_test_pth = 'Fake_news_detector/News_dataset/Y_test.pkl'

# import the save path of vectorizer
vectorizer_pth = 'Fake_news_detector/News_dataset/tfidf_vectorizer.pkl'

# check if there are 'label' and 'clean_text' columns in the datasets
Train_data =  pd.read_csv(Train_file_pth)
print(Train_data[['label', 'clean_text']].head())
Test_data = pd.read_csv(Test_file_pth)
print(Test_data[['label', 'clean_text']].head())

# empty value checking and processing
# replace empty values in 'clean_text' column with empty strings
if Train_data['clean_text'].isnull().sum() > 0:
    print(f'empty values in the train data clean text column: {Train_data['clean_text'].isnull().sum()}')
    print('fixing ... ')
    Train_data['clean_text'] = Train_data['clean_text'].fillna('')
    print(f'{Train_data["clean_text"].isnull().sum()} empty values in the train data clean text column have been filled with empty strings')
else:
    print('no empty values in the train data clean text column')

if Test_data['clean_text'].isnull().sum() > 0:
    print(f'empty values in the test data clean text column: {Test_data['clean_text'].isnull().sum()}')
    print('fixing ... ')
    Test_data['clean_text'] = Test_data['clean_text'].fillna('')
    print(f'{Test_data["clean_text"].isnull().sum()} empty values in the test data clean text column have been filled with empty strings')
else:
    print('no empty values in the test data clean text column')
# check all the values in 'clean_text' are strings
Train_data['clean_text'] = Train_data['clean_text'].astype(str)
Test_data['clean_text'] = Test_data['clean_text'].astype(str)
# replace the empty strings with ' '
Train_data['clean_text'] = Train_data['clean_text'].replace('',' ')
Test_data['clean_text'] = Test_data['clean_text'].replace('', ' ')

# turn the 'clean_text' into vector
vectorizer = TfidfVectorizer(max_features = 5000)
X_train = vectorizer.fit_transform(Train_data['clean_text'])
print(f'Vectorized training data shape: {X_train.shape}')
X_test = vectorizer.transform(Test_data['clean_text'])
print(f'Vectorized testing data shape: {X_test.shape}')

# save the vectorizer
jb.dump(vectorizer, vectorizer_pth)
print(f'TF-IDF vector quantizer has been saved in {vectorizer_pth}')

# save the vectorized data
jb.dump(X_train, X_train_pth)
print(f'X_train has been saved in {X_train_pth}')
jb.dump(X_test, X_test_pth)
print(f'X_test has been saved in {X_test_pth}')

# save the label
Y_train = Train_data['label']
Y_test = Test_data['label']
jb.dump(Y_train, Y_train_pth)
print(f'Y_train has been saved in {Y_train_pth}')
jb.dump(Y_test, Y_test_pth)
print(f'Y_test has been saved in {Y_test_pth}')

print(f"Text vectorization has been completed!")