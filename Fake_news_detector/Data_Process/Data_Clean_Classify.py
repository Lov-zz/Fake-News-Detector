import re
import pandas as pd
from sklearn.model_selection import train_test_split

def Data_clean(merged_file_pth, Save_file_pth):
    '''
    function:
        clean the merged dataset
        remove the HTML tags, URLs, Non-letter characters, and extra spaces
        convert to lowercase
        check and fill empty values in the 'text' and 'clean_text' columns
    paraemeters:
        merged_file_pth : str : the path of the merged dataset
        Save_file_pth : str : the path to save the cleaned dataset
    '''
    try:
        data = pd.read_csv(merged_file_pth)
    except Exception as e:
        print(f"Error reading merged dataset: {e}")
        return
    
    # check if the 'text' column exists
    print(f'empty values in the text column: {data["text"].isnull().sum()}')
    if data["text"].isnull().sum() > 0:
        data['text'] = data['text'].fillna('')
    
    # Clean the 'text' column
    data['clean_text'] = data['text'].apply(Clean_text)

    # check if the 'clean_text' column exists
    print(f'empty values in the clean_text column: {data["clean_text"].isnull().sum()}')
    if data["clean_text"].isnull().sum() > 0:
        data['clean_text'] = data['clean_text'].fillna('')
        print(f'{data["clean_text"].isnull().sum()} empty values in the clean_text column have been filled with empty strings')

    # Save the cleaned dataset
    data.to_csv(Save_file_pth, index=False)
    print(f'Cleaned data has been saved in {Save_file_pth}')

def Clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove non-letter characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    return text.lower()

def Data_Classify(Save_file_pth, train_file_pth, test_file_pth):
    '''
    function:
        classify the cleaned dataset into training and testing datasets
        save the training and testing datasets
    paraemeters:
        Save_file_pth : str : the path of the cleaned dataset
        train_file_pth : str : the path to save the training dataset
        test_file_pth : str : the path to save the testing dataset    
    '''
    data = pd.read_csv(Save_file_pth)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.to_csv(train_file_pth, index = False)
    print(f'The training dataset has {train_data.shape[0]}')
    print(f'Training data has been saved in {train_file_pth}')
    test_data.to_csv(test_file_pth, index = False)
    print(f'The testing dataset has {test_data.shape[0]}')
    print(f'Testing data has been saved in {test_file_pth}')

def main():
    '''parameters:
        merged_file_pth : str : the path of the merged dataset
        Save_file_pth : str : the path to save the cleaned dataset
    '''
    merged_file_pth = 'Fake_news_detector/News_dataset/Merged.csv'
    Save_file_pth = 'Fake_news_detector/News_dataset/Cleaned_Merged.csv'
    train_file_pth = 'Fake_news_detector/News_dataset/Train.csv'
    test_file_pth = 'Fake_news_detector/News_dataset/Test.csv'

    Data_clean(merged_file_pth, Save_file_pth)
    Data_Classify(Save_file_pth, train_file_pth, test_file_pth)

if __name__ == "__main__":
    main()