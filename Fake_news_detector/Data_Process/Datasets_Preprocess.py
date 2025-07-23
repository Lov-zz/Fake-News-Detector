import pandas as pd

def Data_Check(T_file_pth, F_file_pth):
    '''
    function:
        open the file of datasets(including the true news and fake news)
        check the details of the datasets : num_rows, num_cols, num_empty_val
    parameters:
        T_file_pth : str : the path of the true news dataset
        F_file_pth : str : the path of the fake news dataset
    type : csv
    '''
    num_rows = 0        # number of rows
    num_cols = 0        # number of columns
    num_empty_val = 0   # number of empty values    

    # Check True News Dataset
    try:
        T_data = pd.read_csv(T_file_pth)
        num_rows, num_cols = T_data.shape
        num_empty_val = T_data.isnull().sum().sum()
        print(f"True file row:{num_rows}, col: {num_cols}, num_empty_val: {num_empty_val}")
    except Exception as e:
        print(f"Error reading True News dataset: {e}")
        return

    # Check Fake News Dataset
    try:
        F_data = pd.read_csv(F_file_pth)
        num_rows, num_cols = F_data.shape
        num_empty_val = F_data.isnull().sum().sum()
    except Exception as e:
        print(f"Error reading Fake News dataset: {e}")
        return
    print(f"Fake file row: {num_rows}, col: {num_cols}, num_empty_val: {num_empty_val}")

def Data_Merge(T_file_pth, F_file_pth, Save_file_pth):
    '''
    function:
        add the label to the datasets('1' for true news, '0' for fake news)
        merge the two datasets into one
        disrupt the order
    parameters:
        T_file_pth : str : the path of the true news dataset
        F_file_pth : str : the path of the fake news dataset
        Save_file_pth : str : the path to save the merged dataset
    '''
    # add the label to the true news dataset
    T_data = pd.read_csv(T_file_pth)
    T_data['label'] = 1

    # add the label to the fake news dataset
    F_data = pd.read_csv(F_file_pth)
    F_data['label'] = 0

    # merge the two datasets and disrupt the order
    merged_data = pd.concat([T_data, F_data], ignore_index=True)
    merged_data = merged_data.sample(frac = 1).reset_index(drop = True)
    print(f"Merged completed! Total rows: {merged_data.shape[0]}, Total cols: {merged_data.shape[1]}")
    merged_data.to_csv(Save_file_pth, index = False)
    print(f'Merged data has been saved in {Save_file_pth}')


def main():
    '''paraemters:
        T_file_pth : str : the path of the true news dataset
        F_file_pth : str : the path of the fake news dataset
        Save_file_pth : str : the path to save the merged dataset
    '''
    T_file_pth = 'dataset/True.csv' # please change the path to your true news dataset
    F_file_pth = 'dataset/Fake.csv' # please change the path to your fake news dataset
    Save_file_pth = 'Fake_news_detector/News_dataset/Merged.csv'

    Data_Check(T_file_pth, F_file_pth)
    Data_Merge(T_file_pth, F_file_pth, Save_file_pth)

    print("Data check has been completed!")

if __name__ == "__main__":
    main()