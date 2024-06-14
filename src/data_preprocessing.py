# data_preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_data(file_path):
    """ Function to clean up the data such that it can be fed into the embedding model"""

    df = pd.read_csv(file_path, delimiter=';')
    
    # drop the id and quantity columns as they don't provide any meaningful semantic information
    df.drop(columns=['id', 'quantity'], inplace=True)
    
    
    def clean_up_text(text):
        """ Basic cleaning of data like removing HTML tags and lemmatization"""
        
        text = text.lower().strip() # convert to lowercase and strip trailing and ending whitespaces
        text = re.sub('<.*?>', '', text) #remove html tags
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # remove special chars
        stop_words = set(stopwords.words('english')) # remove stopwords that don't provide any semantic information
        text = ' '.join([word for word in text.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # lemmatization of the text
        return text

    cols = ['Description', 'Marketing_Text', 'Typical_Use_Cases'] # only clean the columns containing long sentences
    for col in cols:
        df[col] = df[col].apply(clean_up_text)
        
    # Combine text columns into a single column which will be used to generate the embeddings
    df['combined'] = df['productName'] + " " + df['Product_Category'] + " " + df['Description'] + " " + df['Marketing_Text'] + " " + df['Typical_Use_Cases'] + " " + df['Technical_Attributes']
    
    cleaned_file_path = '../data/cleaned_electronic_devices.csv'
    df.to_csv(cleaned_file_path, index=False, sep=';')
    
    return cleaned_file_path

if __name__ == "__main__":
    preprocess_data('../data/Semantic_Search_Test_Set_Electronic_Devices.csv')


