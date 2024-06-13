import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

file_path = '../data/Semantic_Search_Test_Set_Electronic_Devices.csv'
df = pd.read_csv(file_path, delimiter=';')

# drop the id and quantity columns as they don't provide any meaningful semantic information
df.drop(columns=['id', 'quantity'], inplace=True)

# do basic text cleaning like removing html tags and lemmatization
def clean_up_text(text):
    
    text = text.lower().strip()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# clean up the texts in the columns 
cols = ['Description', 'Marketing_Text', 'Technical_Attributes', 'Typical_Use_Cases']
for col in cols:
    df[col] = df[col].apply(clean_up_text)


# save the processed data into new file 
df.to_csv('../data/cleaned_electronic_devices.csv', index=False, sep=';')