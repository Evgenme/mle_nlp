import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(data):
    # Пример предварительной обработки данных
    data = data.dropna()
    return data

# Load the CSV file into a DataFrame
file_path = 'G:\Praktikum\MLE\Master_mle\data\mqp.csv'
df = pd.read_csv(file_path, names=['claster', 'lable', 'phrase', 'target'])

