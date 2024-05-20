import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Пример предварительной обработки данных
    data = data.dropna()
    return data
