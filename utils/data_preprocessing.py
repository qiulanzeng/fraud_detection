import pandas as pd

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    # Clean column names
    df.columns = df.columns.str.strip()
    df = df.dropna()
    y = df['Class']  # Adjust to your fraud label column
    X = df.drop(columns=['Class'])
    return X, y