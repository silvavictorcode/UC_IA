import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Carrega o dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Codifica e normaliza o dataset."""
    label_encoder = LabelEncoder()
    df['mfr'] = label_encoder.fit_transform(df['mfr'])
    df['type'] = label_encoder.fit_transform(df['type'])
    
    scaler = StandardScaler()
    df[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass']] = scaler.fit_transform(
        df[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass']])
    
    return df

def save_processed_data(df, output_path):
    """Salva o dataset processado."""
    df.to_csv(output_path, index=False)
