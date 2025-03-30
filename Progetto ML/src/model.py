import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import os

# Caricamento del dataset
def load_data(filepath):
    """Carica il dataset da un file CSV."""
    return pd.read_csv(filepath)

# Funzione per calcolare la qualità di un laptop
def compute_quality(df):
    """Calcola la qualità del laptop in base ai punteggi di RAM, GPU e Storage."""
    df['GPU Score'] = df['GPU'].apply(lambda x: 1 if 'RTX' in x or 'GTX' in x else 0.7 if 'Intel' in x or 'AMD' in x else 0.5)
    df['Storage Score'] = df['Storage'].apply(lambda x: 1 if 'SSD' in x else 0.5 if 'HDD' in x and 'SSD' in x else 0)
    df['Quality'] = (0.3 * df['RAM (GB)'] + 0.2 * df['GPU Score'] + 0.1 * df['Storage Score'])
    df['Quality Level'] = df['Quality'].apply(lambda x: 1 if x >= 0.8 else 2 if x >= 0.6 else 3)
    return df

# Preprocessing dei dati
def preprocess_data(df):
    """Preprocessa i dati e calcola i nuovi fattori, gestisce i dati mancanti."""
    df = compute_quality(df)
    
    # Gestione dati mancanti
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Calcolo della variabile target Overpriced
    avg_price_per_quality = df['Price ($)'].mean() / df['Quality'].mean()
    df['Overpriced'] = (df['Price ($)'] > df['Quality'] * avg_price_per_quality).astype(int)
    
    return df

# Modello di classificazione
def train_model(df):
    """Addestra un modello RandomForestClassifier e salva le metriche come immagini."""
    features = ['RAM (GB)', 'GPU Score', 'Storage Score', 'Quality']
    target = 'Overpriced'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Gestione dello sbilanciamento dei dati con SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot della ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('Dataset/roc_curve.png')
    plt.close()
    
    # Plot della matrice di confusione
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Overpriced', 'Overpriced'], yticklabels=['Non Overpriced', 'Overpriced'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('Dataset/confusion_matrix.png')
    plt.close()
    
    # Plot dell'accuracy
    plt.figure()
    plt.bar(['Accuracy'], [accuracy], color='green')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model Accuracy')
    plt.savefig('Dataset/accuracy.png')
    plt.close()
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Percentuale di sovrapprezzo: {df['Overpriced'].mean() * 100:.2f}%")
    print(f"Percentuale non sovrapprezzo: {100 - df['Overpriced'].mean() * 100:.2f}%")

# Salvataggio del dataset aggiornato
def save_data(df, output_filepath):
    """Salva il dataset aggiornato in un file CSV."""
    df.to_csv(output_filepath, index=False)
    print(f"File salvato: {output_filepath}")

# Main function
def main():
    """Funzione principale per eseguire il flusso di elaborazione e salvataggio."""
    file_path = 'Dataset/laptop_prices.csv'  # Percorso dataset
    output_file = 'Dataset/laptop_price_with_quality.csv'  # Output dataset
    
    df = load_data(file_path)
    df = preprocess_data(df)
    save_data(df, output_file)
    train_model(df)

if __name__ == "__main__":
    main()

