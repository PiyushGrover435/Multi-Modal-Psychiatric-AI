import pandas as pd
import numpy as np
import os

def prepare_dass42_data():
    input_path = r'data/raw/DASS_data_21.02.19/data.csv'
    output_path = r'data/processed/cleaned_dass42.csv'
    
    print(f"Loading raw DASS-42 data from {input_path}...")
    
    # We only need the response columns: Q1A, Q2A ... Q42A
    usecols = [f'Q{i}A' for i in range(1, 43)]
    
    # Read CSV (it's tab separated based on codebook)
    df = pd.read_csv(input_path, sep='\t', usecols=usecols)
    
    # Rename columns to q1...q42
    df.columns = [f'q{i}' for i in range(1, 43)]
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert from 1-4 scale to 0-3 scale for easier scoring
    for col in df.columns:
        df[col] = df[col] - 1
        
    # Calculate a proxy risk score
    # (A real DASS-42 uses specific subscales, but we'll use total score for demonstration)
    total_scores = df.sum(axis=1)
    
    def assign_label(score):
        if score < 40:
            return 'Normal/Mild'
        elif score < 70:
            return 'Moderate'
        else:
            return 'High Risk'
            
    df['label'] = total_scores.apply(assign_label)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path} with {len(df)} records.")

if __name__ == '__main__':
    prepare_dass42_data()
