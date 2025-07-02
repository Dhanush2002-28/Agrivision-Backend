import os
import pandas as pd

DATASET_FOLDER = './Datasets'
FILES_TO_EXPLORE = [
    'Final_POV_Crops.csv',
    'Soil data.csv',
    'Weather Data.csv'
]

for filename in FILES_TO_EXPLORE:
    file_path = os.path.join(DATASET_FOLDER, filename)
    print(f"\n=== Exploring: {filename} ===")
    try:
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", list(df.columns))
        print("Data types:\n", df.dtypes)
        print("Missing values:\n", df.isnull().sum())
        print("Sample data:\n", df.head(), "\n")
        if filename == 'Weather Data.csv':
            print("States in data:", df['State'].unique())
            print("Year range:", df['year'].min(), "-", df['year'].max())
    except Exception as e:
        print(f"Could not read {filename}: {e}")