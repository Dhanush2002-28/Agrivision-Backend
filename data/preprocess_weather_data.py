import pandas as pd
import os

RAW_PATH = r"C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Datasets\cleaned\Weather_Data_cleaned.csv"
PROCESSED_DIR = './processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load weather data
df = pd.read_csv(RAW_PATH)

# Keep only relevant columns and drop duplicates
lookup_cols = ['State', 'year', 'Avg_Temp_C', 'Total_Rainfall_mm']
weather_lookup = df[lookup_cols].drop_duplicates()

# Save for fast lookup
weather_lookup.to_csv(os.path.join(PROCESSED_DIR, 'weather_lookup.csv'), index=False)
print("Weather lookup table saved to:", os.path.join(PROCESSED_DIR, 'weather_lookup.csv'))