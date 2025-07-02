import pandas as pd
import os

RAW_PATH = r"C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Datasets\cleaned\merged_soil_pincode_grouped_cleaned.csv"
PROCESSED_DIR = './processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Select relevant columns for lookup (including Pincode and using _x columns)
lookup_cols = [
    'State', 'District', 'Block', 'Pincode',
    'Total Nitrogen_x', 'Total Phosphorous_x', 'Total Potassium_x', 'Total pH_x'
    # Add more columns if needed, e.g., 'Total OC_x', 'Total EC_x'
]

soil_lookup = df[lookup_cols]

# Save lookup table
soil_lookup.to_csv(os.path.join(PROCESSED_DIR, 'soil_lookup.csv'), index=False)
print("Soil lookup table saved to:", os.path.join(PROCESSED_DIR, 'soil_lookup.csv'))