import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

RAW_PATH = r'C:\Users\DHANUSH C\OneDrive\Desktop\AgriVision_Project\Project\Datasets\cleaned\Final_POV_Crops_cleaned.csv'
PROCESSED_DIR = './processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)

le = LabelEncoder()
df['Crop_encoded'] = le.fit_transform(df['Crop'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
pd.Series(label_map).to_csv(os.path.join(PROCESSED_DIR, 'crop_label_map.csv'))

X = df.drop(['Crop', 'Crop_encoded'], axis=1)
y = df['Crop_encoded']

# Non-stratified split to keep all classes, even those with 1 sample
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

print("Preprocessing complete. Files saved in:", PROCESSED_DIR)